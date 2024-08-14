from dm_ultrasound_util import *
from plane_detection.SononetInference import *

def extract_planes_from_video(args, plane_detector):
    print("Extracting planes...")

    class_list = [
        '3VT', '4CH', 'ABDOMINAL', 'BACKGROUND', 
        'BRAIN-CB', 'BRAIN-TV', 'FEMUR', 'KIDNEYS', 
        'LIPS', 'LVOT', 'PROFILE', 'RVOT', 'SPINE-CORONAL', 
        'SPINE-SAGITTAL'
    ]

    # all_frame_list = [[] for _ in range(len(class_list))]
    all_frame_list = []
    detected_class_list = []

    video = cv2.VideoCapture(args.video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize tqdm progress bar
    with tqdm(total=total_frames, desc="Processing Video Frames") as pbar:
        i = 0
        success, frame = video.read()
        
        while success:
            if i % args.process_every_x_frame == 0:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, pred_plane, _ = plane_detector.detect_scan_planes(gray_frame)
                
                all_frame_list.append(gray_frame)
                detected_class_list.append(class_list.index(pred_plane))

            success, frame = video.read()
            i += 1

            # Update progress bar
            pbar.update(1)

    return all_frame_list, detected_class_list

def save_video_frame(args, path, all_frame_list, top_label_list, top_image_indices, num_classes):
    all_frames = np.array(all_frame_list)[top_image_indices]
    path = os.path.join(path, args.method)
    for c in range(num_classes):
        class_path = os.path.join(path, f"{c:02}")
        if not os.path.exists(class_path):
            os.makedirs(class_path)

    assert all_frames.shape[0] == top_label_list.shape[0] == len(top_image_indices)
    for frame, c, idx in tqdm(zip(all_frames, top_label_list, top_image_indices), total=len(top_image_indices)):
        class_path = os.path.join(path, f"{int(c):02}")
        image = Image.fromarray(frame)
        image.save(os.path.join(class_path, f"{idx}.png"))   


class VideoFrameDataset(Dataset):
    def __init__(self, video_frame_list, class_list, transform=None):

        self.images = video_frame_list
        self.labels = class_list
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label 

def main(args):

    args.use_sample_ratio = bool(args.sample_ratio is not None or args.sample_ratio > 0)
    args.Iteration = 2000 if args.method == "dm" else 1000

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True
    args.syn_ce = True if args.method == "idm_ce" else False

    sononet = SononetInference()
    all_frame_list, detected_class_list = extract_planes_from_video(args, sononet)

    channel = 1
    im_size = (args.res, args.res)
    mean = [0.5]
    std = [0.5]
    num_classes = 14

    transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(mean=mean, std=std),
                            transforms.Resize(args.res),
                            transforms.CenterCrop(args.res)])

    class_map = {x:x for x in range(num_classes)}
    dst_train = VideoFrameDataset(all_frame_list, detected_class_list, transform)
    class_map = {x:x for x in range(num_classes)}

    images_all, labels_all, indices_class = build_dataset(dst_train, class_map, num_classes)

    if args.use_sample_ratio:
        n_sample_list = get_sample_syn_label(labels_all, args.sample_ratio, num_classes=num_classes, min_syn=args.min_syn, max_syn=args.max_syn)
        print(sum(n_sample_list))
        print(n_sample_list)
    
    img_class_cnt = [0]*num_classes
    for c in labels_all:
        img_class_cnt[c] += 1

    ignore_class = [c for c in range(num_classes) if n_sample_list[c] >= img_class_cnt[c]]
    print(f"Ignore class: {ignore_class}")


    if args.use_gan:
        mean_tensor = torch.Tensor([0.5])
        std_tensor = torch.Tensor([0.5])
    else:
        mean_tensor = torch.Tensor(mean)
        std_tensor = torch.Tensor(std)
    unnormalize = transforms.Normalize((-mean_tensor / std_tensor).tolist(), (1.0 / std_tensor).tolist())

    gan_model_path = "./gan/models/net_G_relu_64_video_net_new_epoch_49.pth"
    if args.use_gan:
        generator = get_dcgan(args, gan_model_path, ngf=64, channel=channel, display_img=True, unnormalize=unnormalize)
    else:
        generator = None

    if args.method == "dm":
        latents_tmp = run_dm(
            args, 
            indices_class, 
            images_all, 
            channel, 
            num_classes, 
            im_size=im_size, 
            generator=generator, 
            n_sample_list=n_sample_list, 
            is_save_img=True, 
            is_save_latent=True,
            unnormalize=unnormalize,
            ignore_class = ignore_class
        )
    elif args.method in ["idm", "idm_ce"]:
        latents_tmp = run_idm(
            args, 
            indices_class, 
            images_all, 
            labels_all,
            channel, 
            num_classes, 
            im_size=im_size, 
            generator=generator, 
            n_sample_list=n_sample_list, 
            is_save_img=False, 
            is_save_latent=False,
            unnormalize=unnormalize,
            ignore_class = ignore_class
        )

    if args.use_sample_ratio:
        tensor_split = split_tensor_to_list(latents_tmp, n_sample_list)

    embed_list = get_embed_list(args, channel, num_classes, im_size, num_net=10)

    if args.use_sample_ratio:
        mse_latent_dict, _, _ = get_most_similar_img(
            tensor_split, args, indices_class, images_all, is_stack=False, embed_list=embed_list, ret_img_latent=True
        )
    else:
        mse_latent_dict, _ = get_most_similar_img(latents_tmp, args, embed_list=embed_list)

    torch.random.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    top_image_list, top_label_list, top_image_indices = get_top_img(images_all, mse_latent_dict, ignore_class=ignore_class)
    top_image_list = torch.cat(top_image_list)
    top_label_list = torch.tensor(top_label_list)

    for c in ignore_class:
        top_image_indices += indices_class[c]
        top_label_list = torch.cat((top_label_list, torch.tensor([c]*len(indices_class[c]))))

    assert top_label_list.shape[0] == len(top_image_indices)

    save_video_frame(args, args.output_path, all_frame_list, top_label_list, top_image_indices, num_classes)


if __name__ == '__main__':
    if __name__ == '__main__':
        import shared_args

        parser = add_shared_args()

        parser.add_argument('--lr_teacher', type=float, default=0.01, help='learning rate for updating network parameters')
        parser.add_argument('--batch_syn', type=int, default=None, help='batch size for syn data')
        parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
        parser.add_argument('--load_all', action='store_true')
        parser.add_argument('--max_start_epoch', type=int, default=5)
        parser.add_argument('--max_files', type=int, default=None)
        parser.add_argument('--max_experts', type=int, default=None)
        parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
        parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')

        parser.add_argument('--lr_img', type=float, default=1, help='learning rate for pixels or f_latents')
        parser.add_argument('--lr_w', type=float, default=0.001, help='learning rate for updating synthetic latent w')
        parser.add_argument('--lr_lr', type=float, default=1e-06, help='learning rate learning rate')
        parser.add_argument('--lr_g', type=float, default=0.1, help='learning rate for gan weights')

        parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
        parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
        parser_bool(parser, 'net_decay', False)

        parser.add_argument('--nz', type=int, default=100, help='DCGAN latent dimension size')


        parser.add_argument('--net_num', type=int, default=10, help='Number of networks to use')
        parser.add_argument('--fetch_net_num', type=int, default=2, help='Number of networks to fetch')
        parser.add_argument('--train_net_num', type=int, default=2, help='Number of networks to train')
        parser.add_argument('--net_generate_interval', type=int, default=30, help='Interval for network generation')
        parser.add_argument('--net_begin', type=int, default=0, help='Starting value for network ID range')
        parser.add_argument('--net_end', type=int, default=100000, help='Ending value for network ID range')
        parser.add_argument('--aug_num', type=int, default=1, help='Number of augmentations')
        parser.add_argument('--outer_loop', type=int, default=1, help='Number of outer loop iterations')
        parser.add_argument('--inner_loop', type=int, default=1, help='Number of inner loop iterations')
        parser.add_argument('--model_train_steps', type=int, default=10, help='Number of training steps for the model')
        parser.add_argument('--trained_bs', type=int, default=256, help='Batch size for training')

        parser.add_argument('--mismatch_type', type=str, default='l1', help='Type of mismatch to use')
        parser.add_argument('--ij_selection', type=str, default='random', help='Selection method for i and j')

        parser.add_argument('--ce_weight', type=float, default=0.1, help='Weight for cross-entropy loss')
        
        parser.add_argument('--init', type=str, default='random', choices=['random', 'real'],
                                help='whether to initialize the latent using the random initialization or random real image selection.')
        parser.add_argument('--gan_type', type=str, default='dcgan', choices=['dcgan', 'stylegan2'],
                                help='GAN model to use.')
        parser.add_argument('--method', type=str, default='dm', choices=['dm', 'idm', "idm_ce"],
                                        help='Dataset condensation method')
        
        parser.add_argument('--min_syn', type=int, default=10, help='Minimum subset frame per class')
        parser.add_argument('--max_syn', type=int, default=200, help='Maximum subset frame per class')
        parser.add_argument('--sample_ratio', type=float, default=0.01, help='Subset selection ratio')

        parser.add_argument('--process_every_x_frame', type=int, default=1, help='video sampling frequency')
        parser.add_argument('--video_path', type=str, default="./videos/iFIND01622_20Jul2015_1.MP4", help='MP4 input file path')
        parser.add_argument('--output_path', type=str, default='./subset/', help="subset frame output path")

        parser_bool(parser, 'add_variance', False)
        parser_bool(parser, 'use_gan', True)

        args = parser.parse_args()

        main(args)
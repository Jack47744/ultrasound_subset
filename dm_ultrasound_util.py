import time
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torchnet
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

from PIL import Image

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

from collections import defaultdict

import os
from tqdm.autonotebook import tqdm
from ema_pytorch import EMA
import wandb
import random
import copy
import matplotlib.pyplot as plt
import gc
import pickle
from IPython.display import display
import xlsxwriter
from concurrent.futures import ThreadPoolExecutor, as_completed

from networks import * 
from gan.networks import * 
from glad_utils import build_dataset, prepare_latents, get_optimizer_img, get_eval_lrs, eval_loop_v2
from utils import config, get_dataset, get_default_convnet_setting, get_network, get_time, epoch, evaluate_synset, \
    get_eval_pool, ParamDiffAug, DiffAugment, TensorDataset, match_loss

from torchvision.utils import make_grid

import argparse

import warnings

def add_shared_args():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='ultrasound', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='M',
                        help='eval_mode')  # S: the same to training model, M: multi architectures
    parser.add_argument('--num_eval', type=int, default=5, help='the number of evaluating randomly initialized models')
    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')
    parser.add_argument('--save_it', type=int, default=None, help='how often to evaluate')
    parser.add_argument('--epoch_eval_train', type=int, default=100,
                        help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')

    parser.add_argument('--mom_img', type=float, default=0.5, help='momentum for updating synthetic images')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_test', type=int, default=128, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='noise', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')

    parser.add_argument('--save_path', type=str, default='result', help='path to save results')

    parser.add_argument('--space', type=str, default='p', choices=['p', 'wp'])
    parser.add_argument('--res', type=int, default=64, choices=[64, 128, 256, 512], help='resolution')
    parser.add_argument('--layer', type=int, default=12)
    parser.add_argument('--avg_w', action='store_true')

    parser.add_argument('--eval_all', action='store_true')

    parser.add_argument('--min_it', type=bool, default=False)
    parser.add_argument('--no_aug', type=bool, default=False)

    parser.add_argument('--force_save', action='store_true')

    parser.add_argument('--sg_batch', type=int, default=10)

    parser.add_argument('--rand_f', action='store_true')

    parser.add_argument('--logdir', type=str, default='./logged_files')

    parser.add_argument('--wait_eval', action='store_true')

    parser.add_argument('--idc_factor', type=int, default=1)

    parser.add_argument('--rand_gan_un', action='store_true')
    parser.add_argument('--rand_gan_con', action='store_true')

    parser.add_argument('--learn_g', action='store_true')

    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--depth', type=int, default=4)

    parser.add_argument('--special_gan', default=None)

    return parser

def get_sample_syn_label(labels_all, ratio, num_classes=None, min_syn=None, max_syn=None):
    if num_classes:
        train_class_counts = np.bincount(labels_all, minlength=num_classes)
    else:
        _, train_class_counts = np.unique(labels_all.numpy(), return_counts=True)
    num_sample_class = [round(ratio*e) for e in train_class_counts]
    if max_syn is None:
        return np.array(num_sample_class)
    else:
        return np.clip(np.array(num_sample_class), a_min=min_syn, a_max=max_syn)

def get_images(c, n, args, indices_class, images_all, labels_all=None):
    if c is not None:
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle].to(args.device)
    else:
        assert n > 0, 'n must be larger than 0'
        indices_flat = [_ for sublist in indices_class for _ in sublist]
        idx_shuffle = np.random.permutation(indices_flat)[:n]
        return images_all[idx_shuffle].to(args.device), labels_all[idx_shuffle].to(args.device)

def denorm(x, unnormalize, channels=None, w=None ,h=None, resize = False):

    x = unnormalize(x)
    if resize:
        if channels is None or w is None or h is None:
            print('Number of channels, width and height must be provided for resize.')
        x = x.view(x.size(0), channels, w, h)
    return x

def display_latent(latent, num_classes, n_img, args, is_denorm=False, channel=1, im_size=(64, 64), unnormalize=None):
    # reshaped_tensor = latent.view(num_classes*args.ipc, channel, im_size[0], im_size[1])
    reshaped_tensor = latent.view(-1, channel, im_size[0], im_size[1])
    if is_denorm:
        reshaped_tensor = denorm(reshaped_tensor, unnormalize, channel, im_size[0], im_size[1])
    reshaped_tensor = reshaped_tensor.view(num_classes, n_img, channel, im_size[0], im_size[1]).detach().cpu()
    # Plotting
    fig, axs = plt.subplots(n_img, num_classes, figsize=(num_classes*1.5, n_img))
    for i in range(num_classes):
        for j in range(n_img):
            if args.dataset == "ultrasound":
                axs[j, i].imshow(reshaped_tensor[i, j].squeeze(0), cmap='gray')
            else:
                axs[j, i].imshow(reshaped_tensor[i, j].permute(1, 2, 0), )
            axs[j, i].axis('off')
            axs[j, i].set_title(f'Class {i + 1}, Image {j + 1}', fontsize=8)

    plt.tight_layout()
    plt.show()


label_number_to_name_dict = {
    0: '10_3VV', 
    1: '09_4CH', 
    2: '04_ABDOMINAL', 
    3: '13_BACKGROUND', 
    4: '00_BRAIN-CB', 
    5: '01_BRAIN-TV', 
    6: '06_FEMUR',
    7: '05_KIDNEYS', 
    8: '03_LIPS', 
    9: '12_LVOT', 
    10: '02_PROFILE', 
    11: '11_RVOT', 
    12: '07_SPINE-CORONAL', 
    13: '08_SPINE-SAGITTAL'
}

def count_classes(dataloader):
    class_counts = Counter()
    
    for _, labels in tqdm(dataloader):
        class_counts.update(labels.cpu().numpy())
    
    return class_counts

def get_processed_metrics(metric_list_all):
    metric_list_full = []
    conf_mtx_list = []

    for metric_list in metric_list_all:
        metric_dict = metric_list[0]
        metric_list_full.append({i:metric_dict[i] for i in metric_dict if i!='confusion_matrix'})
        conf_mtx_list.append(metric_dict["confusion_matrix"])

    metric_df = pd.DataFrame(metric_list_full)
    precision_arr = np.array(metric_df["precision"].tolist())
    recall_arr = np.array(metric_df["recall"].tolist())
    f1_arr = np.array(metric_df["f1"].tolist())
    accuracy_arr = metric_df["accuracy"].to_numpy()
    conf_mtx_arr = np.array(conf_mtx_list)

    return precision_arr, recall_arr, f1_arr, accuracy_arr, conf_mtx_arr

def process_mean_std_metrics(precision_arr, recall_arr, f1_arr, accuracy_arr, conf_mtx_arr, class_count):
    mean_df = pd.DataFrame({
        "class": [label_number_to_name_dict[i] for i in range(len(class_count))],
        "precision_avg": precision_arr.mean(axis=0),
        "recall_avg": recall_arr.mean(axis=0),
        "f1_avg": f1_arr.mean(axis=0),
        "img_cnt": class_count,
    }).sort_values(by=['class'])

    std_df = pd.DataFrame({
        "class": [label_number_to_name_dict[i] for i in range(len(class_count))],
        "precision_std": precision_arr.std(axis=0),
        "recall_std": recall_arr.std(axis=0),
        "f1_std": f1_arr.std(axis=0),
        "img_cnt": class_count,
    }).sort_values(by=['class'])

    mean_conf_mtx = pd.DataFrame(conf_mtx_arr.mean(axis=0))
    std_conf_mtx = pd.DataFrame(conf_mtx_arr.std(axis=0))

    mean_conf_mtx.rename(columns=label_number_to_name_dict, index=label_number_to_name_dict, inplace=True)
    std_conf_mtx.rename(columns=label_number_to_name_dict, index=label_number_to_name_dict, inplace=True)

    mean_conf_mtx.sort_index(axis=0, inplace=True)
    mean_conf_mtx.sort_index(axis=1, inplace=True)
    std_conf_mtx.sort_index(axis=0, inplace=True)
    std_conf_mtx.sort_index(axis=1, inplace=True)

    mean_acc = accuracy_arr.mean(axis=0)
    std_acc = accuracy_arr.std(axis=0)

    print(f"Accuracy avg:{mean_acc:05f}, std:{std_acc:05f}")

    return mean_df, std_df, mean_conf_mtx, std_conf_mtx, mean_acc, std_acc

def get_excel_data_range(start_row_arr, start_col_arr, height, width):
    start_cell = xlsxwriter.utility.xl_rowcol_to_cell(start_row_arr, start_col_arr)
    end_cell = xlsxwriter.utility.xl_rowcol_to_cell(start_row_arr + height - 1, start_col_arr + width - 1)
    data_range = f'{start_cell}:{end_cell}'
    return data_range

def cond_color_cell(worksheet, start_row_arr, start_col_arr, height, width, max_green=True):
    data_range = get_excel_data_range(start_row_arr, start_col_arr, height, width)
    if max_green:
        worksheet.conditional_format(data_range, {'type': '3_color_scale',
                                        'min_color': "#F8696B",  # Red
                                        'mid_color': "#FFEB84",  # Yellow
                                        'max_color': "#63BE7B"})  # Green
    else:
        worksheet.conditional_format(data_range, {'type': '3_color_scale',
                                          'min_color': "#63BE7B",  # Green
                                          'mid_color': "#FFEB84",  # Yellow
                                          'max_color': "#F8696B"})  # Red
        
def cond_bar_cell(worksheet, start_row_arr, start_col_arr, height, width, color="#63C384"):
    data_range = get_excel_data_range(start_row_arr, start_col_arr, height, width)
    worksheet.conditional_format(data_range, {'type': 'data_bar',
                                          'bar_color': color})

def save_res_dict_excel(args, res_dict, class_count, dir, file_prefix=""):
    for model_eval, metric_test_all in res_dict.items():

        precision_arr, recall_arr, f1_arr, accuracy_arr, conf_mtx_arr = get_processed_metrics(metric_test_all)

        mean_df, std_df, mean_conf_mtx, std_conf_mtx, mean_acc, std_acc = process_mean_std_metrics(
            precision_arr, recall_arr, f1_arr, accuracy_arr, conf_mtx_arr, class_count
        )

        if not os.path.exists(dir):
            os.makedirs(dir)

        file_dir = os.path.join(dir, f"{file_prefix}{args.dataset}_{args.ipc:03d}_{model_eval}.xlsx")
        with pd.ExcelWriter(file_dir, engine='xlsxwriter') as writer:

            workbook  = writer.book
            sheet_name = model_eval
            worksheet = workbook.add_worksheet(sheet_name)
            
            row = 2
            col = 2
            worksheet.write(f'B{row}', 'Mean')
            mean_df.to_excel(writer, sheet_name=sheet_name, startrow=row, startcol=col, index=False)
            cond_color_cell(worksheet, row+1, col+1, mean_df.shape[0], 1, max_green=True)
            cond_color_cell(worksheet, row+1, col+2, mean_df.shape[0], 1, max_green=True)
            cond_color_cell(worksheet, row+1, col+3, mean_df.shape[0], 1, max_green=True)
            cond_bar_cell(worksheet, row+1, col+4, mean_df.shape[0], 1, color="#63C384")
            row += len(mean_df) + 4

            worksheet.write(f'B{row}', 'STD')
            std_df.to_excel(writer, sheet_name=sheet_name, startrow=row, startcol=col, index=False)
            cond_color_cell(worksheet, row+1, col+1, std_df.shape[0], 1, max_green=False)
            cond_color_cell(worksheet, row+1, col+2, std_df.shape[0], 1, max_green=False)
            cond_color_cell(worksheet, row+1, col+3, std_df.shape[0], 1, max_green=False)
            cond_bar_cell(worksheet, row+1, col+4, mean_df.shape[0], 1, color="#63C384")
            row += len(std_df) + 4

            worksheet.write(f'B{row}', 'Confusion Matrix Mean')
            mean_conf_mtx.to_excel(writer, sheet_name=sheet_name, startrow=row, startcol=col, index=True)
            cond_color_cell(worksheet, row+1, col+1, mean_conf_mtx.shape[0], mean_conf_mtx.shape[1], max_green=True)
            row += len(mean_conf_mtx) + 4

            worksheet.write(f'B{row}', 'Confusion Matrix STD')
            std_conf_mtx.to_excel(writer, sheet_name=sheet_name, startrow=row, startcol=col, index=True)
            cond_color_cell(worksheet, row+1, col+1, std_conf_mtx.shape[0], std_conf_mtx.shape[1], max_green=False)
            row += len(std_conf_mtx) + 4

            worksheet.write(f'B{row}', 'Accuracy Mean')
            worksheet.write(f'C{row}', mean_acc)
            row += 1

            worksheet.write(f'B{row}', 'Accuracy STD')
            worksheet.write(f'C{row}', std_acc)
        
        print(f"Save at: {file_dir}")
        
def show(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))

def get_latent_sample(n_sample_list, args, num_classes, channel, im_size=(64, 64)):

    n_latent = int(sum(n_sample_list))

    label_syn = torch.tensor([i for i in range(num_classes) for _ in range(n_sample_list[i])])
    f_latents = None
    if args.use_gan:
        if args.gan_type == "dcgan":
            latents = torch.randn(size=(n_latent, args.nz, 1, 1), dtype=torch.float, requires_grad=False, device=args.device)
        elif args.gan_type == "stylegan2":
            latents = torch.rand(size=(n_latent, args.nz), dtype=torch.float, requires_grad=False, device=args.device)
    else:
        latents = torch.randn(size=(n_latent, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=False, device=args.device)
    
    latents = latents.detach().to(args.device).requires_grad_(True)

    return latents, f_latents, label_syn

def get_latent_ipc(args, num_classes):
    if args.use_gan:
        label_syn = torch.tensor([i*np.ones(args.ipc, dtype=np.int64) for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        latents = torch.randn(size=(num_classes * args.ipc, args.nz, 1, 1), dtype=torch.float, requires_grad=False, device=args.device)
        f_latents = None
        latents = latents.detach().to(args.device).requires_grad_(True)
    else:
        latents, f_latents, label_syn = prepare_latents(channel=channel, num_classes=num_classes, im_size=im_size,
                                                        zdim=zdim, G=G, class_map_inv=class_map_inv, get_images=get_images,
                                                        args=args)
        
    return latents, f_latents, label_syn

def get_latent_sample_class_start_end_idx(c, n_sample_list):
    cumsum_n_sample_list = np.cumsum(n_sample_list)
    cumsum_n_sample_list = np.append(0, cumsum_n_sample_list)
    start = cumsum_n_sample_list[c]
    end = cumsum_n_sample_list[c+1]
    return start, end

def get_latent_sample_class(c, latents, n_sample_list, chunk_size=None, n_chunk=None):
    class_start, class_end = get_latent_sample_class_start_end_idx(c, n_sample_list)

    if chunk_size is None and n_chunk is None:
        return latents[class_start:class_end]
    elif class_end - class_start <= chunk_size:
        return latents[class_start:class_end]
    elif chunk_size is not None and n_chunk is not None:
        start_idx = class_start + chunk_size * (n_chunk)
        end_idx = start_idx + chunk_size
        end_idx = min(end_idx, class_end)
        if start_idx >= class_end:
            raise IndexError("Chunk start index exceeds the boundary of the class segment.")
        return latents[start_idx:end_idx]
    else:
        raise ValueError("Both chunk_size and n_chunk must be specified together or not at all.")



def sample_tensors(tensors, class_counts, n):
    """
    Sample n tensors from each class.

    Parameters:
    tensors (list): List of tensors.
    class_counts (array): Array where each element represents the number of tensors in the corresponding class.
    n (int): Number of tensors to sample from each class.

    Returns:
    dict: A dictionary with class indices as keys and lists of sampled tensors as values.
    """
    sampled_tensors = []
    start_index = 0
    
    for class_index, count in enumerate(class_counts):
        end_index = start_index + count
        class_tensors = tensors[start_index:end_index]
        
        # Ensure we do not sample more than available tensors
        if n > count:
            sampled_tensors.append(class_tensors)
        else:
            sampled_tensors.append(class_tensors[:n])
        
        start_index = end_index
    
    return torch.stack(sampled_tensors)

def get_embed_list(args, channel, num_classes, im_size, num_net=10):
    net_list = [
        get_network(args.model, channel, num_classes, im_size, depth=args.depth, width=args.width).to(args.device).eval() for _ in range(num_net)
    ]

    embed_list = [net.module.embed if torch.cuda.device_count() > 1 else net.embed for net in net_list]
    del net_list

    return embed_list

def get_most_similar_img(
        latents_tmp, 
        args, 
        indices_class, 
        images_all, 
        get_mean_embed_only=False, 
        is_stack=True, 
        embed_list=[], 
        ret_img_latent=False,
        ignore_class=[],
    ):

    if not embed_list:
        net_list = get_embed_list(args, 10)

    mse_latent_dict = defaultdict(list)

    with torch.no_grad():

        latent_embed_mean_list = []

        img_latent_mean_all_list = []

        for c, img_class_idx in enumerate(indices_class):
            torch.cuda.empty_cache()
            gc.collect()

            latent_embed_class_list = []

            img_latent_mean_list = {}
            for img_idx in tqdm(img_class_idx):
                    
                img = images_all[img_idx].to(args.device)
                img = torch.unsqueeze(img, 0)

                img_embed_list = [embed(img) for embed in embed_list]
                img_embed_list = torch.cat(img_embed_list)
                img_embed = torch.mean(img_embed_list, 0)
                img_latent_mean_list[img_idx] = img_embed
            img_latent_mean_all_list.append(img_latent_mean_list)

            for latent_idx, latent in enumerate(tqdm(latents_tmp[c], desc=f"Processing Latents class: {c}")):
                
                latent = torch.unsqueeze(latent, 0)
                latent_embed_list = [embed(latent) for embed in embed_list]
                latent_embed_list = torch.cat(latent_embed_list)
                latent_embed = torch.mean(latent_embed_list, 0)
                latent_embed_class_list.append(latent_embed)

                if get_mean_embed_only:
                    continue

                for img_idx in img_class_idx:
                    img_embed = img_latent_mean_list[img_idx]
                    mse_loss = F.mse_loss(img_embed, latent_embed).item()
                    mse_latent_dict[(c, latent_idx)].append((mse_loss, img_idx))

            latent_embed_class_list = torch.stack(latent_embed_class_list, 0)
            latent_embed_mean_list.append(latent_embed_class_list)
    if is_stack:
        latent_embed_mean_list = torch.stack(latent_embed_mean_list, 0)

    if ret_img_latent:
        return mse_latent_dict, latent_embed_mean_list, img_latent_mean_all_list
    
    return mse_latent_dict, latent_embed_mean_list



def split_tensor_to_list(tensor, class_counts):
    """
    Split a tensor into a list of tensors based on the provided class counts.

    Parameters:
    tensor (torch.Tensor): The input tensor of shape (798, 1, 64, 64).
    class_counts (array): Array where each element represents the number of tensors in the corresponding class.

    Returns:
    list: A list of tensors where each tensor's number matches the counts in class_counts.
    """
    assert tensor.shape[0] == sum(class_counts), "The sum of class counts must match the first dimension of the tensor."

    split_tensors = []
    start_index = 0
    
    for count in class_counts:
        end_index = start_index + count
        split_tensors.append(tensor[start_index:end_index])
        start_index = end_index
    
    return split_tensors

def plot_images_with_similarity(args, all_img_top_k_list, similarity_loss_list, ipc, num_classes, unnormalize):
    # Determine the number of classes

    
    # Create a figure with subplots in a grid: num_classes rows and ipc+1 columns
    fig, axs = plt.subplots(nrows=len(all_img_top_k_list), ncols=all_img_top_k_list[0].shape[0], figsize=(ipc * 2, num_classes * 2))
    
    for i, (images, similarity_losses) in enumerate(zip(all_img_top_k_list, similarity_loss_list)):
        # print(images.shape)
        for j in range(images.shape[0]):

            ax = axs[i, j]

            if args.use_gan:
                images[j] = denorm(torch.unsqueeze(images[j], 0), unnormalize, channels=None, w=None ,h=None, resize = False)
            else:
                images[j] = denorm(torch.unsqueeze(images[j], 0), unnormalize, channels=None, w=None ,h=None, resize = False)

            if images[j].shape[0] == 3:
                img = images[j].permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
                ax.imshow(img.numpy())
            else:

                ax.imshow(images[j].squeeze(0), cmap='gray')

            
            ax.axis('off')  # Turn off axis numbers and ticks
            
            # Annotate the top image with its similarity loss
            if j > 0:  # Skip the first image (latent image)
                ax.set_title(f"Loss: {similarity_losses[j-1]:.5f}", fontsize=25)
            else:
                ax.set_title("Latent", fontsize=25)

    plt.tight_layout()
    plt.show()

def parser_bool(parser, name, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no_' + name, dest=name, action='store_false')
    parser.set_defaults(**{name:default})

def get_img_optimizer(latents, args):
    if args.use_gan:
        optimizer_img = torch.optim.SGD([latents], lr=args.lr_w, momentum=0.5)
    else:
        optimizer_img = torch.optim.SGD([latents], lr=args.lr_img, momentum=0.5)
    return optimizer_img

def number_sign_augment(image_syn, label_syn):
    half_length = image_syn.shape[2]//2
    # import pdb; pdb.set_trace()
    a, b, c, d = image_syn[:, :, :half_length, :half_length].clone(), image_syn[:, :, half_length:, :half_length].clone(), image_syn[:, :, :half_length, half_length:].clone(), image_syn[:, :, half_length:, half_length:].clone()
    a, b, c, d = F.upsample(a, scale_factor=2, mode='bilinear'), F.upsample(b, scale_factor=2, mode='bilinear'), \
        F.upsample(c, scale_factor=2, mode='bilinear'), F.upsample(d, scale_factor=2, mode='bilinear')
    # a, b, c, d = image_syn.clone(), image_syn.clone(), image_syn.clone(), image_syn.clone()
    image_syn_augmented = torch.concat([a, b, c, d], dim=0)
    label_syn_augmented = label_syn.repeat(4)
    return image_syn_augmented, label_syn_augmented

def get_top_img(images_all, mse_latent_dict, ignore_class=[]):
    top_k = 1

    print(f" ----- top k: {top_k} ----- ")
    top_image_list = []
    top_label_list = []
    top_image_indices = []  # List to store the indices of the top images

    for (c, latent_idx) in tqdm(mse_latent_dict):
        if c in ignore_class:
            continue
        k = (c, latent_idx)
        mse_val_list = sorted(mse_latent_dict[k])[:top_k]
        top_img_idx = [e[1] for e in mse_val_list]
        
        # Store the index of the top image (the one with the lowest MSE value)
        top_image_indices.append(top_img_idx[0])
        
        top_imgs = images_all[top_img_idx]
        top_image_list.append(top_imgs)
        top_label_list += [c] * top_k

    return top_image_list, top_label_list, top_image_indices


def plot_embedding(images_all, img_latent_mean_all_list, mse_latent_dict, args, save_path=""):
    top_image_list, top_label_list, top_image_indices = get_top_img(images_all, mse_latent_dict)

    top_class_dict = defaultdict(list)

    for top_img, top_label in zip(top_image_list, top_label_list):
        top_class_dict[top_label].append(top_img)

    for c, img_latent_mean_dict in enumerate(img_latent_mean_all_list):
        img_embed_tensor = torch.stack(list(img_latent_mean_dict.values()), dim=0).cpu()
        tsne = TSNE(n_components=2, random_state=0, perplexity=50, max_iter=300)
        tsne_results = tsne.fit_transform(img_embed_tensor)

        top_indices = [i for i, k in enumerate(img_latent_mean_dict.keys()) if k in top_image_indices]

        n_all_img = tsne_results.shape[0]
        n_top_img = tsne_results[top_indices].shape[0]

        fig, ax = plt.subplots(figsize=(10, 7))
        ax = sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], label=f'All images: {n_all_img} imgs', alpha=0.5)
        ax = sns.scatterplot(x=tsne_results[top_indices, 0], y=tsne_results[top_indices, 1], color='red', label=f'Top images: {n_top_img} imgs')
        ax.set_title(f't-SNE of the image embeddings {label_number_to_name_dict[c]} {args.method}')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.legend()

        if save_path:
            img_save_name = f"tsne_class_{c:02d}_{label_number_to_name_dict[c]}_{args.method}.png"
            fig.savefig(os.path.join(save_path, img_save_name))

        plt.show()


def plot_class_distribution(labels_all):

    unique, counts = np.unique(labels_all.numpy(), return_counts=True)
    dist = list(zip(unique, counts))
    class_indices = [pair[0] for pair in dist]
    counts = [pair[1] for pair in dist]

    # Plot the histogram
    bars = plt.bar(class_indices, counts, color='blue')

    # Add labels and title
    plt.xlabel('Class Index')
    plt.ylabel('Count')
    plt.title('Histogram of Class Counts')
    plt.xticks([])

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, -10, str(int(bar.get_x())), ha='center', va='top')
    plt.show()

    return counts

def get_dcgan(args, model_path, ngf=64, channel=1, display_img=True, unnormalize=None):
    
    nz = 100  # Size of z latent vector (i.e., generator input)
    ngf = 64  # Size of feature maps in generator
    ndf = 64  # Size of feature maps in discriminator
    nc = 1    # Number of channels in the training images

    generator = Generator(args.nz, ngf, channel).to(args.device)
    generator.load_state_dict(torch.load(model_path, map_location=args.device))

    generator = generator.eval()
    for param in generator.parameters():
        param.requires_grad = False
    if display_img:
        fixed_noise = torch.randn(32, args.nz, 1, 1, device=args.device)
        with torch.no_grad():
            fake = generator(fixed_noise)
        show(make_grid(denorm(fake, unnormalize), nrow=8, padding=2, normalize=False, value_range=None, scale_each=False, pad_value=0))
    return generator

def save_latent_images(args, syn_images, latents, unnormalize, it=0):

    with torch.no_grad():
                    
        sample_path = f"./{args.method}_samples/{args.dataset}" if args.use_gan else f"./{args.method}_samples/{args.dataset}_no_gan"

        if not os.path.exists(sample_path):
            os.makedirs(sample_path)

        save_img_path = f"{sample_path}/fake_{it:04d}.png" if not args.add_variance else f"{sample_path}/var_fake_{it:04d}.png"
        if args.use_gan:
            save_image(denorm(syn_images.cpu(), unnormalize).float(), save_img_path)
        else:
            save_image(denorm(latents.cpu(), unnormalize).float(), save_img_path)

def save_latents(args, latents, generator=None, it=0):

    if args.use_gan:
        with torch.no_grad():
            save_latent = generator(latents)
    else:
        save_latent = latents

    latent_path = f"./{args.method}_latents/{args.dataset}" if args.use_gan else f"./{args.method}_latents/{args.dataset}_no_gan"

    if not os.path.exists(latent_path):
        os.makedirs(latent_path)

    if args.use_sample_ratio:
        save_latent_path =  f'{latent_path}/{str(args.sample_ratio).replace(".", "__")}_ori_{it:04d}.pt' if not args.add_variance else f'{latent_path}/var_{str(args.sample_ratio).replace(".", "__")}_ori_{it:04d}.pt'
    else:
        save_latent_path =  f'{latent_path}/{args.ipc}_ori_{it:04d}.pt' if not args.add_variance else f'{latent_path}/var_{args.ipc}_ori_{it:04d}.pt'
    torch.save(save_latent, save_latent_path)
    print(f"Save at {save_latent_path}")


def run_dm(
    args, 
    indices_class,
    images_all, 
    channel, 
    num_classes, 
    im_size=(64, 64), 
    generator=None, 
    n_sample_list=None, 
    is_save_img=False, 
    is_save_latent=False, 
    unnormalize=None,
    ignore_class = [],
):

    run = wandb.init(
        project="GLaD",
        job_type=args.method.upper(),
        config=args
    )

    if args.use_sample_ratio:
        print(f"Sample with ratio of {args.sample_ratio}")
        latents, f_latents, label_syn = get_latent_sample(n_sample_list, args, num_classes=num_classes, channel=channel, im_size=im_size)
    else:
        print(f"Sample with number of {args.ipc}")
        latents, f_latents, label_syn = get_latent_ipc(args, num_classes=num_classes)

    torch.random.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    if args.use_gan:
        optimizer_img = torch.optim.SGD([latents], lr=args.lr_w, momentum=0.5)
    else:
        optimizer_img = torch.optim.SGD([latents], lr=args.lr_img, momentum=0.5)
    print('Hyper-parameters: \n', args.__dict__)
    print('%s training begins'%get_time())

    for it in tqdm(range(args.Iteration+1), desc="Training Progress"):
        net = get_network(args.model, channel, num_classes, im_size, depth=args.depth, width=args.width).to(args.device) # get a random model
        net.train()
        for param in list(net.parameters()):
            param.requires_grad = False

        embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel
        loss_avg = 0

        if args.use_gan:
            with torch.no_grad():
                image_syn_w_grad = generator(latents)
        else:
            image_syn_w_grad = latents

        if args.use_gan:
            image_syn = image_syn_w_grad.detach()
            image_syn.requires_grad_(True)
        else:
            image_syn = image_syn_w_grad

        loss = torch.tensor(0.0).to(args.device)
        for c in range(num_classes):
            if c in ignore_class:
                continue
            img_real = get_images(c, args.batch_real, args, indices_class, images_all).to(args.device)
            if args.use_sample_ratio:
                img_syn = get_latent_sample_class(c, image_syn, n_sample_list).reshape((n_sample_list[c], channel, im_size[0], im_size[1]))
            else:
                img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

            if args.dsa:
                seed = int(time.time() * 1000) % 100000
                img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

            output_real = embed(img_real).detach()
            output_syn = embed(img_syn)

            loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

        optimizer_img.zero_grad()
        loss.backward()

        if args.use_gan:
            latents_detached = latents.detach().clone().requires_grad_(True)
            syn_images = generator(latents_detached)
            syn_images.backward((image_syn.grad,))
            latents.grad = latents_detached.grad
            
        else:
            latents.grad = image_syn.grad.detach().clone()

        optimizer_img.step()
        loss_avg += loss.item()
        loss_avg /= (num_classes)

        wandb.log({
            "Loss": loss_avg
        }, step=it)

        if it%50 == 0 and is_save_img:
            print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))
            if not args.use_gan:
                syn_images = None
            save_latent_images(args, syn_images, latents, unnormalize, it=it)

        if it % 200 == 0 and it > 0 and is_save_latent:
            save_latents(args, latents, generator=generator, it=it)

    torch.cuda.empty_cache()
    gc.collect()
    with torch.no_grad():
        save_latent = generator(latents) if args.use_gan else latents
        return save_latent


def init_real_img(n_sample_list, channel, im_size, args, indices_class, images_all):
    n_latent = int(sum(n_sample_list))
    latents = torch.randn(size=(n_latent, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=False, device=args.device)
    for c, sample_size in enumerate(n_sample_list):
        start_idx, end_idx = get_latent_sample_class_start_end_idx(c, n_sample_list)
        latents[start_idx:end_idx] = get_images(c, sample_size, args, indices_class, images_all).detach().data
    latents = latents.detach().to(args.device).requires_grad_(True)
    return latents

def run_idm(
    args, 
    indices_class, 
    images_all, 
    labels_all,
    channel, 
    num_classes, 
    im_size=(64, 64), 
    generator=None, 
    n_sample_list=None, 
    is_save_img=False, 
    is_save_latent=False,
    unnormalize=None,
    ignore_class = [],
):
    run = wandb.init(
        project="GLaD",
        job_type=args.method.upper(),
        config=args
    )

    if args.use_sample_ratio:
        print(f"Sample with ratio of {args.sample_ratio}")
        latents, f_latents, label_syn = get_latent_sample(n_sample_list, args, num_classes=num_classes, channel=channel, im_size=im_size)
    else:
        print(f"Sample with number of {args.ipc}")
        latents, f_latents, label_syn = get_latent_ipc(args, num_classes=num_classes)

    if not args.use_gan and args.init == "real":
        latents = init_real_img(n_sample_list, channel, im_size, args, indices_class, images_all)
        

    torch.random.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    optimizer_img = get_img_optimizer(latents, args)

    net_num = args.net_num
    net_list = list()
    optimizer_list = list()
    acc_meters = list()

    for net_index in range(3):
        net = get_network(args.model, channel, num_classes, im_size, depth=args.depth, width=args.width).to(args.device)
        net.train()
        if args.net_decay:
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net, momentum=0.9, weight_decay=0.0005)
        else:
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
        optimizer_net.zero_grad()
        net_list.append(net)
        optimizer_list.append(optimizer_net)
        acc_meters.append(torchnet.meter.ClassErrorMeter(accuracy=True))

    criterion = nn.CrossEntropyLoss().to(args.device)
    for it in tqdm(range(args.Iteration+1), desc="Training Iterations"):

        if it % args.net_generate_interval == 0:
            # append and pop net list:
            if len(net_list) == net_num:
                net_list.pop(0)
                optimizer_list.pop(0)
                acc_meters.pop(0)
            net = get_network(args.model, channel, num_classes, im_size, depth=args.depth, width=args.width).to(args.device) # get a random model
            net.train()
            if args.net_decay:
                optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net, momentum=0.9, weight_decay=0.0005)
            else:
                optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            net_list.append(net)
            optimizer_list.append(optimizer_net)
            acc_meters.append(torchnet.meter.ClassErrorMeter(accuracy=True))

        _ = list(range(len(net_list)))
        if len(_[args.net_begin: args.net_end]) > 10:
            _ = _[args.net_begin: args.net_end]
        random.shuffle(_)
        if args.ij_selection == 'random':
            net_index_list = _[:args.train_net_num]
        else:
            raise NotImplemented()
        train_net_list = [net_list[ind] for ind in net_index_list]
        train_acc_list = [acc_meters[ind] for ind in net_index_list]

        embed_list = [net.module.embed if torch.cuda.device_count() > 1 else net.embed for net in train_net_list]

        for _ in range(args.outer_loop):
            loss_avg = 0
            mtt_loss_avg = 0

            dm_loss_avg = 0
            ce_loss_weight = 0

            metrics = {'syn': 0, 'real': 0}
            acc_avg = {'syn':torchnet.meter.ClassErrorMeter(accuracy=True)}

            if args.use_gan:
                with torch.no_grad():
                    image_syn_w_grad = generator(latents)
            else:
                image_syn_w_grad = latents

            if args.use_gan:
                image_syn = image_syn_w_grad.detach()
                image_syn.requires_grad_(True)
            else:
                image_syn = image_syn_w_grad

            ''' update synthetic data '''
            if 'BN' not in args.model or args.model=='ConvNet_GBN': # for ConvNet
                for image_sign, image_temp in [['syn', image_syn]]:
                    loss = torch.tensor(0.0).to(args.device)
                    for net_ind in range(len(train_net_list)):
                        net = train_net_list[net_ind]
                        net.eval()
                        embed = embed_list[net_ind]
                        net_acc = train_acc_list[net_ind]
                        for c in range(num_classes):
                            if c in ignore_class:
                                continue
                            loss_c = torch.tensor(0.0).to(args.device)
                            img_real = get_images(c, args.batch_real, args, indices_class, images_all)
                            if args.use_sample_ratio:
                                img_syn = get_latent_sample_class(c, image_syn, n_sample_list).reshape((n_sample_list[c], channel, im_size[0], im_size[1]))
                            else:
                                img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                            lab_syn = torch.ones((img_syn.shape[0],), device=args.device, dtype=torch.long) * c
                            assert args.aug_num == 1

                            # if args.aug:
                            #     img_syn, lab_syn = number_sign_augment(img_syn, lab_syn)

                            if args.dsa:
                                img_real_list = list()
                                img_syn_list = list()
                                for aug_i in range(args.aug_num):
                                    seed = int(time.time() * 1000) % 100000
                                    img_real_list.append(DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param))
                                    img_syn_list.append(DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param))
                                img_real = torch.cat(img_real_list)
                                img_syn = torch.cat(img_syn_list)

                            output_real = embed(img_real).detach()
                            output_syn = embed(img_syn)

                            dm_loss = torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)
                            loss_c += dm_loss
                            dm_loss_avg += dm_loss.item()


                            logits_syn = net(img_syn)
                            metrics[image_sign] += F.cross_entropy(logits_syn, lab_syn.repeat(args.aug_num)).detach().item()
                            acc_avg[image_sign].add(logits_syn.detach(), lab_syn.repeat(args.aug_num))

                            syn_ce_loss = 0
                            if args.syn_ce:
                                weight_i = net_acc.value()[0] if net_acc.n != 0 else 0
                                syn_ce_loss += (F.cross_entropy(logits_syn, lab_syn.repeat(args.aug_num)) * weight_i)
                                loss_c += (syn_ce_loss * args.ce_weight)
                                ce_loss_weight += (syn_ce_loss * args.ce_weight).item()

                            optimizer_img.zero_grad()
                            loss_c.backward()

                            if args.use_gan:
                                latents_detached = latents.detach().clone().requires_grad_(True)
                                syn_images = generator(latents_detached)
                                syn_images.backward((image_syn.grad,))
                                latents.grad = latents_detached.grad

                            else:
                                latents.grad = image_syn.grad.detach().clone()

                            optimizer_img.step()
                            loss += loss_c.item()

                            del output_real, output_syn, loss_c 
                            torch.cuda.empty_cache()
                            gc.collect()

                        del net, embed
                        torch.cuda.empty_cache()
                        gc.collect()

                    if image_sign == 'syn':
                        loss_avg += loss.item()
            else:
                raise NotImplemented()

            loss_avg /= (num_classes)
            mtt_loss_avg /= (num_classes)
            dm_loss_avg /= (num_classes)
            ce_loss_weight /= (num_classes)
            metrics = {k:v/num_classes for k, v in metrics.items()}

            wandb.log({
                "Total Loss": loss_avg,
                "DM Loss": dm_loss_avg,
                "Weighted CE Loss": ce_loss_weight,
            }, step=it)

            shuffled_net_index = list(range(len(net_list)))
            random.shuffle(shuffled_net_index)
            for j in range(min(args.fetch_net_num, len(shuffled_net_index))):
                training_net_idx = shuffled_net_index[j]
                net_train = net_list[training_net_idx]
                net_train.train()
                optimizer_net_train = optimizer_list[training_net_idx]
                acc_meter_net_train = acc_meters[training_net_idx]
                for i in range(args.model_train_steps):
                    img_real_, lab_real_ = get_images(c=None, n=args.trained_bs, args=args, indices_class=indices_class, images_all=images_all, labels_all=labels_all)
                    real_logit = net_train(img_real_)
                    syn_cls_loss = criterion(real_logit, lab_real_)
                    optimizer_net_train.zero_grad()
                    syn_cls_loss.backward()
                    optimizer_net_train.step()
                    acc_meter_net_train.add(real_logit.detach(), lab_real_)

            if is_save_img and it % 20 == 0:
                if not args.use_gan:
                    syn_images = None
                save_latent_images(args, syn_images, latents, unnormalize, it=it)

            if is_save_latent and it % 50 == 0 and it > 0:
                save_latents(args, latents, generator=generator, it=it)

    torch.cuda.empty_cache()
    gc.collect()
    with torch.no_grad():
        save_latent = generator(latents) if args.use_gan else latents
        return save_latent
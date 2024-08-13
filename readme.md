## Getting Started

First, download our repo:
```bash
git clone https://github.com/Jack47744/ultrasound_subset.git
cd ultrasound_subset
```

To setup an environment, please run

```bash
conda env create -n ultrasound_subset python=3.10.12
conda activate ultrasound_subset
pip install -r requirements.txt
```

## Usage
Below are some example commands to run each method.

### Distillation by Distribution matching
The following command will then create the subset selection from the given MP4 file using distribution matching method:
```bash
python process_video_ultrasound.py --method=dm --video_path={path_to_mp4_file} --output_path={path_to_output_subset_frame}
```

### Distillation by Improved Distribution Matching
The following command will then create the subset selection from the given MP4 file using improved distribution matching method:
```bash
python process_video_ultrasound.py --method=idm --video_path={path_to_mp4_file} --output_path={path_to_output_subset_frame}
```

### Distillation by Improved Distribution Matching with Cross Entropy Regularization
The following command will then create the subset selection from the given MP4 file using improved distribution matching with cross entropy regularization method:
```bash
python process_video_ultrasound.py --method=idm_ce --video_path={path_to_mp4_file} --output_path={path_to_output_subset_frame}
```

### Extra Options
Adding ```--use_gan``` will detemine whether to use DCGAN or not which is significantly different.

Adding ```--process_every_x_frame``` will perform frame sampling of the video which reduces the computation


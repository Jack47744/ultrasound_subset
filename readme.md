# Docker Setup

This repository contains the necessary files to set up and run a Python application in a Docker container. The Docker image is built using a `python:3.10.12-slim-buster` base image.

## Prerequisites

- **Docker**: Make sure Docker is installed on your system. You can download and install Docker from the official website:
  - [Docker Desktop](https://www.docker.com/products/docker-desktop/)



## Getting Started

First, download our repo:
```bash
git clone https://github.com/Jack47744/ultrasound_subset.git
cd ultrasound_subset
```

To build the docker image, please run

```bash
docker build -t process_ultrasound_video .
```

## Usage
Below are some example commands to run each method.

### Distillation by Distribution matching
The following command will then create the subset selection from the given MP4 file using distribution matching method:
```bash
docker run -it --rm \
  --name running-app \
  -v {host_path_to_video_directory}:/usr/src/app/videos \
  -v {host_path_to_output_subset_frame}:/usr/src/app/output \
  process_ultrasound_video --method=dm --video_path "/usr/src/app/videos/{video_file_name}.MP4" --output_path "/usr/src/app/output" 
```

### Distillation by Improved Distribution Matching
The following command will then create the subset selection from the given MP4 file using improved distribution matching method:
```bash
docker run -it --rm \
  --name running-app \
  -v {host_path_to_video_directory}:/usr/src/app/videos \
  -v {host_path_to_output_subset_frame}:/usr/src/app/output \
  process_ultrasound_video --method=idm --video_path "/usr/src/app/videos/{video_file_name}.MP4" --output_path "/usr/src/app/output" 
```

### Distillation by Improved Distribution Matching with Cross Entropy Regularization
The following command will then create the subset selection from the given MP4 file using improved distribution matching with cross entropy regularization method:
```bash
docker run -it --rm \
  --name running-app \
  -v {host_path_to_video_directory}:/usr/src/app/videos \
  -v {host_path_to_output_subset_frame}:/usr/src/app/output \
  process_ultrasound_video --method=idm --video_path "/usr/src/app/videos/{video_file_name}.MP4" --output_path "/usr/src/app/output" 
```

### Extra Options
Adding ```--use_gan``` will detemine whether to use DCGAN or not which is significantly different. It can be ```True``` or ```False```. The default value is ```True```.

Adding ```--process_every_x_frame``` will perform frame sampling of the video which reduces the computation. The value must be an integer. The default value is ```1```.


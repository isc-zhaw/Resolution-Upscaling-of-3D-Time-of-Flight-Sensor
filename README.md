<h2 align="center">Resolution Upscaling of 3D Time-of-Flight Sensor by Fusion with RGB Camera</h2>
  <p align="center">
  <strong>Yannick Waelti</strong>,
  <strong>Matthias Ludwig</strong>,
  <strong>Josquin Rosset</strong>
  <strong>Teddy Loeliger</strong>,
  </p>
  <p align="center">
Institute of Signal Processing and Wireless Communications (ISC),<br>ZHAW Zurich University of Applied Sciences
</p>

3D time-of-flight (3D ToF) cameras enable depth perception but typically suffer from low resolution. To increase the resolution of the 3D ToF depth map, a fusion approach with a high-resolution RGB camera featuring a new edge extrapolation algorithm is proposed, implemented and benchmarked here. Despite the presence of artifacts in the output, the resulting high-resolution depth maps exhibit very clean edges when compared to other state-of-the-art spatial upscaling methods. The new algorithm first interpolates the depth map of the 3D ToF camera and combines it with an RGB image to extract an edge map. The blurred edges of the depth map are then replaced by an extrapolation from neighboring pixels for the final high-resolution depth map. A custom 3D ToF and RGB fusion hardware is used to create a new 3D ToF dataset for evaluating the image quality of the upscaling approach. In addition, the algorithm is benchmarked using the Middlebury 2005 stereo vision dataset. The proposed edge extrapolation algorithm typically achieves an effective upscaling factor greater than 2 in both the x and y directions.

## Setup

### Dependencies

We recommend using the provided docker file to run our code. Use the below commands to build and run the container.

`docker build -t tof_rgb_fusion:1.0 --build-arg USER_ID=$(id -u)   --build-arg GROUP_ID=$(id -g) .`

`docker run --name tof_rgb_fusion --gpus all --mount type=bind,source=/path/to/repository,target=/ToF_RGB_Fusion -dt tof_rgb_fusion:1.0`

Make sure to include the submodules by either cloning the repo with the `--recursive` option or running `git submodule update --init --recursive` if the repo has been cloned already

### Datasets
#### ZHAW-ISC 3D ToF and RGB Fusion
Download the dataset from [Zenodo](https://doi.org/10.5281/zenodo.10732158) and place the files under `data/ZHAW_ISC`

#### Middlebury
From within the dataset directory run the `download_middlebury_2014.sh` script to download the hole filled Middlebury 2005 and the original Middlebury 2014 datasets. To create the downscaled images, run `dataset/create_middlebury_dataset.py` from the repository root.

### Methods
#### DADA Checkpoints
Get the model checkpoints for the DADA approach from their [repository](https://github.com/prs-eth/Diffusion-Super-Resolution/blob/main/README.md#-checkpoints) and extract the contents of the .zip file into the `model_checkpoints/DADA` folder.

#### AHMF
Get the model checkpoints from the official [repository](https://github.com/zhwzhong/AHMF?tab=readme-ov-file) and place the files under `model_checkpoints/AHMF`.
To make all models loadable, change all `kernel_size` in the `UpSampler` and `InvUpSampler` to 5. Also, replace `from collections import Iterable` with `from collections.abc import Iterable`.

#### FDKN
To use the DKN and FDKN models, some changes need to be made to the code from the official [repository](https://github.com/cvlab-yonsei/dkn?tab=readme-ov-file). Add align_corners=True to all calls of `F.grid_sample` if you use a PyTorch version > 1.12
If you get a `CUDNN_STATUS_NOT_SUPPORTED` error, wrap the `F.grid_sample` status in a with `torch.backends.cudnn.flags(enabled=False):` statement

## Evaluation
Run `model_evaluation.py` to get metrics and upscaled depthmaps for different approaches. Methods can be specified with the `-m` option (default: all) and upscaling factors can be specified with `-s` (one or multiple of `x4`, `x8`, `x16` or `x32`).

## Citation
```
@software{Waelti_Efficient_Depth_and,
author = {Waelti, Yannick and Ludwig, Matthias and Rosset, Josquin and Loeliger, Teddy},
license = {MIT},
title = {{Resolution Upscaling of 3D Time-of-Flight Sensor by Fusion with RGB Camera}},
url = {https://github.com/isc-zhaw/Resolution-Upscaling-of-3D-Time-of-Flight-Sensor}
}
```
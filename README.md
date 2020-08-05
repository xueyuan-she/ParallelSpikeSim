# ParallelSpikeSim (PSS)

ParallelSpikeSim (PSS) is a GPU accelerated spiking neural network simulator. 

PSS is first introduced in [paper](https://ieeexplore.ieee.org/abstract/document/8714846) (Author: Xueyuan She, Yun Long and Saibal Mukhopadhyay). Since the introduction, some more functions are added to PSS, including:

- Spiking convolutional neural network
- Frequency-dependent STDP
- Process in memory (ReRAM) hardware simulation


If you use PSS in your work, please cite this [paper](https://ieeexplore.ieee.org/abstract/document/8714846).

## Compile
Include options: -I/usr/include/boost

library options (-l): opencv_highgui, cudadevert, cublas_device, curand, boost_system, boost_filesystem, cudnn, opencv_imgproc, opencv_core

The latest release was tested on Ubuntu 16.04

## Prerequisites
- CUDA Toolkit 9.0
- A GPU with compute capability 5.0 or higher


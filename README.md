# ParallelSpikeSim (PSS)

ParallelSpikeSim (PSS) is a GPU accelerated spiking neural network simulator. 

PSS is first introduced in [paper](https://ieeexplore.ieee.org/abstract/document/8714846) (Author: Xueyuan She, Yun Long and Saibal Mukhopadhyay). Since the introduction, some more functions are added to PSS, including:

- Spiking convolutional neural network
- Frequency-dependent STDP
- Process in memory (ReRAM) hardware simulation
- Heterogeneous Spiking Neural Network

If you use PSS in your work, please cite this [paper](https://ieeexplore.ieee.org/abstract/document/8714846).

## Support of Heterogeneous Spiking Neural Network
Heterogeneous Spiking Neural Network (H-SNN) as described in [paper](https://www.frontiersin.org/articles/10.3389/fnins.2020.615756/full?&utm_source=Email_to_authors_&utm_medium=Email&utm_content=T1_11.5e1_author&utm_campaign=Email_publication&field=&journalName=Frontiers_in_Neuroscience&id=615756) (Author: Xueyuan She, Saurabh Dash, Daehyun Kim and Saibal Mukhopadhyay), is now supported. To run a learning example, choose option 7 upon start up. For inference, use option 8.

## Compile
Include options: boost_1_66_0, opencv4, CImg-2.9.2_pre072920

library options (-l): opencv_highgui, opencv_imgcodecs, cudadevert, cublas, curand, boost_system, boost_filesystem, cudnn, opencv_imgproc, opencv_core

The latest release was tested on Ubuntu 18

## Prerequisites
- CUDA Toolkit 10.0
- A GPU with compute capability 5.0 or higher
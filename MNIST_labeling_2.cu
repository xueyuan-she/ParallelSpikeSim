#include "header.h"
#include <iostream>
#include <string>
#include <fstream>
#include<stdlib.h>
#include <stdio.h>
#include<time.h>
#include<device_functions.h>
#include<cuda.h>
#include<math.h>

using namespace std;
#define MNIST_img_width 28
#define MNIST_img_len 28

__global__ void init_v (float* output_v, int output_neuron_size){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	if(index>output_neuron_size){
		return;
	}
	output_v[index] = 0;
}


__global__ void calculate_v (Neuron *NeuronList, float *img_raw, float *output_v, int output_neuron_size){
	//printf("gpu(update_synapse_counter)");
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	if(index>output_neuron_size){
		return;
	}
	int i = 0;
	while(NeuronList[index].connected_in[i] > 0.1){
		output_v[index] = output_v[index] + NeuronList[index].connected_weight[i]*img_raw[i]*255;

		i++;
	}
	//printf("index:%d_added value is: %f\n", index, output_v[index]);


}
void MNIST_labeling_2(Neuron *NeuronList, float *img_raw, float *output_v, int output_neuron_size){

	int SIZE_PER_SIDE = sqrt(output_neuron_size)+1;
	dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );
	dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));

	float *MNIST_stimulus_freq_device;

	int signal_size = MNIST_img_width*MNIST_img_len;

	cudaMalloc((void **)&MNIST_stimulus_freq_device, signal_size*sizeof(float));
	cudaMemcpy(MNIST_stimulus_freq_device, img_raw, signal_size*sizeof(float), cudaMemcpyHostToDevice);

	int run_time = 2;
	init_v<<<dimGrid, dimBlock>>>(output_v, output_neuron_size);

	for(int i=0;i<run_time;i++){
		calculate_v<<<dimGrid, dimBlock>>>(NeuronList, MNIST_stimulus_freq_device, output_v, output_neuron_size);
	}


	cudaDeviceSynchronize();
	//cudaFree(signal_device);
	cudaFree(MNIST_stimulus_freq_device);


}

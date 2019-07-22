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

#define img_width 64
#define img_len 64

//This function is used to assign image signal values to input neuron

__global__ void switch_off_input (Neuron *NeuronList, int network_size, int start, int end){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	if(index>end||index<start){
		return;
	}
	if(NeuronList[index].type == 4){
		NeuronList[index].state[2] = 0;
	}
}

__global__ void switch_on_input (Neuron *NeuronList, int network_size, int start, int end){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	if(index>end||index<start){
		return;
	}
	if(NeuronList[index].type == 4){
		NeuronList[index].state[2] = 1;
	}
}

__global__ void update_signal (Neuron *NeuronList, float *signal, int network_size, int start, int end){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	if(index>end||index<start){
		return;
	}
	int signal_index = index-start;
	if(NeuronList[index].type == 4){

		NeuronList[index].state[0] = signal[signal_index];
		//printf("No.: %d, signal is: %f \n",index, NeuronList[index].state[0]);
	}
}

void ROI_drive(Neuron *NeuronList, float* image_signal, int network_size, int start_index, int end_index, int function_select){

	int SIZE_PER_SIDE = sqrt(network_size)+1;
	dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );
	dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));
	float *signal_device;

	if(function_select==0){//update signal
		int signal_size = img_width*img_len*3;

		cudaMalloc((void **)&signal_device, signal_size*sizeof(float));
		cudaMemcpy(signal_device, image_signal,signal_size*sizeof(float),cudaMemcpyHostToDevice);

		update_signal<<<dimGrid, dimBlock>>>(NeuronList, signal_device, network_size, start_index, end_index);
		cudaDeviceSynchronize();
		//cudaFree(signal_device);
	}
	else if(function_select==1){//switch on
		switch_on_input<<<dimGrid, dimBlock>>>(NeuronList, network_size, start_index, end_index);

	}
	else if(function_select==2){//switch off
		switch_off_input<<<dimGrid, dimBlock>>>(NeuronList, network_size, start_index, end_index);
	}
	//delete[] image_signal;
	cudaFree(signal_device);
}

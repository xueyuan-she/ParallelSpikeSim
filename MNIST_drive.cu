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

__global__ void update_stimulus (Input_neuron *Input_neuronlist, float *MNIST_stimulus_freq, int network_size, int start, int end){

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	if(index>end||index<start){
		return;
	}
//	printf("Neuron:%d_isType:%d|",index,Input_neuronlist[index].type);

	int signal_index = index-start;
	if(Input_neuronlist[index].type == 4){
		Input_neuronlist[index].state[1] = 1000.0/MNIST_stimulus_freq[signal_index];	//now this is period btw each spike
//		printf("No.: %d, spike interval is: %f, RAW_freq is: %f \n",index, Input_neuronlist[index].state[1], MNIST_stimulus_freq[signal_index]);
	}
}

__global__ void turn_off_stimulus (Neuron *NeuronList, float *MNIST_stimulus_freq, int network_size, int start, int end){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	if(index>end||index<start){
		return;
	}
	//int signal_index = index-start;
	if(NeuronList[index].type == 4){
		NeuronList[index].state[1] = 0;
		//printf("No.: %d, signal is: %f \n",index, NeuronList[index].state[0]);
	}
}
void image_to_in_phase_proptioanl_normalized_old(float *MNIST_stimulus_freq, float *image, int training_set_number, float max_frequency, float min_frequency, int pixel_number){
	for (int img_i=0; img_i<training_set_number; ++img_i){
		int start_index = img_i*pixel_number;
		float sum = 0, mean, stdv=0.0;
		int i;
		for (i=0; i<pixel_number; ++i) sum += image[start_index+i];
		mean = sum/pixel_number;
		for (i=0; i<pixel_number; ++i) stdv += pow((image[start_index+i]-mean), 2);
		for (i=0; i<pixel_number; ++i) MNIST_stimulus_freq[start_index+i]=(image[start_index+i]-mean)/stdv;
		float min = 1000000000000;
		for (i=0; i<pixel_number; ++i){
			if(min>MNIST_stimulus_freq[start_index+i]) min=MNIST_stimulus_freq[start_index+i];
		}
		for (i=0; i<pixel_number; ++i)MNIST_stimulus_freq[start_index+i]=(MNIST_stimulus_freq[start_index+i]+min);
		for(i=0;i<pixel_number;i++){
			MNIST_stimulus_freq[start_index+i] = (max_frequency-min_frequency)*MNIST_stimulus_freq[start_index+i]+min_frequency;
			printf(" %f|", MNIST_stimulus_freq[start_index+i]);
		}

		printf("\n \n");
	}

//    for (int y=0; y<28; ++y) {
//    	    for (int x=0; x<28; ++x) {
//    	      //std::cout << ((one_mnist_img[y*28+x] == 0.0)? ' ' : '*');
//    	      std::cout << std::to_string((MNIST_stimulus_freq[y*28+x])) << ' ';
//    	    }
//    	    std::cout << std::endl;
//    }
//    cout<<"#############inphasepro#############"<<endl;

}

void image_to_in_phase_proptioanl_normalized(float *MNIST_stimulus_freq, float *image, int training_set_number, float max_frequency, float min_frequency, int pixel_number){
	float total_sum = 0.0, total_mean;
	for(int tot_i=0;tot_i<pixel_number*training_set_number;tot_i++){
		total_sum += image[tot_i];
	}
	total_mean = total_sum/(training_set_number*pixel_number);
	for (int img_i=0; img_i<training_set_number; ++img_i){
		int start_index = img_i*pixel_number;
		float sum = 0, mean, stdv=0.0;
		int i;
		for (i=0; i<pixel_number; ++i) sum += image[start_index+i];
		mean = sum/pixel_number;
		float ratio = total_mean/mean;
		//sbprintf("Ration %f|", ratio);
		for (i=0; i<pixel_number; ++i)MNIST_stimulus_freq[start_index+i]=(image[start_index+i]*ratio);
		for(i=0;i<pixel_number;i++){
			MNIST_stimulus_freq[start_index+i] = (max_frequency-min_frequency)*MNIST_stimulus_freq[start_index+i]+min_frequency;
			//printf(" %f|", MNIST_stimulus_freq[start_index+i]);
		}
		//printf("\n \n");
	}

//    for (int y=0; y<28; ++y) {
//    	    for (int x=0; x<28; ++x) {
//    	      //std::cout << ((one_mnist_img[y*28+x] == 0.0)? ' ' : '*');
//    	      std::cout << std::to_string((MNIST_stimulus_freq[y*28+x])) << ' ';
//    	    }
//    	    std::cout << std::endl;
//    }
//    cout<<"#############inphasepro#############"<<endl;

}

void image_to_in_phase_proptioanl(float *MNIST_stimulus_freq, float *image, int training_set_number, float max_frequency, float min_frequency, int pixel_number){

	for(int i=0;i<pixel_number*training_set_number;i++){
		MNIST_stimulus_freq[i] = (max_frequency-min_frequency)*image[i]+min_frequency;
		printf(" %f|", MNIST_stimulus_freq[i]);
	}

//    for (int y=0; y<28; ++y) {
//    	    for (int x=0; x<28; ++x) {
//    	      //std::cout << ((one_mnist_img[y*28+x] == 0.0)? ' ' : '*');
//    	      std::cout << std::to_string((MNIST_stimulus_freq[y*28+x])) << ' ';
//    	    }
//    	    std::cout << std::endl;
//    }
//    cout<<"#############inphasepro#############"<<endl;

}


void MNIST_drive(Neuron *NeuronList, Input_neuron *Input_neuronlist, float *MNIST_stimulus_freq, int network_size, int training_set_number, int start, int end, float max_frequency, float min_frequency, int function_select){


	if(function_select==0){//update signal

		int SIZE_PER_SIDE = sqrt(network_size)+1;
		dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );
		dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));

		float *MNIST_stimulus_freq_device;

		int signal_size = input_image_w*input_image_l*input_image_channel;

		cudaMalloc((void **)&MNIST_stimulus_freq_device, signal_size*sizeof(float));
		cudaMemcpy(MNIST_stimulus_freq_device, MNIST_stimulus_freq,signal_size*sizeof(float),cudaMemcpyHostToDevice);

		update_stimulus<<<dimGrid, dimBlock>>>(Input_neuronlist, MNIST_stimulus_freq_device, network_size, start, end);
		cudaDeviceSynchronize();
		//cudaFree(signal_device);
		cudaFree(MNIST_stimulus_freq_device);
	}

	if(function_select == 1){//change raw img data to frequency_signal
		int pixel_number = input_image_w*input_image_l*input_image_channel;
		float *old_img = MNIST_stimulus_freq;
		image_to_in_phase_proptioanl_normalized(MNIST_stimulus_freq, old_img, training_set_number, max_frequency, min_frequency, pixel_number);
	}

}

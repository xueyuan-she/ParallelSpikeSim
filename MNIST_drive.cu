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

__global__ void sc2_update (Input_neuron *Input_neuronlist, float *MNIST_stimulus_freq, int one_depth_size, int start, int end, int target, bool shifting, bool get_only_one){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	if(index>=end||index<start){
		return;
	}

	int signal_index = index-start;
	if(Input_neuronlist[index].type == 4){
		if(shifting){
			if(index%one_depth_size!=0){
				Input_neuronlist[index].state[1] = Input_neuronlist[index-1].state[1];
			}
		}
		else{
			if(index%one_depth_size==0) {
				int idx = index/one_depth_size;
//				printf(" %d gets updated, idx: %d, for value: %2.0f |", index, idx, MNIST_stimulus_freq[target+idx]);
				float temp_value = MNIST_stimulus_freq[target+idx];
				if(get_only_one) temp_value = temp_value - Input_neuronlist[index].state[1];
				//if(temp_value!=1.0) printf("\ %f %f /", MNIST_stimulus_freq[target+idx], Input_neuronlist[index].state[1]);
				Input_neuronlist[index].state[1] = 20/(temp_value+0.1);//MNIST_stimulus_freq[target+idx];//500/(MNIST_stimulus_freq[target+idx]+0.1);
			}
		}



	}
}


__global__ void time_seq_update_v2 (Input_neuron *Input_neuronlist, int network_size, int start, int end, int target){

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	if(index>=end||index<start){
		return;
	}
//	printf("Neuron:%d_isType:%d|",index,Input_neuronlist[index].type);

	int signal_index = index-start;
	if(Input_neuronlist[index].type == 4){
		if(index==end-1) {
			Input_neuronlist[index].state[1] = 500/(target+0.1);
		}else{
			Input_neuronlist[index].state[1] = Input_neuronlist[index+1].state[1];
		}

//		printf("No.: %d, spike interval is: %f, RAW_freq is: %f \n",index, Input_neuronlist[index].state[1], MNIST_stimulus_freq[signal_index]);
	}
}

__global__ void time_seq_update (Input_neuron *Input_neuronlist, int network_size, int start, int end, int target){

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	if(index>=end||index<start){
		return;
	}
//	printf("Neuron:%d_isType:%d|",index,Input_neuronlist[index].type);

	int signal_index = index-start;
	if(Input_neuronlist[index].type == 4){
		if(index==target) {
			Input_neuronlist[index].state[1] = 2;
		}else{
			if(Input_neuronlist[index].state[1]>0) Input_neuronlist[index].state[1] += 2;
		}

//		printf("No.: %d, spike interval is: %f, RAW_freq is: %f \n",index, Input_neuronlist[index].state[1], MNIST_stimulus_freq[signal_index]);
	}
}

__global__ void reset_stimulus (Input_neuron *Input_neuronlist, int network_size, int start, int end){

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	if(index>=end||index<start){
		return;
	}
//	printf("Neuron:%d_isType:%d|",index,Input_neuronlist[index].type);

	int signal_index = index-start;
	if(Input_neuronlist[index].type == 4){
		Input_neuronlist[index].state[2] = 0;
		Input_neuronlist[index].state[1] = 0;	//now this is period btw each spike
		Input_neuronlist[index].state[4] = 0;
		Input_neuronlist[index].state[3] = 0;
//		printf("No.: %d, spike interval is: %f, RAW_freq is: %f \n",index, Input_neuronlist[index].state[1], MNIST_stimulus_freq[signal_index]);
	}
}

__global__ void update_stimulus (Input_neuron *Input_neuronlist, float *MNIST_stimulus_freq, int network_size, int start, int end){

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	if(index>=end||index<start){
		return;
	}
//	printf("Neuron:%d_isType:%d|",index,Input_neuronlist[index].type);

	int signal_index = index-start;
	if(Input_neuronlist[index].type == 4){
		Input_neuronlist[index].state[1] = 1000.0/(MNIST_stimulus_freq[signal_index]+0.0001);	//now this is period btw each spike
		if (Input_neuronlist[index].state[1]<=0) Input_neuronlist[index].state[1] = 1000;
//		printf("No.: %d, spike interval is: %f, RAW_freq is: %f \n",index, Input_neuronlist[index].state[1], MNIST_stimulus_freq[signal_index]);
	}
}

__global__ void turn_off_stimulus (Neuron *NeuronList, float *MNIST_stimulus_freq, int network_size, int start, int end){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	if(index>=end||index<start){
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
	double total_sum[input_image_channel];
	for(int i=0; i<input_image_channel; i++) total_sum[i] = 0;
	double total_mean[input_image_channel];
	int per_channel_pixel = pixel_number/input_image_channel;
	int channel_flag = -1;
	for(int tot_i=0;tot_i<pixel_number*training_set_number;tot_i++){
		if(tot_i%per_channel_pixel==0) channel_flag ++;
		if (channel_flag>=input_image_channel) channel_flag = 0;
		total_sum[channel_flag] += image[tot_i];
	}
	double denominator = (training_set_number*pixel_number/input_image_channel);
	printf("====Normalization Per Channel Used======: ");

	for(int i=0; i<input_image_channel; i++){
		total_mean[i] = total_sum[i]/denominator;
		printf("total_mean: %f ", total_mean[i]);
	}

//	printf("====Manually override the total pixel number at image to frequency======: ");
//	total_mean = total_sum/training_set_number/28/28;

	for (int img_i=0; img_i<training_set_number; ++img_i){
		int start_index = img_i*pixel_number;
		double sum[input_image_channel];
		for(int i=0; i<input_image_channel; i++) sum[i] = 0;
		double mean[input_image_channel];
		double stdv=0.0;
		int i;
		int channel_flag = -1;
		for (i=0; i<pixel_number; ++i) {
			if(i%per_channel_pixel==0) channel_flag ++;
			if (channel_flag>=input_image_channel) channel_flag = 0;
			sum[channel_flag] += image[start_index+i];
		}
		for(int channel_i=0; channel_i<input_image_channel; channel_i++) mean[channel_i] = sum[channel_i]/(pixel_number/input_image_channel);
		//float ratio = total_mean/mean;
//		printf("Ratio %f|", ratio);
		channel_flag = -1;
		for (i=0; i<pixel_number; ++i){
			if(i%per_channel_pixel==0) channel_flag ++;
			if (channel_flag>=input_image_channel) channel_flag = 0;
			MNIST_stimulus_freq[start_index+i]=(image[start_index+i]*(total_mean[channel_flag]/mean[channel_flag]));
			MNIST_stimulus_freq[start_index+i] = (max_frequency-min_frequency)*MNIST_stimulus_freq[start_index+i]+min_frequency;
			//channel_flag ++;

		}
//		for(i=0;i<pixel_number;i++){
//			MNIST_stimulus_freq[start_index+i] = (max_frequency-min_frequency)*MNIST_stimulus_freq[start_index+i]+min_frequency;
//			if(MNIST_stimulus_freq[start_index+i]>10) printf(" %f|", MNIST_stimulus_freq[start_index+i]);
//		}
		//printf("\n \n");
	}
}

void image_to_in_phase_proptioanl_normalized_0_1(float *MNIST_stimulus_freq, float *image, int training_set_number, float max_frequency, float min_frequency, int pixel_number){
	float total_sum = 0.0, total_mean;
	float global_min = 1000000;
	float global_max = -1000000;
	for(int tot_i=0;tot_i<pixel_number*training_set_number;tot_i++){
		if(image[tot_i]>global_max) global_max = image[tot_i];
		if(image[tot_i]<global_min) global_min = image[tot_i];
		total_sum += image[tot_i];
	}
	float global_gap = global_max - global_min;
	cout<<"global max: "<<global_max<<" and global min: "<<global_min<<endl;
	total_mean = total_sum/(training_set_number*pixel_number);
	for (int img_i=0; img_i<training_set_number; ++img_i){
		int start_index = img_i*pixel_number;
		float sum = 0, mean, stdv=0.0;
		int i;
		for (i=0; i<pixel_number; ++i) sum += image[start_index+i];
		mean = sum/pixel_number;
		float ratio = total_mean/mean;
		//sbprintf("Ration %f|", ratio);
		for (i=0; i<pixel_number; ++i)MNIST_stimulus_freq[start_index+i]=(image[start_index+i]-global_min)/global_gap+0;
		for(i=0;i<pixel_number;i++){
			MNIST_stimulus_freq[start_index+i] = (max_frequency-min_frequency)*MNIST_stimulus_freq[start_index+i]+min_frequency;
//			printf(" %f|", MNIST_stimulus_freq[start_index+i]);
		}
		//printf("\n \n");
	}
}

void image_to_in_phase_proptioanl_normalized_imagenet(float *MNIST_stimulus_freq, float *image, int training_set_number, float max_frequency, float min_frequency, int pixel_number){


	for(int i=0;i<pixel_number*training_set_number;i++){

		if(image[i]>2) image[i]=2;
		if(image[i]<-2) image[i]=-2;

		MNIST_stimulus_freq[i] = (max_frequency-min_frequency)/2+((max_frequency-min_frequency)/4)*image[i];
//		printf(" %f|", MNIST_stimulus_freq[i]);
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

void image_to_in_phase_proptioanl_normalized_cifar(float *MNIST_stimulus_freq, float *image, int training_set_number, float max_frequency, float min_frequency, int pixel_number){
	float mean[3] = {0.4914, 0.4822, 0.4465};
	float std[3] = {0.247, 0.243, 0.261};
	int per_channel_pixel = pixel_number/input_image_channel;
	printf("====Normalization For CIFAR-10 Used======");
	for (int img_i=0; img_i<training_set_number; ++img_i){
		int start_index = img_i*pixel_number;
		int i;
		int channel_flag = -1;

		//float ratio = total_mean/mean;
//		printf("Ratio %f|", ratio);
		channel_flag = -1;
		for (i=0; i<pixel_number; ++i){
			if(i%per_channel_pixel==0) channel_flag ++;
			if (channel_flag>=input_image_channel) channel_flag = 0;
			MNIST_stimulus_freq[start_index+i]=(image[start_index+i]-mean[channel_flag])/std[channel_flag];
			MNIST_stimulus_freq[start_index+i] = (max_frequency-min_frequency)*MNIST_stimulus_freq[start_index+i]+min_frequency;
			//channel_flag ++;
		}
//		for(i=0;i<pixel_number;i++){
//			MNIST_stimulus_freq[start_index+i] = (max_frequency-min_frequency)*MNIST_stimulus_freq[start_index+i]+min_frequency;
//			if(MNIST_stimulus_freq[start_index+i]>10) printf(" %f|", MNIST_stimulus_freq[start_index+i]);
//		}
		//printf("\n \n");
	}

}

void image_to_in_phase_proptioanl(float *MNIST_stimulus_freq, float *image, int training_set_number, float max_frequency, float min_frequency, int pixel_number){
	printf("Not using input normalization___!!");
	for(int i=0;i<pixel_number*training_set_number;i++){
		MNIST_stimulus_freq[i] = (max_frequency-min_frequency)*image[i]+min_frequency;
		//printf(" %f|", MNIST_stimulus_freq[i]);
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
	if(function_select == -2){//change raw img data to frequency_signal, imagenet norm
		int pixel_number = input_image_w*input_image_l*input_image_channel;
		float *old_img = MNIST_stimulus_freq;
		image_to_in_phase_proptioanl_normalized_imagenet(MNIST_stimulus_freq, old_img, training_set_number, max_frequency, min_frequency, pixel_number);
	}

	if(function_select == -1){//change raw img data to frequency_signal
		int pixel_number = input_image_w*input_image_l*input_image_channel;
		float *old_img = MNIST_stimulus_freq;
		image_to_in_phase_proptioanl(MNIST_stimulus_freq, old_img, training_set_number, max_frequency, min_frequency, pixel_number);
	}

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
		//image_to_in_phase_proptioanl_normalized_cifar(MNIST_stimulus_freq, old_img, training_set_number, max_frequency, min_frequency, pixel_number);
		image_to_in_phase_proptioanl(MNIST_stimulus_freq, old_img, training_set_number, max_frequency, min_frequency, pixel_number);
		//image_to_in_phase_proptioanl_normalized(MNIST_stimulus_freq, old_img, training_set_number, max_frequency, min_frequency, pixel_number);
	}

	if(function_select == 2){//reset all input frequency
		int SIZE_PER_SIDE = sqrt(network_size)+1;
		dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );
		dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));
		int signal_size = input_image_w*input_image_l*input_image_channel;
		reset_stimulus <<<dimGrid, dimBlock>>>(Input_neuronlist, network_size, start, end);
		cudaDeviceSynchronize();
	}

}




void MNIST_drive(Neuron *NeuronList, Input_neuron *Input_neuronlist, float *MNIST_stimulus_freq, int network_size, int training_set_number, int start, int end, float max_frequency, float min_frequency, int function_select, int target){
	if(function_select == 1){//for sc2 sequence
		int SIZE_PER_SIDE = sqrt(network_size)+1;
		dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );
		dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));
//        cout<<"current target: "<<target<<endl;
        int one_depth_size = input_image_w*input_image_l;
		sc2_update<<<dimGrid, dimBlock>>>(Input_neuronlist, MNIST_stimulus_freq, one_depth_size, start, end, target, True, False);
		sc2_update<<<dimGrid, dimBlock>>>(Input_neuronlist, MNIST_stimulus_freq, one_depth_size, start, end, target, False, False);
		cudaDeviceSynchronize();
	}


	if(function_select == 2){//reset all input frequency
		//printf("resetting all input neurons\n");
		int SIZE_PER_SIDE = sqrt(network_size)+1;
		dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );
		dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));
		int signal_size = input_image_w*input_image_l*input_image_channel;
//		cout<<"start: "<<start<<", end: "<<end<<endl;
		reset_stimulus <<<dimGrid, dimBlock>>>(Input_neuronlist, network_size, start, end);
		cudaDeviceSynchronize();
	}

	if(function_select == 3){//for time sequence, update one frequency
		int SIZE_PER_SIDE = sqrt(network_size)+1;
		dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );
		dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));
		int signal_size = input_image_w*input_image_l*input_image_channel;
		time_seq_update_v2<<<dimGrid, dimBlock>>>(Input_neuronlist, network_size, start, end, target);
		cudaDeviceSynchronize();
	}

	if(function_select == 4){//for debug: print out input neuron
		int input_neuron_num = input_image_w*input_image_l*input_image_channel;
	    Input_neuron *Input_neuronlist_host = new Input_neuron[input_neuron_num];
	    cudaMemcpy(Input_neuronlist_host,Input_neuronlist,input_neuron_num*sizeof(Input_neuron),cudaMemcpyDeviceToHost);
	    cout<<endl<<"check for input neuron: "<<endl;
	    for(int i=0; i<input_image_channel; i++){
	    	for(int j=0; j<input_image_w*input_image_l; j++){
	    		int idx = i*input_image_w*input_image_l+j;
	    		cout<<(int)Input_neuronlist_host[idx].state[1]<<" ";

	    	}
	    	cout<<endl;
	    }
	    delete[] Input_neuronlist_host;
		cudaDeviceSynchronize();
	}


}


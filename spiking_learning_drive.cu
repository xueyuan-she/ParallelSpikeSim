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
#define HOMEOSTASIS_CONSTANT 150

//currently using LIF for spike learning




__global__ void update_threshold (Neuron *NeuronList, int network_size, float *log_total_spike, float target_frequency, int time){
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	if(index>=network_size){
		return;
	}
	if(NeuronList[index].type==2){
		float frequency_mean = log_total_spike[index]/time;
		float delta_thres = HOMEOSTASIS_CONSTANT*(frequency_mean-target_frequency);
		NeuronList[index].param[1] = NeuronList[index].param[1] + delta_thres;
		//printf("NeuronNo%d:%f] ", index, delta_thres);
	}
	else{
		return;
	}

}
__global__ void lateral_inhibition (Neuron *NeuronList, int network_size, int inhibit_time){
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	if(index>=network_size){
		return;
	}
	if(NeuronList[index].type==4){
		return;
	}
	if(NeuronList[index].state[2]>0.1){
		return;
	}

	//NeuronList[index].state[7] = inhibit_time;	//
	NeuronList[index].state[0] = NeuronList[index].state[0] - 7;//NeuronList[index].param[2];				//change mem potential to reset_value
	//float *result = std::find(std::begin(NeuronList[index].state), std::end(NeuronList[index].state), 123);
	printf("#");
}

__global__ void lateral_inhibition_2 (Neuron *NeuronList, int network_size, int inhibit_time, float start_depth, float end_depth){
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	//printf("%d %d| ", index, network_size);
	if(index>=network_size){
		return;
	}
	if(NeuronList[index].type==4){
		return;
	}
	if(NeuronList[index].state[2]>0.1){
		//printf("******************%d*****************\n", index);
		return;
	}
	if(NeuronList[index].param[7]<start_depth||NeuronList[index].param[7]>end_depth){
		//printf("StartDepth:%f_End:%f__current:%f||", start_depth, end_depth, NeuronList[index].param[7]);
		return;
	}

	//printf("%d | ", index);
	NeuronList[index].state[7] = inhibit_time;	//
	NeuronList[index].state[0] = NeuronList[index].state[0] - 7;//NeuronList[index].param[2];				//change mem potential to reset_value
	//float *result = std::find(std::begin(NeuronList[index].state), std::end(NeuronList[index].state), 123);

}

__global__ void read_learning_output (Neuron *NeuronList, int network_size){
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	if(index>=network_size){
		return;
	}
	//printf("|");
	int i = 0;
		while(NeuronList[index].connected_in[i] > 0.1){
			if(NeuronList[index].connected_weight[i]>1.0){
				printf("connection%d---->%d_has_changed_weight:%f\n",i,index,NeuronList[index].connected_weight[i]);
			}
			i++;
		}

}

__global__ void lateral_inhibition_CNN (Neuron *NeuronList, int network_size, int inhibit_time, float *log_spike){
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	if(index>=network_size){
		//printf("network size: %d, Return on index: %d", network_size, index);
		return;
	}
	if(NeuronList[index].type==4||NeuronList[index].type==5){
		//printf("1.network size: %d, Return on index: %d", network_size, index);
		return;
	}
	if(NeuronList[index].state[2]>0.1){
		//printf("2.network size: %d, Return on index: %d", network_size, index);
		return;
	}
	int fire_neuron_depth = (int)NeuronList[index].param[7];
	if(log_spike[fire_neuron_depth]<0.5){
		//printf("Neuron_index: %d, Return on: depth of %d has log value: %f.\n", index, fire_neuron_depth, log_spike[fire_neuron_depth]);
		//return;
	}
	//NeuronList[index].state[7] = inhibit_time;	//
	NeuronList[index].state[0] = NeuronList[index].state[0] - 3;//NeuronList[index].param[2];				//change mem potential to reset_value
	//float *result = std::find(std::begin(NeuronList[index].state), std::end(NeuronList[index].state), 123);
	//printf("Depth of %f has log value: %f.", fire_neuron_depth, log_spike[fire_neuron_depth]);

}

void spiking_learning_drive(Neuron *NeuronList, int network_size, int inhibit_time, float *log_total_spike, float target_frequency, int time, float *log_spike, int current_layer, int function_select){

	int SIZE_PER_SIDE = sqrt(network_size)+1;
	dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );
	dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));

	int output_neuron_size = OUTPUT_LAYER_NEURON_NUM - 1;

	if(function_select==0){//run lateral_inhibition
		lateral_inhibition<<<dimGrid, dimBlock>>>(NeuronList, output_neuron_size, inhibit_time);
	}
	else if(function_select==1){//run update threshold
		//printf("\nTIME is: %d\n", time);
		update_threshold<<<dimGrid, dimBlock>>>(NeuronList, network_size, log_total_spike, target_frequency, time);
	}
	else if(function_select==2){
		read_learning_output<<<dimGrid, dimBlock>>>(NeuronList, network_size);
	}

	else if(function_select==3){
		lateral_inhibition_CNN<<<dimGrid, dimBlock>>>(NeuronList, network_size, inhibit_time, log_spike);
	}


}

void spiking_learning_drive(Neuron *NeuronList, int network_size, int inhibit_time, float *log_total_spike, float target_frequency, int time, float *log_spike, int current_layer, CNN_struct *CNN_setttings, int function_select){

	int SIZE_PER_SIDE = sqrt(network_size)+1;
	dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );
	dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));

	int output_neuron_size = OUTPUT_LAYER_NEURON_NUM - 1;

	if(function_select==0){//run lateral_inhibition
		lateral_inhibition<<<dimGrid, dimBlock>>>(NeuronList, output_neuron_size, inhibit_time);
	}
	else if(function_select==1){//run update threshold
		//printf("\nTIME is: %d\n", time);
		update_threshold<<<dimGrid, dimBlock>>>(NeuronList, network_size, log_total_spike, target_frequency, time);
	}
	else if(function_select==2){
		read_learning_output<<<dimGrid, dimBlock>>>(NeuronList, network_size);
	}

	else if(function_select==3){
		lateral_inhibition_CNN<<<dimGrid, dimBlock>>>(NeuronList, network_size, inhibit_time, log_spike);
	}
	else if(function_select==4){
		//printf("\nNEW INHIB RUN\n\n");
    	float start_depth = CNN_setttings->layer[current_layer].first_depth_id - 0.1;
    	float end_depth = CNN_setttings->layer[current_layer].last_depth_id + 0.1;
    	//printf("Start_depth: %f, end_depth: %f||", start_depth, end_depth);
		lateral_inhibition_2<<<dimGrid, dimBlock>>>(NeuronList, network_size, inhibit_time, start_depth, end_depth);
	}


}


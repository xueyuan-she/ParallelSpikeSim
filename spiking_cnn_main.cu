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
#define tau 10
#define PSP 1
#define TimeStep 0.04
#define HH_threshold -15
#define test_current 0

/*
test IZH: 1 0 0.15 0.3 -72.14 2.44 30 -60 -14 0 ; 1 0 .
test LIF: 1 2 -70 -55 -75 20 10 10 -70 0 0 ; 1 0 .
test HH: 1 3 0.01 55.17 -72.14 -49.42 1.2 0.36 0.003 -60 0.0529 0 0.3177 0.5961 ; 1 0 .
test signal input: 1 4 1 1 0 1 ; 0 0 .
*/


__global__ void run_spiking_learning (Neuron *NeuronList, Input_neuron *Input_neuronlist, CNN_struct *CNN_setttings, float *random_number, float **input_2d, float **instance_matrix_2d, int current_layer, int network_size, int input_size, float *log_v, float *log_spike, float *log_total_spike, int *spike_flag, int signal_width, float input_float, int time_stamp){
	//printf("its in gpu(main)\n");
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    //printf("=%d-%d=",index,network_size);
    //printf("type is %d\n",NeuronList[1].type);
    //printf("%d\n",index);
    float current_multiplier = 1;//1+1*(current_layer);
    float current_divider = input_float;
    //printf("-%f", current_multiplier/current_divider);

	if(index>=network_size){
		//printf("its wrong!\n");
		return;
	}

	if(index<input_size){
		if(Input_neuronlist[index].type == 4){
				//printf("No.: %d, counter is: %f \n",index, NeuronList[index].state[1]);
				//P.S. state[0] is the signal strength, state[1] is firing frequency;
				//printf("-%d-",index);
				if(Input_neuronlist[index].index >= 0){
					if(Input_neuronlist[index].state[1] == 0){//if the target frequency is zero, turn off
						Input_neuronlist[index].state[2] = 0;
						return;
					}
					if(Input_neuronlist[index].state[2] > 0){	//state[3] is used to count current signal width
						//printf("No.: %d, counter is: %f \n",index, NeuronList[index].state[3]);
						//printf("*%d*",index);
						Input_neuronlist[index].state[3] = Input_neuronlist[index].state[3] + 1;
						//printf("`");
						if(Input_neuronlist[index].state[3]>signal_width){
							Input_neuronlist[index].state[2] = 0;
							Input_neuronlist[index].state[3] = 0;
						}
					}else{
						Input_neuronlist[index].state[4] = Input_neuronlist[index].state[4] + 1;//use state[4] to count the time it has not fired
						//printf("counter is: %f, period is: %f \n", NeuronList[index].state[3], NeuronList[index].state[1]);
						//if(index==210)printf("%f", log_total_spike[index]);
						if((Input_neuronlist[index].state[4])>(Input_neuronlist[index].state[1])){
										//printf("SignalNeuron_%d:counter is: %f, period is: %f \n", index, NeuronList[index].state[4], NeuronList[index].state[1]);
										//log_total_spike[index] = log_total_spike[index] + 1;

									Input_neuronlist[index].state[2] = 1;
									Input_neuronlist[index].state[4] = 0;
						}
					}


				}

			}

			int instance_index = index - CNN_setttings->layer[current_layer].depth_list[0].first_neuron;
			if(instance_index<0||instance_index>CNN_setttings->layer[current_layer].neuron_num){
				return;
			}
			//output_instance_matrix[instance_index] = NeuronList[index].state[2];
			instance_matrix_2d[current_layer][instance_index] = Input_neuronlist[index].state[2];
	}else{

		int neuron_relative_index = index - CNN_setttings->layer[current_layer].depth_list[0].first_neuron;
		index = index - input_size;
		if(NeuronList[index].state[7] > 0.1){
			NeuronList[index].state[7] = NeuronList[index].state[7] - 1;
			current_multiplier = 0;
			//printf("%d is inhibited, mem potential:%d\n",index, NeuronList[index].state[0]);
			//return;
		}//okay

		float start_depth = CNN_setttings->layer[current_layer].first_depth_id - 0.1;
		float end_depth = CNN_setttings->layer[current_layer].last_depth_id + 0.1;
		if(NeuronList[index].param[7]<start_depth||NeuronList[index].param[7]>end_depth){
			//printf("StartDepth:%f_End:%f__current:%f||", start_depth, end_depth, NeuronList[index].param[7]);
			return;
		}
		int convolution_result_index = current_layer - 1;
		if (current_layer==0) convolution_result_index = 0;
		//okay
		//float *input = input_2d[convolution_result_index];
		//float *instance_matrix = instance_matrix_2d[current_layer];

		//float *input_current_matrix;
		//float *output_instance_matrix;


		if(NeuronList[index].type == 2){//run LIF
			if(NeuronList[index].index >= 0){
	//			if(index==30){//this is the membrane potential logger
	//				log_v[time_stamp] = NeuronList[index].state[0];
	//			}
				/*
				if (old_device_neurons[index].state[2] > 0.1){
					old_device_neurons[index].state[2] = old_device_neurons[index].state[2] - 1;
					NeuronList[index].state[2] = NeuronList[index].state[2] - 1;
					if(old_device_neurons[index].state[2]<0.1){
						old_device_neurons[index].state[0] = old_device_neurons[index].param[2];
						log_total_spike[index] = log_total_spike[index] + 1;
						spike_flag[0] = spike_flag[0] + 1;
					}
				}
				*/
				if (NeuronList[index].state[2] > 0.1){
					NeuronList[index].state[0] = NeuronList[index].param[2];
					log_total_spike[index] = log_total_spike[index] + 1;

					//printf("spike_flag[%d]: %d\n", current_layer, spike_flag[current_layer]);
				}
				float Isynapses = test_current;
				Isynapses += input_2d[convolution_result_index][neuron_relative_index];

				//if(Isynapses>0.1)printf("index(%d,%d):%f|", index, neuron_relative_index,Isynapses);

//				if(index>1000){
//					//if(input_2d[convolution_result_index][neuron_relative_index]>0.1) printf("index(%d,%d):%f|", index, neuron_relative_index,input_2d[convolution_result_index][neuron_relative_index]);
//				}
//
//				if(index==1000){
//					if(input_2d[convolution_result_index][23]>0.1) {
//						//for(int ij=0;ij<1000;ij++) printf(" %f ", input_2d[convolution_result_index][ij]);
//					}
//				}

				Isynapses = Isynapses/current_divider;
				Isynapses = Isynapses * current_multiplier;
				/*
				float v_temp_0 = NeuronList[index].param[0] + Isynapses*NeuronList[index].param[4];
				float old_v = old_device_neurons[index].state[0];
				float temp_v = v_temp_0 + (old_v-v_temp_0)*expf(-1*TimeStep/NeuronList[index].param[5]);
				*/
				//if((time_stamp!=0)) log_total_spike[index] = (log_total_spike[index]*(time_stamp-1) + Isynapses)/(time_stamp);

				float temp_v = NeuronList[index].state[0]+TimeStep*(NeuronList[index].param[5]+NeuronList[index].param[0]*NeuronList[index].state[0] + Isynapses*NeuronList[index].param[4]);
				if (LOW_BIT_MEM_POT){
					if((temp_v>NeuronList[index].state[0])<0.07){
						if((temp_v - NeuronList[index].state[0])<0.07) temp_v = NeuronList[index].state[0] + 0.07;
					}
					int intermediate_v = (int)((temp_v + 0.035 + 74.7)*100/7);
					temp_v = (float)intermediate_v*7.0/100.0 -74.7;
				}
				NeuronList[index].state[0] = (temp_v);
				//if (index==800) printf("V:%f,F:%f,lV:%f|",temp_v,Isynapses,NeuronList[index].state[0]);
				if (temp_v>(NeuronList[index].param[1])){

					NeuronList[index].state[0] = NeuronList[index].param[2];
					int fire_neuron_depth = (int)NeuronList[index].param[7];
					spike_flag[current_layer] = spike_flag[current_layer] + 1;
	//				printf("-%d-", fire_neuron_depth);
					log_spike[fire_neuron_depth] = 1;
					NeuronList[index].state[2] = 1.0;
	//				NeuronList[index].state[2] = MID_LAYER_STDP_DURATION + 0.0;
	//				if(index<OUTPUT_LAYER_NEURON_NUM){
	//					NeuronList[index].state[2] = 1.0;
	//				}
					//printf("-fired: %d-", index);
				}
				else{
					//if(index<OUTPUT_LAYER_NEURON_NUM) NeuronList[index].state[2] = 0;
					NeuronList[index].state[2] = 0;
				}
			}
		}

		int instance_index = neuron_relative_index;
		if(instance_index<0||instance_index>CNN_setttings->layer[current_layer].neuron_num){
			return;
		}
		//output_instance_matrix[instance_index] = NeuronList[index].state[2];
		instance_matrix_2d[current_layer][instance_index] = NeuronList[index].state[2];



	}
}

void spiking_cnn_main(Neuron *NeuronList, Input_neuron *Input_neuronlist, CNN_struct *CNN_setttings, float *random_number, float **input, float **instance_matrix, int current_layer, int network_size, int input_size, float *log_v, float *log_spike, float *log_total_spike, int *spike_flag, int signal_width, float input_float, int time_stamp){

	//cout<<"In spiking_cnn_main"<<endl;


	int SIZE_PER_SIDE = sqrt(network_size)+1;
    dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );
    dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));
    run_spiking_learning<<<dimGrid, dimBlock>>>(NeuronList,Input_neuronlist, CNN_setttings, random_number, input, instance_matrix, current_layer, network_size, input_size,log_v, log_spike, log_total_spike, spike_flag, signal_width, input_float, time_stamp);
    //printf("inSpikingLearning");
    //cudaDeviceSynchronize();

}

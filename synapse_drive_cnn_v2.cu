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
#include <curand.h>
#include <curand_kernel.h>

using namespace std;
#define ALPHA_M 0.01
#define ALPHA_P 0.005
#define BETA_P 3
#define BETA_M 3
#define G_MAX 1
#define G_MIN 0
#define STOCH_gamma_pot 0.3
#define STOCH_tau_pot 100
#define STOCH_gamma_dep 0.2
#define STOCH_tau_dep 5

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      //if (abort) exit(code);
   }
}

__global__ void random_synapse_drive_v1 (float *random_number, int rand_number_size, curandState_t *state){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int index = blockId * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
    int connection_index = blockIdx.z * blockDim.z + threadIdx.z;
    index = index*MAX_CONNECTION+connection_index;
    if (index>=rand_number_size) return;
	random_number[index] = (curand(&state[index])%1000)/1000.0;
    //printf("rand_gen_complete\n");
	//if(index==31)printf("The no.%d of random nubmer is %f\n", index, random_number[index]);
	//if(index==31)printf("%f|", random_number[index]);

}

__global__ void update_filter_v2(Neuron *NeuronList, Input_neuron *Input_neuronlist, int index, CNN_struct *CNN_setttings, float **filter, int current_layer, int network_size, int input_neuron_size, int connection_size, long two_power, float half_delta_g_step, float *random_number_list_device, float *random_number_normal_device, int neuron_number_per_layer, int start_index, float StochSTDP_param_1, float StochSTDP_param_2){
//    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
//    int thread_index = blockId * (blockDim.x * blockDim.y * blockDim.z) + threadIdx.z * (blockDim.y * blockDim.x) + threadIdx.y * blockDim.x + threadIdx.x;
//    int connection_blockID =
//    int connection_index = blockIdx.z * blockDim.z + threadIdx.z;
//	int max_connection_index = CNN_setttings->layer[current_layer].conv_setting.filter_depth*CNN_setttings->layer[current_layer].conv_setting.filter_length*CNN_setttings->layer[current_layer].conv_setting.filter_width;
//	if(connection_index>max_connection_index){
//		return;
//	}
//    int neuron_number_per_layer = CNN_setttings->layer[current_layer].depth * CNN_setttings->layer[current_layer].width * CNN_setttings->layer[current_layer].length;
//    int connection_index = thread_index%(neuron_number_per_layer);
//    int index = thread_index/neuron_number_per_layer;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int connection_index = blockId * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
    if(connection_index>connection_size) return;

	//index = index + start_index;
	int random_index = index*MAX_CONNECTION+connection_index;// - start_index;
	//if(connection_index==1) printf("#%d", index);

	if(NeuronList[index].type==4||NeuronList[index].type==5){//if the post-synapse neuron is input-signal-neuron, jump over
		return;
	}
	//printf("-%d",index);
	//printf("-%d",index);
    //if(index==1000) printf("-%d",connection_index);
//	if((NeuronList[index].param[7]-current_layer)>0.01||(NeuronList[index].param[7]-current_layer)<-0.01){
//		printf("param_7 is: %f", NeuronList[index].param[7]);
//		return;
//	}


	//if(connection_index==1&&index<1001)printf("%d|",index);
    //float ALPHA_P_layer = ALPHA_P*(1+2*current_layer);
    //float ALPHA_M_layer = ALPHA_M*(1+2*current_layer);
    float ALPHA_P_layer = ALPHA_P;
    float ALPHA_M_layer = ALPHA_M;
	//int neuron_relative_index = index - CNN_setttings->layer[current_layer].depth_list[0].first_neuron;
	//int number_of_neurons_per_depth = CNN_setttings->layer[current_layer].depth_list[0].total_neuron_num;

	//int current_depth = neuron_relative_index/number_of_neurons_per_depth + CNN_setttings->layer[current_layer].first_depth_id - 1;
	int current_depth = NeuronList[index].param[7] - CNN_setttings->layer[current_layer].first_depth_id;
	if(NeuronList[index].connected_in[connection_index] < 0.1){
		return;
	}
	//if(connection_index==1)printf("-%d", current_depth);
	//for debug
//	for(int ii=0; ii<MAX_CONNECTION; ii++){
//		if(NeuronList[index].connected_in[ii] > 0.1){
//			int filter_index_db = current_depth*connection_size+ii;
////			if(current_layer==2) printf("|%d_%d|", current_depth, connection_size);
//			filter[current_layer-1][filter_index_db] += 0.02;
//		}
//	}
	//printf("**[%d]**neuron_relative_index:%d__Current_depth:%d||", number_of_neurons_per_depth, neuron_relative_index, current_depth);
	//printf("*");

	float delta_g = 0;
	if(NeuronList[index].state[2]>0.1){//if post-synapse neuron fired

			//printf("| %d fired", index);
			//if(index==1000) printf("%d!", connection_index);

				if(NeuronList[index].connected_in[connection_index] > 0.1){

					int connected_in = NeuronList[index].connected_in[connection_index] - 1;
					float pre_neuron_state_1;
					float pre_neuron_state_2;
					float pre_neuron_state_3;
					float pre_neuron_state_4;
					int pre_neuron_type;

					if(NeuronList[index].connected_in[connection_index]<=input_neuron_size){
						pre_neuron_state_1 = Input_neuronlist[connected_in].state[1];
						pre_neuron_state_2 = Input_neuronlist[connected_in].state[2];
						pre_neuron_state_3 = Input_neuronlist[connected_in].state[3];
						pre_neuron_state_4 = Input_neuronlist[connected_in].state[4];

						pre_neuron_type = Input_neuronlist[connected_in].type;
					}else{
						connected_in = connected_in - input_neuron_size;
						pre_neuron_state_1 = NeuronList[connected_in].state[1];
						pre_neuron_state_2 = NeuronList[connected_in].state[2];
						pre_neuron_state_3 = NeuronList[connected_in].state[3];
						pre_neuron_state_4 = NeuronList[connected_in].state[4];

						pre_neuron_type = NeuronList[connected_in].type;
					}
//					printf("$%d$", pre_neuron_state_2);
					if((pre_neuron_state_2 > 0.1)&&(pre_neuron_type!=5)){//if pre-neuron fired
						//printf("-%d_%f-", connected_in, pre_neuron_state_2);
						if(LOW_BIT_TRAINING){
							if(LOW_BIT_NUM <= 8){
								delta_g = half_delta_g_step*2;
							}else{
								if(STOCHASTIC_ROUNDING){
									int32_t fixed = (int32_t)(delta_g * (two_power+0.0) / (LOW_BIT_NUM+0.0));
									float delta_g_truncated = LOW_BIT_NUM*((fixed+0.0)/two_power);
									float rounding_up_prob = (delta_g - delta_g_truncated)/(2*half_delta_g_step);
									if(random_number_list_device[random_index]<rounding_up_prob) {
										delta_g = delta_g_truncated+half_delta_g_step;
									}else{
										delta_g = delta_g_truncated;
									}

								}else{
									delta_g += half_delta_g_step;
									int32_t fixed = (int32_t)(delta_g * (two_power+0.0) / (LOW_BIT_NUM+0.0));
									delta_g = LOW_BIT_NUM*((fixed+0.0)/two_power);
								}
							}
						}else{
							delta_g = ALPHA_P_layer*__expf(-1*BETA_P*((NeuronList[index].connected_weight[connection_index]-G_MIN)/(G_MAX-G_MIN)));	//use hardware implemetation for exp
						}
						if(DEVICE_VARIATION){
							delta_g = (1+random_number_normal_device[random_index]) * delta_g;
						}
						float StochSTDP_tau = STOCH_tau_pot;
						if(FREQUENCY_DEPENDED_STDP){
							//first get period between spikes, convert back to hertz, /1000, then 1 over
							//for 1Hz-22Hz
							float input_freq = 1000/pre_neuron_state_1;
							float phi = 0.6;
							StochSTDP_tau = StochSTDP_tau*(1+phi*(input_freq-1)/(22-1));
							//printf("input_freq is: %f, tau is: %f, phi is %f\n", input_freq, StochSTDP_tau, phi);
						}

						if(STOCHASTIC_STDP){
							float prob = StochSTDP_param_1*__expf(-1*pre_neuron_state_3/StochSTDP_tau);
							if(random_number_list_device[random_index]>prob) {
								delta_g = 0;
							}
						}



					}else{

						if(LOW_BIT_TRAINING){
							if(LOW_BIT_NUM <= 8){
								delta_g = half_delta_g_step*2;
							}else{
								if(STOCHASTIC_ROUNDING){
									int32_t fixed = (int32_t)(delta_g * (two_power+0.0) / (LOW_BIT_NUM+0.0));
									float delta_g_truncated = LOW_BIT_NUM*((fixed+0.0)/two_power);
									float rounding_up_prob = (delta_g - delta_g_truncated)/(2*half_delta_g_step);
									if(random_number_list_device[random_index]<rounding_up_prob) {
										delta_g = delta_g_truncated+half_delta_g_step;
									}else{
										delta_g = delta_g_truncated;
									}

								}else{
									delta_g += half_delta_g_step;
									int32_t fixed = (int32_t)(delta_g * (two_power+0.0) / (LOW_BIT_NUM+0.0));
									delta_g = LOW_BIT_NUM*((fixed+0.0)/two_power);
								}
							}
						}else{
							delta_g = ALPHA_M_layer*__expf(-1*BETA_M*((G_MAX-NeuronList[index].connected_weight[connection_index])/(G_MAX-G_MIN)));
						}
						if(DEVICE_VARIATION){
							//printf("-%f", random_number_normal_device[random_index*MAX_CONNECTION+connection_index]);
							delta_g = (1+random_number_normal_device[random_index]) * delta_g;

						}

						float StochSTDP_tau = STOCH_tau_dep;
						if(FREQUENCY_DEPENDED_STDP){
							//first get period between spikes, convert back to hertz, /1000, then 1 over
							//for 1Hz-22Hz
							float input_freq = 1000/pre_neuron_state_1;
							float phi = 0.6;
							StochSTDP_tau = StochSTDP_tau*(1+phi*(input_freq-1)/(22-1));
							//if(input_freq>15)
							//printf("input_freq is: %f, tau is: %f \n", input_freq, StochSTDP_tau);
						}

						if(STOCHASTIC_STDP){
							float prob = StochSTDP_param_2*__expf(pre_neuron_state_4/StochSTDP_tau);
							if(random_number_list_device[random_index]>prob) {
								//if(random_index>800&&random_index<900) printf("%f|", random_number_list_device[random_index*MAX_CONNECTION+connection_index]);
								delta_g = 0;
							}
						}
						delta_g = -1*delta_g;
					}
					int filter_index = current_depth*connection_size+connection_index;
					//if(connection_index==1&&index<1001)printf("%d|",filter_index);
					filter[current_layer-1][filter_index] += delta_g;
					NeuronList[index].connected_weight[connection_index] = filter[current_layer-1][filter_index];
		//			printf("%f-", filter[current_layer-1][filter_index]);

				}
			}


}

__global__ void log_all_fired(Neuron *NeuronList, Input_neuron *Input_neuronlist, CNN_struct *CNN_settings, float **filter, int current_layer, int start_index, int connection_size,float *random_number, float *random_number_normal_device, int network_size, int input_neuron_size, int filter_size, long two_power, float half_delta_g, int neuron_number_per_layer, float StochSTDP_param_1, float StochSTDP_param_2){
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int index = blockId * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
	//printf("%d|", index);
	if(index>=network_size){
		return;
	}

	if(NeuronList[index].type==4||NeuronList[index].type==5){//if the post-synapse neuron is input-signal-neuron, jump over
		return;
	}
	float start_depth = CNN_settings->layer[current_layer].first_depth_id - 0.1;
	float end_depth = CNN_settings->layer[current_layer].last_depth_id + 0.1;
	if(NeuronList[index].param[7]<start_depth||NeuronList[index].param[7]>end_depth){
		return;
	}
	if(NeuronList[index].state[2]>0.1){
		//printf("%d|", index);
    	int SIZE_PER_SIDE = sqrt((float)filter_size)+1;
		dim3 dimBlock(4,4);
		dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1) );


		update_filter_v2<<<dimGrid, dimBlock>>>(NeuronList, Input_neuronlist, index, CNN_settings, filter, current_layer, network_size, input_neuron_size, filter_size, two_power, half_delta_g, random_number, random_number_normal_device,  neuron_number_per_layer, start_index, StochSTDP_param_1, StochSTDP_param_2);


	}
}

__global__ void init_indicator(char *spike_indicator, int network_size){
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	if(index>=network_size){
		return;
	}
	//printf("%d|", index);
	spike_indicator[index] = 0;

}

__global__ void list_fired_index(int *fired_index, char *spike_indicator, int total_spike, int network_size){
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int index = blockId * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

	if(index==1){
		int fired_count = 0;
		for (int i=1; i<network_size; i++){
			if(spike_indicator[i]){
				fired_index[fired_count] = i;
				if(fired_count>total_spike) printf("Error in spike counting\n");
			}
		}
	}
}


//__global__ int sum_spiked(int spike_counter, char *spike_indicator, int network_size){
//	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
//	int index = blockId * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
//
//	if(index==1){
//		for(int i=0; i<network_size; i++){
//			if(spike_indicator[i]) spike_counter ++;
//		}
//	}
//}

void synapse_drive_cnn_v2(Neuron *NeuronList, Input_neuron *Input_neuronlist, CNN_struct *host_CNN_settings, CNN_struct *CNN_settings, float **filter, int current_layer, int network_size, int input_neuron_size, int syn_timer_max, int connection_size, float *random_number, float *random_number_normal_device, curandState_t *state, float StochSTDP_param_1, float StochSTDP_param_2){

	//first make an array of fired neuron index, and sum up the total number of fired neuron
	int SIZE_PER_SIDE = sqrt(network_size)+1;
	dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );
	dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));


//	dim3 sum_block(1,1);
//	dim3 pre_process_grid(1, 1);
//	int *spike_counter;
//	cudaMalloc((void *)&spike_indicator,sizeof(int));
//	sum_spiked(spike_counter, *spike_indicator, network_size);
//	printf("\n total fired:%d \n", total_spike);


	//int SIZE_PER_SIDE = sqrt(network_size*MAX_CONNECTION)+1;
	int neuron_number_per_layer = host_CNN_settings->layer[current_layer].depth * host_CNN_settings->layer[current_layer].width * host_CNN_settings->layer[current_layer].length;
	int filter_size_per_depth = host_CNN_settings->layer[current_layer].conv_setting.filter_length * host_CNN_settings->layer[current_layer].conv_setting.filter_width* host_CNN_settings->layer[current_layer].conv_setting.filter_depth;
	int start_index = host_CNN_settings->layer[current_layer].depth_list[0].first_neuron;
	//printf("Neuron#is:%d, filterSizeIs:%d", neuron_number_per_layer, filter_size_per_depth);
	//int SIZE_ALONG_Z = 784;
	//int SIZE_PER_SIDE = std::pow(neuron_number_per_layer*filter_size_per_depth, 1.0/3) + 1;

	//dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock, ThreadsPerBlock);
	//int SIZE_PER_SIDE = sqrt(network_size)+1;
	//int SIZE_PER_SIDE = sqrt(neuron_number_per_layer)+1;
	//dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1), (filter_size_per_depth/dimBlock.z+1));
	/*
	float *random_number_list_device;
	curandState_t *states;

	if(STOCHASTIC_STDP){
		int rand_numb_size = SPIKING_NEURON_NUM*MAX_CONNECTION;
		cudaMalloc((void **)&random_number_list_device,rand_numb_size*sizeof(float));
		cudaMalloc((void **)&states, rand_numb_size * sizeof(curandState_t));
		synapse_rand_gen(rand_numb_size, random_number_list_device, states);
	}
	*/
	long two_power;
	if(StochSTDP_param_1<=0){
		StochSTDP_param_1 = STOCH_gamma_pot;
	}

	if(StochSTDP_param_2<=0){
		StochSTDP_param_2 = STOCH_gamma_dep;
	}

	float half_delta_g;
	switch (LOW_BIT_NUM){

					case 2: two_power = TWO_POWER_2;
					break;
					case 4: two_power = TWO_POWER_4;
					break;
					case 8: two_power = TWO_POWER_8;
					break;
					case 16: two_power = TWO_POWER_16;
					break;
					case 32: two_power = TWO_POWER_32;
					break;

	}
	half_delta_g = 0.5/two_power;

	//printf("\n \n Updating filter\n");
	//update_synapse_counter<<<dimGrid, dimBlock>>>(NeuronList, network_size, syn_timer_max, connection_size);

	//update_filter_NOSTOCH_NOLOWBIT<<<dimGrid, dimBlock>>>(NeuronList, CNN_settings, filter, current_layer, network_size, connection_size, two_power, half_delta_g, random_number, neuron_number_per_layer, StochSTDP_param_1, StochSTDP_param_2);

	log_all_fired<<<dimBlock, dimGrid>>>(NeuronList, Input_neuronlist, CNN_settings, filter, current_layer, start_index, connection_size, random_number, random_number_normal_device, network_size, input_neuron_size, filter_size_per_depth, two_power, half_delta_g, neuron_number_per_layer, StochSTDP_param_1, StochSTDP_param_2);
}

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

__global__ void inhibition_through_depth(Neuron *NeuronList, int network_size, int input_size){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	if(index>=network_size){
		//printf("its wrong!\n");
		return;
	}
	if(index>=input_size){
		index = index - input_size;
		if(NeuronList[index].type == 2){//run LIF
			if(NeuronList[index].state[7]<-3){
				NeuronList[index].state[7] = 10; //inhibition time: 10
				NeuronList[index].state[0] = NeuronList[index].param[2] + 0.5*(NeuronList[index].param[1]-NeuronList[index].param[2]);//NeuronList[index].state[0] - 0.5*(NeuronList[index].param[1]-NeuronList[index].param[2]);

			}
		}
	}
}

__global__ void run_spiking_learning_event_based_input (Neuron *NeuronList, Input_neuron *Input_neuronlist, Event_Camera_Input *events, int event_cnt, CNN_struct *CNN_setttings, float *random_number, float **input_2d, \
		float **instance_matrix_2d, int current_layer, int network_size, int input_size, float *log_v, float *log_spike, float *log_total_spike, int *spike_flag, \
		int signal_width, float input_float, int time_stamp, bool enable_inhibition){
	//printf("its in gpu(main)\n");
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    //printf("=%d-%d=",index,network_size);
    //printf("type is %d\n",NeuronList[1].type);
    //printf("%d\n",index);
    float current_multiplier = 2;//+5*(current_layer-1);
    float current_divider = input_float;
    //printf("-%f", current_multiplier/current_divider);
    //if(time_stamp>0) index = (index + time_stamp)%network_size;
	if(index>=network_size){
		//printf("its wrong!\n");
		return;
	}


//	float start_depth = CNN_setttings->layer[current_layer].first_depth_id - 0.1;
//	float end_depth = CNN_setttings->layer[current_layer].last_depth_id + 0.1;
//	if(NeuronList[index].param[7]<start_depth||NeuronList[index].param[7]>end_depth){
//		return;
//	}
//	index = index + CNN_setttings->layer[current_layer].depth_list[0].first_neuron;
	Event_Camera_Input this_event = events[event_cnt];

	int event_target_index = this_event.loc_x+input_image_w*this_event.loc_y;
	if (this_event.sign) event_target_index += input_image_w*input_image_l;

	if(index<input_size){
		if(current_layer>0) return;
		if(Input_neuronlist[index].type == 4){
				//printf("No.: %d, counter is: %f \n",index, NeuronList[index].state[1]);
				//P.S. state[0] is the signal strength, state[1] is firing frequency;
				//printf("-%d-",index);
				if(Input_neuronlist[index].index >= 0){

					if(index==event_target_index){
									//printf("SignalNeuron_%d:counter is: %f, period is: %f \n", index, NeuronList[index].state[4], NeuronList[index].state[1]);
									//log_total_spike[index] = log_total_spike[index] + 1;
						log_total_spike[index] = log_total_spike[index] + 1;
						Input_neuronlist[index].state[2] = 1;
						Input_neuronlist[index].state[4] = 0;
						Input_neuronlist[index].state[3] = -1;
					}

					if(Input_neuronlist[index].state[2] > 0){	//state[2] is used to indicate fired

						//printf("No.: %d, counter is: %f \n",index, NeuronList[index].state[3]);
						//printf("*%d*",index);
						Input_neuronlist[index].state[3] = Input_neuronlist[index].state[3] + 1;//state[3] is used to count current signal width
						//printf("`");
						if(Input_neuronlist[index].state[3]>signal_width){
							Input_neuronlist[index].state[2] = 0;
							Input_neuronlist[index].state[3] = 0;
						}
					}else{
						Input_neuronlist[index].state[4] = Input_neuronlist[index].state[4] + 1;//use state[4] to count the time it has not fired
						//printf("counter is: %f, period is: %f \n", NeuronList[index].state[3], NeuronList[index].state[1]);
						//if(index==210)printf("%f", log_total_spike[index]);

					}


				}

			}

			int instance_index = index - CNN_setttings->layer[current_layer].depth_list[0].first_neuron;
			if(instance_index<0||instance_index>CNN_setttings->layer[current_layer].neuron_num){
				return;
			}
			//output_instance_matrix[instance_index] = NeuronList[index].state[2];
			instance_matrix_2d[current_layer][instance_index] = Input_neuronlist[index].state[2] * 8;
			//instance_matrix_2d[current_layer][instance_index] = 2;
			//printf(" %f", instance_matrix_2d[current_layer][instance_index]);
	}
}

__global__ void run_time_seq (Neuron *NeuronList, Input_neuron *Input_neuronlist, CNN_struct *CNN_setttings, float *random_number, float **input_2d, \
		float **instance_matrix_2d, int current_layer, int network_size, int input_size, float *log_v, float *log_spike, float *log_total_spike, int *spike_flag, \
		int signal_width, float input_float, int spike_target, int teaching_mode){
	//printf("its in gpu(main)\n");
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    //printf("=%d-%d=",index,network_size);
    //printf("type is %d\n",NeuronList[1].type);
    //printf("%d\n",index);
    float current_multiplier = 1;//+5*(current_layer-1);
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
		float start_depth = CNN_setttings->layer[current_layer].first_depth_id - 0.1;
		float end_depth = CNN_setttings->layer[current_layer].last_depth_id + 0.1;
		if(NeuronList[index].param[7]<start_depth||NeuronList[index].param[7]>end_depth){
			//printf("StartDepth:%f_End:%f__current:%f||", start_depth, end_depth, NeuronList[index].param[7]);
			return;
		}
		int convolution_result_index = current_layer - 1;
		if (current_layer==0) convolution_result_index = 0;

		if(NeuronList[index].type == 2){//run LIF
			if(NeuronList[index].index >= 0){

				if(teaching_mode){
					if(index==spike_target){
						//printf("%d ", index);
						//printf("%f ", random_number[0]);
						float prob = 0.005;
						//if (spike_target==0) prob = prob*0.8272;
						if(random_number[0]<prob){
							NeuronList[index].state[2] = 1;
							log_total_spike[index] = log_total_spike[index] + 1;
						}
					}else{
						NeuronList[index].state[2] = 0;
					}
					return;
				}


				if (NeuronList[index].state[2] > 0.1){
					NeuronList[index].state[0] = NeuronList[index].param[2];
					log_total_spike[index] = log_total_spike[index] + 1;

					//printf("spike_flag[%d]: %d\n", current_layer, spike_flag[current_layer]);
				}
				float Isynapses = test_current;
				Isynapses += input_2d[convolution_result_index][neuron_relative_index];

				Isynapses = Isynapses/current_divider;
				Isynapses = Isynapses * current_multiplier;

				float temp_v = NeuronList[index].state[0]+TimeStep*(NeuronList[index].param[5]+NeuronList[index].param[0]*NeuronList[index].state[0] + Isynapses*NeuronList[index].param[4]);
				if (LOW_BIT_MEM_POT){
					if((temp_v>NeuronList[index].state[0])<0.07){
						if((temp_v - NeuronList[index].state[0])<0.07) temp_v = NeuronList[index].state[0] + 0.07;
					}
					int intermediate_v = (int)((temp_v + 0.035 + 74.7)*100/7);
					temp_v = (float)intermediate_v*7.0/100.0 -74.7;
				}
				NeuronList[index].state[0] = (temp_v);
				if(NeuronList[index].state[0]<NeuronList[index].param[2]) NeuronList[index].state[0]=NeuronList[index].param[2];

//				if(index==3)printf("V:%f,F:%f,lV:%f|",temp_v,Isynapses,NeuronList[index].state[0]);

				if (temp_v>(NeuronList[index].param[1])){//it fires!

					NeuronList[index].state[0] = NeuronList[index].param[2];
					int fire_neuron_depth = (int)NeuronList[index].param[7];
					spike_flag[current_layer] = spike_flag[current_layer] + 1;
	//				printf("-%d-", fire_neuron_depth);


					NeuronList[index].state[3] = 1; //start counting the fired timer
					NeuronList[index].state[4] = 0; //reset the not fired timer

					log_spike[fire_neuron_depth] = 1;
					NeuronList[index].state[2] = 1.0;

					if(through_depth_inhibition){
						int total_depth_num = CNN_setttings->layer[current_layer].depth;
						int fired_depth = NeuronList[index].param[7];
						fired_depth = fired_depth - CNN_setttings->layer[current_layer].first_depth_id;
						int index_in_one_depth = index + input_size - CNN_setttings->layer[current_layer].depth_list[fired_depth].first_neuron;
						//printf("Fire no: %d, index: %d (in %d)\n", index, index_in_one_depth, CNN_setttings->layer[current_layer].depth_list[fired_depth].first_neuron);
						for(int depth_iter=0; depth_iter<total_depth_num; depth_iter++){
							int target_index = CNN_setttings->layer[current_layer].depth_list[depth_iter].first_neuron - input_size + index_in_one_depth;
							NeuronList[target_index].state[7] = -10;
						}
					}
	//				NeuronList[index].state[2] = MID_LAYER_STDP_DURATION + 0.0;
	//				if(index<OUTPUT_LAYER_NEURON_NUM){
	//					NeuronList[index].state[2] = 1.0;
	//				}
//					printf("-fired: %d-", index);
				}
				else{
					//if(index<OUTPUT_LAYER_NEURON_NUM) NeuronList[index].state[2] = 0;
					NeuronList[index].state[2] = 0;
					NeuronList[index].state[4] += 1;  //count the not fired timer
					//NeuronList[index].state[3] = NeuronList[index].state[3] + 1;
					//if(NeuronList[index].state[3]>signal_width) NeuronList[index].state[3] = 0;
					if(NeuronList[index].state[3]>0)NeuronList[index].state[3] += 1;
					if(NeuronList[index].state[3]>signal_width){
						NeuronList[index].state[3] = 0;
					}

				}

				//count the signal timer





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

__global__ void run_spiking_learning (Neuron *NeuronList, Input_neuron *Input_neuronlist, CNN_struct *CNN_setttings, float *random_number, float **input_2d, \
		float **instance_matrix_2d, int current_layer, int network_size, int input_size, float *log_v, float *log_spike, float *log_total_spike, int *spike_flag, \
		int signal_width, float input_float, int time_stamp, bool enable_inhibition){
	//printf("its in gpu(main)\n");
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    //printf("=%d-%d=",index,network_size);
    //printf("type is %d\n",NeuronList[1].type);
    //printf("%d\n",index);
    float current_multiplier = 2;//+5*(current_layer-1);
    float current_divider = input_float;
    //printf("-%f", current_multiplier/current_divider);
    if(time_stamp>0)index = (index + time_stamp)%network_size;
	if(index>=network_size){
		//printf("its wrong!\n");
		return;
	}


//	float start_depth = CNN_setttings->layer[current_layer].first_depth_id - 0.1;
//	float end_depth = CNN_setttings->layer[current_layer].last_depth_id + 0.1;
//	if(NeuronList[index].param[7]<start_depth||NeuronList[index].param[7]>end_depth){
//		return;
//	}
//	index = index + CNN_setttings->layer[current_layer].depth_list[0].first_neuron;

	if(index<input_size){
		if(current_layer>0) return;
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
						log_total_spike[index] = log_total_spike[index] + 1;
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
			instance_matrix_2d[current_layer][instance_index] = Input_neuronlist[index].state[2] * 8;
			//instance_matrix_2d[current_layer][instance_index] = 2;
			//printf(" %f", instance_matrix_2d[current_layer][instance_index]);
	}else{

		int neuron_relative_index = index - CNN_setttings->layer[current_layer].depth_list[0].first_neuron;
		int within_depth_relative_index = neuron_relative_index%CNN_setttings->layer[current_layer].depth_list[0].total_neuron_num;
		int current_depth = neuron_relative_index/CNN_setttings->layer[current_layer].depth_list[0].total_neuron_num;
		int alter_input_index = within_depth_relative_index*CNN_setttings->layer[current_layer].depth + current_depth;
//		printf("| %d, %d, %d, %d |", neuron_relative_index, within_depth_relative_index, current_depth, alter_input_index);

		index = index - input_size;
		if(NeuronList[index].state[7] > 0.1){
			NeuronList[index].state[7] = NeuronList[index].state[7] - 1;
			current_multiplier = 0;
			//NeuronList[index].state[2] = 0;
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
		//if(current_layer==2) printf("%d ", index+input_size);


		if(NeuronList[index].type == 2){//run LIF
			if(NeuronList[index].index >= 0){
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
					NeuronList[index].state[7] = 20; //set refractory period
					if (!enable_inhibition) NeuronList[index].state[7] = 1;
					NeuronList[index].state[0] = NeuronList[index].param[2];
					log_total_spike[index+input_size] = log_total_spike[index+input_size] + 1;

					//if((index+input_size)>50000) printf("%d ", index+input_size);

					NeuronList[index].spike_cnt += 1;
					if (LEARNER_HOMEOSTASIS_ENABLE && time_stamp%SPIKE_FREQ_SAMPLING_INTV==0){

						NeuronList[index].spike_frequency = NeuronList[index].spike_cnt/SPIKE_FREQ_SAMPLING_INTV + 0.0000000001;
						NeuronList[index].spike_cnt = 0;

						//printf("|%d@%d: %f|", index, time_stamp, NeuronList[index].spike_frequency);

					}
					//if (enable_inhibition) NeuronList[index].param[0] += 0.01;
					//printf("spike_flag[%d]: %d\n", current_layer, spike_flag[current_layer]);
				}

				//if(NeuronList[index].param[0]>0.01) NeuronList[index].param[0] -= 0.0001;

				float Isynapses = test_current;
				Isynapses += input_2d[convolution_result_index][neuron_relative_index];

//				if(current_layer==2 && Isynapses>0.1)printf("index(%d,%d):%f|", index, neuron_relative_index,Isynapses);

//				if(index>1000){
//					//if(input_2d[convolution_result_index][neuron_relative_index]>0.1) printf("index(%d,%d):%f|", index, neuron_relative_index,input_2d[convolution_result_index][neuron_relative_index]);
//				}
//
//				if(index==1000){
//					if(input_2d[convolution_result_index][23]>0.1) {
//						//for(int ij=0;ij<1000;ij++) printf(" %f ", input_2d[convolution_result_index][ij]);
//					}
//				}
//				printf(" %1.2f", Isynapses);
				Isynapses = Isynapses/current_divider;
				Isynapses = Isynapses * current_multiplier;

//				if(current_layer==1 && NeuronList[index].param[7]<3) Isynapses = 30;
				//if(current_layer==1 && NeuronList[index].param[7]<3 && Isynapses>0) printf("id: %d, %1.2f|", index, Isynapses);

//				if(current_layer==1 && index==10000) printf(" %1.2f", Isynapses);
				/*
				float v_temp_0 = NeuronList[index].param[0] + Isynapses*NeuronList[index].param[4];
				float old_v = old_device_neurons[index].state[0];
				float temp_v = v_temp_0 + (old_v-v_temp_0)*expf(-1*TimeStep/NeuronList[index].param[5]);
				*/
				//if((time_stamp!=0)) log_total_spike[index] = (log_total_spike[index]*(time_stamp-1) + Isynapses)/(time_stamp);
				if(Isynapses>30) Isynapses=30;
				//Isynapses = 0;

				float temp_v = NeuronList[index].state[0]+TimeStep*(NeuronList[index].param[5]+NeuronList[index].param[0]*NeuronList[index].state[0] +\
						Isynapses*NeuronList[index].param[4]);
				//if(index==405 && temp_v>-70) printf("%f ", temp_v);

				if (LOW_BIT_MEM_POT){
//					if((temp_v>NeuronList[index].state[0])<0.07){
//						if((temp_v - NeuronList[index].state[0])<0.07) temp_v = NeuronList[index].state[0] + 0.07;
//					}
//					int intermediate_v = (int)((temp_v + 0.035 + 74.7)*100/7);
//					temp_v = (float)intermediate_v*7.0/100.0 -74.7;
					float diff = temp_v - NeuronList[index].state[0];
					if(abs(diff)<=(14.7/256)){
						if(diff>0) temp_v = NeuronList[index].state[0] + (14.7/256);
						else temp_v = NeuronList[index].state[0] - (14.7/256);
					}else{
						float adjusted = (int)(diff/(14.7/256));
						temp_v = NeuronList[index].state[0] + adjusted*(14.7/256);
					}

				}
				if (temp_v<NeuronList[index].param[2]) temp_v = NeuronList[index].param[2];
				NeuronList[index].state[0] = (temp_v);
//				if (index==3) printf("V:%f,F:%f,lV:%f|",temp_v,Isynapses,NeuronList[index].state[0]);
//				if (index==27000) printf("V:%f,F:%f,lV:%f|",temp_v,Isynapses,NeuronList[index].state[0]);
//				if(current_layer==2 && Isynapses>50) printf("i: %1.2f, v: %1.2f|", Isynapses, temp_v);
//				if(current_layer==2 && index==41000) printf("i: %1.2f, v: %1.2f|", Isynapses, temp_v);
//				if(current_layer==2 && index==41000) printf("thres: %1.2f", NeuronList[index].param[1]);
				if (temp_v>(NeuronList[index].param[1])){//it fires!
//					if(current_layer==1 && NeuronList[index].param[7]<3) printf("id: %d, %1.2f|", index, Isynapses);
					NeuronList[index].state[0] = NeuronList[index].param[2];
					int fire_neuron_depth = (int)NeuronList[index].param[7];
					spike_flag[current_layer] = spike_flag[current_layer] + 1;
//					printf("-%d-", fire_neuron_depth);

					log_spike[fire_neuron_depth] = 1;

					NeuronList[index].state[2] = 1.0;
					//enable_inhibition = true;

					NeuronList[index].state[3] = 1;
					NeuronList[index].state[4] = 0;

					if(through_depth_inhibition && enable_inhibition){
						int total_depth_num = CNN_setttings->layer[current_layer].depth;
						int fired_depth = NeuronList[index].param[7];
						fired_depth = fired_depth - CNN_setttings->layer[current_layer].first_depth_id;
						int index_in_one_depth = index + input_size - CNN_setttings->layer[current_layer].depth_list[fired_depth].first_neuron;
						//printf("Fire no: %d, index: %d (in %d)\n", index, index_in_one_depth, CNN_setttings->layer[current_layer].depth_list[fired_depth].first_neuron);
						for(int depth_iter=0; depth_iter<total_depth_num; depth_iter++){

							int target_index = CNN_setttings->layer[current_layer].depth_list[depth_iter].first_neuron - input_size + index_in_one_depth;
							if(NeuronList[target_index].state[7]<=1 && index!=target_index) NeuronList[target_index].state[7] = -10;
							//printf("Fire no: %d, index: %d (in %d), %d inhibited\n", index, index_in_one_depth, CNN_setttings->layer[current_layer].depth_list[fired_depth].first_neuron, target_index);
							if(target_index>=network_size || target_index<0){
								printf("er: %d, from: %d + %d - %d|", target_index, CNN_setttings->layer[current_layer].depth_list[depth_iter].first_neuron, index_in_one_depth, input_size );
								//continue;
							}
						}
					}

					if(apply_local_inhibition && enable_inhibition){
						for (int LI_idx=0; LI_idx<MAX_LOCAL_INHIBITION; LI_idx++){
							if (NeuronList[index].local_inhibition[LI_idx]<input_size) continue;
							int target_index = NeuronList[index].local_inhibition[LI_idx] - 1 - input_size;
							if(NeuronList[target_index].state[7]<=1) NeuronList[target_index].state[7] = -10;

						}
					}

	//				NeuronList[index].state[2] = MID_LAYER_STDP_DURATION + 0.0;
	//				if(index<OUTPUT_LAYER_NEURON_NUM){
	//					NeuronList[index].state[2] = 1.0;
	//				}
					//printf("-fired: %d-", index);
				}else{
					//if(index<OUTPUT_LAYER_NEURON_NUM) NeuronList[index].state[2] = 0;
					//if(index<OUTPUT_LAYER_NEURON_NUM) NeuronList[index].state[2] = 0;
					NeuronList[index].state[2] = 0;
					NeuronList[index].state[4] += 1;  //count the not fired timer
					//NeuronList[index].state[3] = NeuronList[index].state[3] + 1;
					//if(NeuronList[index].state[3]>signal_width) NeuronList[index].state[3] = 0;
					if(NeuronList[index].state[3]>0)NeuronList[index].state[3] += 1;
					if(NeuronList[index].state[3]>signal_width){
						NeuronList[index].state[4] = 0; //if fired count is over, start not fired from beginning
						NeuronList[index].state[3] = 0;
					}
				}

			}
		}

		int instance_index = neuron_relative_index;
		if(instance_index<0||instance_index>CNN_setttings->layer[current_layer].neuron_num){
			return;
		}
		//output_instance_matrix[instance_index] = NeuronList[index].state[2];
		instance_matrix_2d[current_layer][instance_index] = NeuronList[index].state[2]*(current_layer);
		if(NeuronList[index].state[3]>0) instance_matrix_2d[current_layer][instance_index] = 1*(current_layer);
//		instance_matrix_2d[current_layer][instance_index] = 0;
//		instance_matrix_2d[current_layer][50*4+1] = 1000;


	}
}


__global__ void run_spiking_learning_one_layer (Neuron *NeuronList, Input_neuron *Input_neuronlist, CNN_struct *CNN_setttings, float *random_number, float **input_2d, \
		float **instance_matrix_2d, int current_layer, int network_size, int input_size, float *log_v, float *log_spike, float *log_total_spike, int *spike_flag, \
		int signal_width, float input_float, int time_stamp, bool enable_inhibition){
	//printf("its in gpu(main)\n");
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    //printf("=%d-%d=",index,network_size);
    //printf("type is %d\n",NeuronList[1].type);
    //printf("%d\n",index);
    float current_multiplier = 2;//+5*(current_layer-1);
    float current_divider = input_float;
    //printf("-%f", current_multiplier/current_divider);
    //if(time_stamp>0)index = (index + time_stamp)%network_size;
	if(index>=network_size){
		//printf("its wrong!\n");
		return;
	}


//	float start_depth = CNN_setttings->layer[current_layer].first_depth_id - 0.1;
//	float end_depth = CNN_setttings->layer[current_layer].last_depth_id + 0.1;
//	if(NeuronList[index].param[7]<start_depth||NeuronList[index].param[7]>end_depth){
//		return;
//	}
//	index = index + CNN_setttings->layer[current_layer].depth_list[0].first_neuron;

	if(current_layer==0){
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
						log_total_spike[index] = log_total_spike[index] + 1;
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
			instance_matrix_2d[current_layer][instance_index] = Input_neuronlist[index].state[2] * 8;
			//instance_matrix_2d[current_layer][instance_index] = 2;
			//printf(" %f", instance_matrix_2d[current_layer][instance_index]);
	}else{

		if (index>=CNN_setttings->layer[current_layer].neuron_num) return;
		int neuron_relative_index = index;
		int within_depth_relative_index = neuron_relative_index%CNN_setttings->layer[current_layer].depth_list[0].total_neuron_num;
		int current_depth = neuron_relative_index/CNN_setttings->layer[current_layer].depth_list[0].total_neuron_num;
		int alter_input_index = within_depth_relative_index*CNN_setttings->layer[current_layer].depth + current_depth;
//		printf("| %d, %d, %d, %d |", neuron_relative_index, within_depth_relative_index, current_depth, alter_input_index);

		index = index + CNN_setttings->layer[current_layer].depth_list[0].first_neuron - input_size;
		if(NeuronList[index].state[7] > 0.1){
			NeuronList[index].state[7] = NeuronList[index].state[7] - 1;
			current_multiplier = 0;
			//NeuronList[index].state[2] = 0;
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
		//if(current_layer==2) printf("%d ", index+input_size);


		if(NeuronList[index].type == 2){//run LIF
			if(NeuronList[index].index >= 0){
				//printf("%d-", index);

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
					NeuronList[index].state[7] = 20; //set refractory period
					if (!enable_inhibition) NeuronList[index].state[7] = 1;
					NeuronList[index].state[0] = NeuronList[index].param[2];
					log_total_spike[index+input_size] = log_total_spike[index+input_size] + 1;

					//if((index+input_size)>50000) printf("%d ", index+input_size);

					NeuronList[index].spike_cnt += 1;
					if (LEARNER_HOMEOSTASIS_ENABLE && time_stamp%SPIKE_FREQ_SAMPLING_INTV==0){

						NeuronList[index].spike_frequency = NeuronList[index].spike_cnt/SPIKE_FREQ_SAMPLING_INTV + 0.0000000001;
						NeuronList[index].spike_cnt = 0;

						//printf("|%d@%d: %f|", index, time_stamp, NeuronList[index].spike_frequency);

					}
					//if (enable_inhibition) NeuronList[index].param[0] += 0.01;
					//printf("spike_flag[%d]: %d\n", current_layer, spike_flag[current_layer]);
				}

				//if(NeuronList[index].param[0]>0.01) NeuronList[index].param[0] -= 0.0001;

				float Isynapses = test_current;
				Isynapses += input_2d[convolution_result_index][neuron_relative_index];

//				if(current_layer==2 && Isynapses>0.1)printf("index(%d,%d):%f|", index, neuron_relative_index,Isynapses);

//				if(index>1000){
//					//if(input_2d[convolution_result_index][neuron_relative_index]>0.1) printf("index(%d,%d):%f|", index, neuron_relative_index,input_2d[convolution_result_index][neuron_relative_index]);
//				}
//
//				if(index==1000){
//					if(input_2d[convolution_result_index][23]>0.1) {
//						//for(int ij=0;ij<1000;ij++) printf(" %f ", input_2d[convolution_result_index][ij]);
//					}
//				}
//				printf(" %1.2f", Isynapses);
				Isynapses = Isynapses/current_divider;
				Isynapses = Isynapses * current_multiplier;

//				if(current_layer==1 && NeuronList[index].param[7]<3) Isynapses = 30;
				//if(current_layer==1 && NeuronList[index].param[7]<3 && Isynapses>0) printf("id: %d, %1.2f|", index, Isynapses);

//				if(current_layer==1 && index==10000) printf(" %1.2f", Isynapses);
				/*
				float v_temp_0 = NeuronList[index].param[0] + Isynapses*NeuronList[index].param[4];
				float old_v = old_device_neurons[index].state[0];
				float temp_v = v_temp_0 + (old_v-v_temp_0)*expf(-1*TimeStep/NeuronList[index].param[5]);
				*/
				//if((time_stamp!=0)) log_total_spike[index] = (log_total_spike[index]*(time_stamp-1) + Isynapses)/(time_stamp);
				if(Isynapses>30) Isynapses=30;
				//Isynapses = 0;

				float temp_v = NeuronList[index].state[0]+TimeStep*(NeuronList[index].param[5]+NeuronList[index].param[0]*NeuronList[index].state[0] +\
						Isynapses*NeuronList[index].param[4]);
				//if(index==405 && temp_v>-70) printf("%f ", temp_v);

				if (LOW_BIT_MEM_POT){
//					if((temp_v>NeuronList[index].state[0])<0.07){
//						if((temp_v - NeuronList[index].state[0])<0.07) temp_v = NeuronList[index].state[0] + 0.07;
//					}
//					int intermediate_v = (int)((temp_v + 0.035 + 74.7)*100/7);
//					temp_v = (float)intermediate_v*7.0/100.0 -74.7;
					float diff = temp_v - NeuronList[index].state[0];
					if(abs(diff)<=(14.7/256)){
						if(diff>0) temp_v = NeuronList[index].state[0] + (14.7/256);
						else temp_v = NeuronList[index].state[0] - (14.7/256);
					}else{
						float adjusted = (int)(diff/(14.7/256));
						temp_v = NeuronList[index].state[0] + adjusted*(14.7/256);
					}

				}
				if (temp_v<NeuronList[index].param[2]) temp_v = NeuronList[index].param[2];
				NeuronList[index].state[0] = (temp_v);
//				if (index==3) printf("V:%f,F:%f,lV:%f|",temp_v,Isynapses,NeuronList[index].state[0]);
//				if (index==27000) printf("V:%f,F:%f,lV:%f|",temp_v,Isynapses,NeuronList[index].state[0]);
//				if(current_layer==2 && Isynapses>50) printf("i: %1.2f, v: %1.2f|", Isynapses, temp_v);
//				if(current_layer==2 && index==41000) printf("i: %1.2f, v: %1.2f|", Isynapses, temp_v);
//				if(current_layer==2 && index==41000) printf("thres: %1.2f", NeuronList[index].param[1]);
				if (temp_v>(NeuronList[index].param[1])){//it fires!
//					if(current_layer==1 && NeuronList[index].param[7]<3) printf("id: %d, %1.2f|", index, Isynapses);
					NeuronList[index].state[0] = NeuronList[index].param[2];
					int fire_neuron_depth = (int)NeuronList[index].param[7];
					spike_flag[current_layer] = spike_flag[current_layer] + 1;
//					printf("-%d-", fire_neuron_depth);

					log_spike[fire_neuron_depth] = 1;

					NeuronList[index].state[2] = 1.0;
					//enable_inhibition = true;

					NeuronList[index].state[3] = 1;
					NeuronList[index].state[4] = 0;

					if(through_depth_inhibition && enable_inhibition){
						int total_depth_num = CNN_setttings->layer[current_layer].depth;
						int fired_depth = NeuronList[index].param[7];
						fired_depth = fired_depth - CNN_setttings->layer[current_layer].first_depth_id;
						int index_in_one_depth = index + input_size - CNN_setttings->layer[current_layer].depth_list[fired_depth].first_neuron;
						//printf("Fire no: %d, index: %d (in %d)\n", index, index_in_one_depth, CNN_setttings->layer[current_layer].depth_list[fired_depth].first_neuron);
						for(int depth_iter=0; depth_iter<total_depth_num; depth_iter++){

							int target_index = CNN_setttings->layer[current_layer].depth_list[depth_iter].first_neuron - input_size + index_in_one_depth;
							if(NeuronList[target_index].state[7]<=1 && index!=target_index) NeuronList[target_index].state[7] = -10;
							//printf("Fire no: %d, index: %d (in %d), %d inhibited\n", index, index_in_one_depth, CNN_setttings->layer[current_layer].depth_list[fired_depth].first_neuron, target_index);
							if(target_index>=network_size || target_index<0){
								printf("er: %d, from: %d + %d - %d|", target_index, CNN_setttings->layer[current_layer].depth_list[depth_iter].first_neuron, index_in_one_depth, input_size );
								//continue;
							}
						}
					}

					if(apply_local_inhibition && enable_inhibition){
						for (int LI_idx=0; LI_idx<MAX_LOCAL_INHIBITION; LI_idx++){
							if (NeuronList[index].local_inhibition[LI_idx]<input_size) continue;
							int target_index = NeuronList[index].local_inhibition[LI_idx] - 1 - input_size;
							if(NeuronList[target_index].state[7]<=1) NeuronList[target_index].state[7] = -10;

						}
					}

	//				NeuronList[index].state[2] = MID_LAYER_STDP_DURATION + 0.0;
	//				if(index<OUTPUT_LAYER_NEURON_NUM){
	//					NeuronList[index].state[2] = 1.0;
	//				}
					//printf("-fired: %d-", index);
				}else{
					//if(index<OUTPUT_LAYER_NEURON_NUM) NeuronList[index].state[2] = 0;
					//if(index<OUTPUT_LAYER_NEURON_NUM) NeuronList[index].state[2] = 0;
					NeuronList[index].state[2] = 0;
					NeuronList[index].state[4] += 1;  //count the not fired timer
					//NeuronList[index].state[3] = NeuronList[index].state[3] + 1;
					//if(NeuronList[index].state[3]>signal_width) NeuronList[index].state[3] = 0;
					if(NeuronList[index].state[3]>0)NeuronList[index].state[3] += 1;
					if(NeuronList[index].state[3]>signal_width){
						NeuronList[index].state[4] = 0; //if fired count is over, start not fired from beginning
						NeuronList[index].state[3] = 0;
					}
				}

			}
		}

		int instance_index = neuron_relative_index;
		if(instance_index<0||instance_index>CNN_setttings->layer[current_layer].neuron_num){
			return;
		}

		instance_matrix_2d[current_layer][instance_index] = NeuronList[index].state[2]*(current_layer);
		if(NeuronList[index].state[3]>0) instance_matrix_2d[current_layer][instance_index] = 1*(current_layer);
	}
}

//This one is for event based machine vision application
void spiking_cnn_main_event_based(Neuron *NeuronList, Input_neuron *Input_neuronlist, Event_Camera_Input *events, int event_cnt, CNN_struct *host_CNN_setttings, CNN_struct *CNN_setttings, float *random_number, float **input, float **instance_matrix, \
		int current_layer, int network_size, int input_size, float *log_v, float *log_spike, float *log_total_spike, int *spike_flag, int signal_width, \
		float input_float, int time_stamp, bool enable_inhibition){

	//cout<<"In spiking_cnn_main"<<endl;
	if(current_layer==0){
		int SIZE_PER_SIDE = sqrt(input_size)+1;
		dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );
		dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));
		//std::cout<<SIZE_PER_SIDE<<" "<<endl;
		//cout<<"ip size: "<<input_size<<endl;
		run_spiking_learning_event_based_input<<<dimGrid, dimBlock>>>(NeuronList,Input_neuronlist, events, event_cnt, CNN_setttings, random_number, input, instance_matrix, \
				current_layer, network_size, input_size,log_v, log_spike, log_total_spike, spike_flag, signal_width, input_float, time_stamp, enable_inhibition);
	}else{
		int total_neurons_to_simulate = host_CNN_setttings->layer[current_layer].neuron_num;
		int SIZE_PER_SIDE = sqrt(total_neurons_to_simulate)+1;
		dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );
		dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));
	    run_spiking_learning_one_layer<<<dimGrid, dimBlock>>>(NeuronList,Input_neuronlist, CNN_setttings, random_number, input, instance_matrix, \
	    		current_layer, network_size, input_size,log_v, log_spike, log_total_spike, spike_flag, signal_width, input_float, time_stamp, enable_inhibition);
		if(enable_inhibition && (through_depth_inhibition || apply_local_inhibition)) inhibition_through_depth<<<dimGrid, dimBlock>>>(NeuronList, network_size, input_size);

	}
	//printf("inSpikingLearning");
    //cudaDeviceSynchronize();

}

//This one is for machine vision application
void spiking_cnn_main(Neuron *NeuronList, Input_neuron *Input_neuronlist, CNN_struct *CNN_setttings, float *random_number, float **input, float **instance_matrix, \
		int current_layer, int network_size, int input_size, float *log_v, float *log_spike, float *log_total_spike, int *spike_flag, int signal_width, \
		float input_float, int time_stamp, bool enable_inhibition){

	//cout<<"In spiking_cnn_main"<<endl;

	int SIZE_PER_SIDE = sqrt(network_size)+1;
    dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );
    dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));
    //std::cout<<SIZE_PER_SIDE<<" "<<endl;
	//cout<<"ip size: "<<input_size<<endl;
    run_spiking_learning<<<dimGrid, dimBlock>>>(NeuronList,Input_neuronlist, CNN_setttings, random_number, input, instance_matrix, \
    		current_layer, network_size, input_size,log_v, log_spike, log_total_spike, spike_flag, signal_width, input_float, time_stamp, enable_inhibition);
    if(enable_inhibition && (through_depth_inhibition || apply_local_inhibition)) inhibition_through_depth<<<dimGrid, dimBlock>>>(NeuronList, network_size, input_size);

	//printf("inSpikingLearning");
    //cudaDeviceSynchronize();

}

//This is for time sequence
void spiking_cnn_main(Neuron *NeuronList, Input_neuron *Input_neuronlist, CNN_struct *CNN_setttings, float *random_number, float **input, float **instance_matrix, int current_layer, \
		int network_size, int input_size, float *log_v, float *log_spike, float *log_total_spike, int *spike_flag, int signal_width, float input_float, int time_stamp, \
		int optional_inp, bool teaching_mode){

	//cout<<"In spiking_cnn_main"<<endl;



	int SIZE_PER_SIDE = sqrt(network_size)+1;
    dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );
    dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));

    run_time_seq<<<dimGrid, dimBlock>>>(NeuronList,Input_neuronlist, CNN_setttings, random_number, input, instance_matrix, current_layer, network_size, input_size,log_v, \
    		log_spike, log_total_spike, spike_flag, signal_width, input_float, optional_inp, teaching_mode);

    //printf("inSpikingLearning");
    //cudaDeviceSynchronize();

}



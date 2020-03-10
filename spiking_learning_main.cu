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



__global__ void run_spiking_learning (Neuron *NeuronList, Neuron *old_device_neurons, float *random_number, int network_size, float *log_v, float *log_spike, float *log_total_spike, int *spike_flag, int signal_width, int time_stamp){
	//printf("its in gpu(main)\n");
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    //printf("=%d-%d=",index,network_size);
    //printf("type is %d\n",NeuronList[1].type);
    //printf("%d\n",index);
    float current_multiplier = 1;
    float current_divider = 20.0;

	if(index>network_size){
		//printf("its wrong!\n");
		return;
	}
	if(index<OUTPUT_LAYER_NEURON_NUM){
		//current_multiplier = 1;
	}

	if(NeuronList[index].state[7] > 0.1){
		NeuronList[index].state[7] = NeuronList[index].state[7] - 1;
		current_multiplier = 0;
		//printf("%d is inhibited, mem potential:%d\n",index, NeuronList[index].state[0]);
		//return;
	}

	//printf("%d,",NeuronList[index].type);

	if(NeuronList[index].type == 0){//run IZH
		//printf("itInIZH");
		if(NeuronList[index].index > 0){
			//printf("index is %d, a is %f, b is %f, v is %f, u is %f \n", NeuronList[index].index, NeuronList[index].param[0], NeuronList[index].param[1], NeuronList[index].state[0], NeuronList[index].state[1]);
			//printf("U of %d is %f \n", index+1, NeuronList[index].state[0]);
			//printf("%f, ", NeuronList[index].state[0]);
			//printf("_%f_", log_v[time_stamp]);
			log_v[time_stamp] = NeuronList[index].state[0];
			log_spike[time_stamp] = NeuronList[index].state[2];
			if(NeuronList[index].state[2] > 0.1){
				NeuronList[index].state[2] = 0;
				log_total_spike[index] = log_total_spike[index] + 1;
				spike_flag[0] = spike_flag[0] + 1;
			}
			//printf("|%f|", log_v[time_stamp]);
			float Isynapses = test_current;

					int i = 0;
					while(NeuronList[index].connected_in[i] > 0.1){
						int connected_in = NeuronList[index].connected_in[i] - 1;
						//printf("I is: %f;\n", Isynapses);
						//printf("state of connected in (to index: %d): %d is %d\n", index, connected_in, NeuronList[connected_in].state[2]);
						if(old_device_neurons[connected_in].state[2] > 0.1){//fired
							Isynapses = Isynapses + NeuronList[index].connected_weight[i];
							//printf("Connected_fired_I is: %f; weight is %f\n", Isynapses, NeuronList[index].connected_weight[i]);
							if(old_device_neurons[connected_in].type == 4){
								Isynapses = Isynapses + old_device_neurons[connected_in].state[0];
								NeuronList[index].synapse_timer[connected_in-SPIKING_NEURON_NUM] = 1;	//update the synapse timer, this is the only place
								old_device_neurons[index].synapse_timer[connected_in-SPIKING_NEURON_NUM] = 1;
								//printf("I is: %f; signal is: %f\n", Isynapses, old_device_neurons[connected_in].state[0]);
							}
						}
						i++;
					}

					Isynapses = Isynapses/current_divider;
					Isynapses = Isynapses * current_multiplier;

					float old_v = old_device_neurons[index].state[0];
					float old_u = old_device_neurons[index].state[1];


					if(old_v<30){
						float dv = (0.04*old_v+5)*old_v + 140 - old_u;
						NeuronList[index].state[0] = old_v + (dv + Isynapses)*TimeStep;
						float du = NeuronList[index].param[0]*(NeuronList[index].param[1]*old_v-old_u);
						NeuronList[index].state[1] = old_u + TimeStep*du;
					}else{
						old_device_neurons[index].state[0] = 30;
						NeuronList[index].state[0] = NeuronList[index].param[2];
						NeuronList[index].state[1] = old_u + NeuronList[index].param[3];
						NeuronList[index].state[2] = 1;
					}

		}
	}
	else if(NeuronList[index].type == 1){//run Stochastic
		if(NeuronList[index].index >= 0){

			log_v[time_stamp] = NeuronList[index].state[0];
			log_spike[time_stamp] = NeuronList[index].state[2];
			if(NeuronList[index].state[2] != 0){
				if (NeuronList[index].state[2] > tau+1){
					NeuronList[index].state[2] = 0;
					//printf("timing of %d is %d \n", index+1, NeuronList[index].state);
				}
				else{
					NeuronList[index].state[2] = NeuronList[index].state[2] + 1;
				}
			}
			else{
				int i = 0;
				float potential_value = NeuronList[index].param[0];
				while(NeuronList[index].connected_in[i] > 0.1){
					int connected_in = NeuronList[index].connected_in[i] -1 ;
					//printf("state of connected in (to index: %d): %d is %d\n", index, connected_in, NeuronList[connected_in].state);
					if(NeuronList[connected_in].state[2] > 0){
						//int weight_index = index + connected_in*SIZE;
						potential_value = potential_value + PSP*NeuronList[index].connected_weight[i];
						//printf("weight in of %d is %f\n", NeuronList[index].connected_in[i], weight_list[weight_index]);
					}
					i++;
				}
				//printf("MP of %d is %f\n", index, potential_value);
				float probability_of_firing = expf(potential_value)/tau;
				//printf("expf is: %f||",expf(potential_value));
				float random_compare = random_number[index];
				if(probability_of_firing>random_compare){
					//printf("Neuron : %d||potential_value is: %f and random# is %f||probability is %f\n",index, potential_value,random_compare,probability_of_firing);
					NeuronList[index].state[2] = 1;
				}

			}
		}
	}
	else if(NeuronList[index].type == 2){//run LIF
		if(NeuronList[index].index >= 0){
			if(index==30){//this is the membrane potential logger
				log_spike[time_stamp] = NeuronList[index].state[2];
				log_v[time_stamp] = NeuronList[index].state[0];
				//printf("time_is%d, IMultiplier_s_%f__potential_is%f\n",time_stamp, current_multiplier,old_device_neurons[index].state[0]);
			}
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
			if (old_device_neurons[index].state[2] > 0.1){
				old_device_neurons[index].state[0] = old_device_neurons[index].param[2];
				log_total_spike[index] = log_total_spike[index] + 1;
				spike_flag[0] = spike_flag[0] + 1;

			}
			float Isynapses = test_current;

			int i = 0;
			while(NeuronList[index].connected_in[i] > 0.1){
				int connected_in = NeuronList[index].connected_in[i] - 1;
				//printf("state of connected in (to index: %d): %d is %d\n", index, connected_in, NeuronList[connected_in].state);
				if(old_device_neurons[connected_in].state[2] > 0.1){
					//printf("@");
					Isynapses = Isynapses + NeuronList[index].connected_weight[i];
					if(old_device_neurons[connected_in].type == 4){
						Isynapses = Isynapses + old_device_neurons[connected_in].state[0];
						NeuronList[index].synapse_timer[connected_in-SPIKING_NEURON_NUM] = 1;	//update the synapse timer, this is the only place
						old_device_neurons[index].synapse_timer[connected_in-SPIKING_NEURON_NUM] = 1;
						//printf("I is: %f; signal is: %f\n", Isynapses, old_device_neurons[connected_in].state[0]);
						//printf("I is: %f; signal is: %f\n", Isynapses, old_device_neurons[connected_in].state[0]);
					}
				}
				i++;
			}
			//debug
			//int connected_in = NeuronList[index].connected_in[0] - 1;
			//printf("INPUTCurrent_of_no. %d is: %f\n", index, Isynapses);
			//Isynapses = Isynapses + old_device_neurons[connected_in].state[0];
			//printf("input of no. %d is: %f, switch is %f\n",connected_in, NeuronList[connected_in].state[0], NeuronList[connected_in].state[2]);

			//end of debug
			Isynapses = Isynapses/current_divider;
			Isynapses = Isynapses * current_multiplier;
			/*
			float v_temp_0 = NeuronList[index].param[0] + Isynapses*NeuronList[index].param[4];
			float old_v = old_device_neurons[index].state[0];
			float temp_v = v_temp_0 + (old_v-v_temp_0)*expf(-1*TimeStep/NeuronList[index].param[5]);
			*/

			float temp_v = NeuronList[index].state[0]+TimeStep*(NeuronList[index].param[5]+NeuronList[index].param[0]*NeuronList[index].state[0] + Isynapses*NeuronList[index].param[4]);

			NeuronList[index].state[0] = (temp_v);

			if (temp_v>(NeuronList[index].param[1])){
				NeuronList[index].state[0] = NeuronList[index].param[2];
				NeuronList[index].state[2] = MID_LAYER_STDP_DURATION + 0.0;
				if(index<OUTPUT_LAYER_NEURON_NUM){
					NeuronList[index].state[2] = 1.0;
				}
				//printf("-%f-",NeuronList[index].state[2]);
			}
			else{
				if(index<OUTPUT_LAYER_NEURON_NUM) NeuronList[index].state[2] = 0;
			}
		}
	}
	else if(NeuronList[index].type == 3){//run HH
		if(NeuronList[index].index >= 0){

			log_spike[time_stamp] = NeuronList[index].state[2];
			log_v[time_stamp] = NeuronList[index].state[0];
			//printf("%f; ", NeuronList[index].state[0]);
			//printf("U of %d is %f \n", index+1, NeuronList[index].state[0]);
			if(NeuronList[index].state[2] > 0.1){
				NeuronList[index].state[2] = 0;
				//printf("sp\n");
				log_total_spike[index] = log_total_spike[index] + 1;
				spike_flag[0] = spike_flag[0] + 1;
			}
			float Isynapses = test_current;

							int i = 0;
							while(NeuronList[index].connected_in[i] > 0.1){
								int connected_in = NeuronList[index].connected_in[i] - 1;
								//printf("state of connected in (to index: %d): %d is %d\n", index, connected_in, NeuronList[connected_in].state);
								if(old_device_neurons[connected_in].state[2] > 0.1){
									Isynapses = Isynapses + NeuronList[index].connected_weight[i];
									if(old_device_neurons[connected_in].type == 4){
										Isynapses = Isynapses + old_device_neurons[connected_in].state[0];
										NeuronList[index].synapse_timer[connected_in-SPIKING_NEURON_NUM] = 1;	//update the synapse timer, this is the only place
										old_device_neurons[index].synapse_timer[connected_in-SPIKING_NEURON_NUM] = 1;
									}
								}
								i++;
							}
							Isynapses = Isynapses/current_divider;
							Isynapses = Isynapses * current_multiplier;
							//am: y = 0.1*(x+35)/(1-expf(-1*(x+35)/10))
							//bm: y = 4*expf(-0.0556*(x+60))
							//an: y = 0.01*(x+50)/(1-expf(-1*(x+50)/10))
							//bn: y = 0.125*expf(-1*(x+60)/80)
							//ah: y = 0.07*expf(-0.05*(x+60))
							//bh: y = 1/(1+expf(-0.1*(x+30)))
							float old_v = old_device_neurons[index].state[0];
							float old_m = old_device_neurons[index].state[1];
							//float old_flag = old_device_neurons[index].state[2];
							float old_n = old_device_neurons[index].state[3];
							float old_h = old_device_neurons[index].state[4];

							NeuronList[index].state[1] = old_m + TimeStep*(((0.1*(old_v+35)/(1-expf(-1*(old_v+35)/10.0)))*(1-old_m))-((4*expf(-0.0556*(old_v+60)))*old_m));
							NeuronList[index].state[3] = old_n + TimeStep*((0.01*(old_v+50)/(1-expf(-1*(old_v+50)/10.0)))*(1-old_n)-((0.125*expf(-1*(old_v+60)/80.0))*old_n));
							NeuronList[index].state[4] = old_h + TimeStep*(((0.07*expf(-0.05*(old_v+60)))*(1-old_h))-((1.0/(1+expf(-0.1*(old_v+30))))*old_h));

							float gNa = NeuronList[index].param[4]*powf(old_m,3)*old_h;
							float gK = NeuronList[index].param[5]*powf(old_n,4);
							float gl = NeuronList[index].param[6];

							float INa = gNa * (old_v - NeuronList[index].param[1]);
							float IK = gK * (old_v - NeuronList[index].param[2]);
							float Il = gl * (old_v - NeuronList[index].param[3]);

							NeuronList[index].state[0] = old_v + TimeStep*((1.0/NeuronList[index].param[0])*(Isynapses-(INa+IK+Il))); //Euler method to find next voltage

							if (NeuronList[index].state[0] > -50){
								NeuronList[index].state[5] = 1;
							}
							else{
								if (NeuronList[index].state[5] > 0.1){
									NeuronList[index].state[2] = 1;
									//printf("\n spike! \n");
								}
								NeuronList[index].state[5] = 0;
							}
				}
	}
	else if(NeuronList[index].type == 4){
		//printf("No.: %d, counter is: %f \n",index, NeuronList[index].state[1]);
		//P.S. state[0] is the signal strength, state[1] is firing frequency;
		//printf("-%d-",index);
		if(NeuronList[index].index >= 0){
			if(NeuronList[index].state[1] == 0){//if the target frequency is zero, turn off
				NeuronList[index].state[2] = 0;
				return;
			}
			if(NeuronList[index].state[2] > 0){	//state[3] is used to count current signal width
				//printf("No.: %d, counter is: %f \n",index, NeuronList[index].state[3]);
				//printf("*%d*",index);
				NeuronList[index].state[3] = NeuronList[index].state[3] + 1;
				//printf("`");
				if(NeuronList[index].state[3]>signal_width){
					NeuronList[index].state[2] = 0;
					NeuronList[index].state[3] = 0;
				}
			}else{
				NeuronList[index].state[4] = NeuronList[index].state[4] + 1;//use state[4] to count the time it has not fired
				//printf("counter is: %f, period is: %f \n", NeuronList[index].state[3], NeuronList[index].state[1]);

				if((NeuronList[index].state[4])>(NeuronList[index].state[1])){
								//printf("SignalNeuron_%d:counter is: %f, period is: %f \n", index, NeuronList[index].state[4], NeuronList[index].state[1]);
								log_total_spike[index] = log_total_spike[index] + 1;
								NeuronList[index].state[2] = 1;
								NeuronList[index].state[4] = 0;
				}
			}


		}

	}

}

void spiking_learning_main(Neuron *NeuronList, Neuron *old_device_neurons, float *random_number, int network_size, float *log_v, float *log_spike, float *log_total_spike, int *spike_flag, int signal_width, int time_stamp){

	int SIZE_PER_SIDE = sqrt(network_size)+1;
	//printf("sizeperside: %d\n",SIZE_PER_SIDE);
    dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );
    dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));
    //cout<<to_string(dimBlock);
    //time_stamp = 1;
    run_spiking_learning<<<dimGrid, dimBlock>>>(NeuronList, old_device_neurons, random_number, network_size, log_v, log_spike, log_total_spike, spike_flag, signal_width, time_stamp);
    //printf("inSpikingLearning");
    cudaDeviceSynchronize();





}

#include "header.h"
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>
#include <iostream>
#include <fstream>
cudaError_t cudaerr;
using namespace std;

__global__ void read_filter (CNN_struct *settings, float **device_filter_array){
	int counter = 0;
	printf("Printing filter array on GPU\n");
	for (int i=0;i<CNN_total_layer_num-1;i++){
		int filter_size = settings->layer[i+1].conv_setting.filter_depth * settings->layer[i+1].conv_setting.filter_width * settings->layer[i+1].conv_setting.filter_length * settings->layer[i+1].depth;
		for(int j=0;j<filter_size;j++){
			printf("%f ", device_filter_array[i][j]);
			counter ++;
		}
		printf("\n");
	}
}

__global__ void read_filter_one_layer (CNN_struct *settings, float *device_filter_array, int layer_num){
	int counter = 0;
	printf("Printing one layer filter array on GPU\n");

	int filter_size = settings->layer[layer_num].conv_setting.filter_depth * settings->layer[layer_num].conv_setting.filter_width * settings->layer[layer_num].conv_setting.filter_length * settings->layer[layer_num].depth;
	filter_size = 5;
	printf("depth: %f\n", settings->layer[layer_num].conv_setting.filter_depth);
	printf("width: %f\n", settings->layer[layer_num].conv_setting.filter_width);
	printf("depth: %f\n", settings->layer[layer_num].conv_setting.filter_depth);
	for(int j=0;j<filter_size;j++){
		printf("%f ", device_filter_array[j]);
		counter ++;
	}
	printf("\n");

}

__global__ void weight_cpy_filter_to_neuronlist (CNN_struct *CNN_setttings, Neuron *NeuronList, int network_size, int input_neuron_size, float **filter, int current_layer){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    //printf(" %d", index);
	if(index<input_neuron_size) return;
    if(index>=network_size) return;

//	printf("relative %d, ", CNN_setttings->layer[current_layer].depth_list[0].first_neuron);


//	if((NeuronList[index].param[7]-current_layer)>0.01||(NeuronList[index].param[7]-current_layer)<-0.01){
//		printf("param_7 is: %f", NeuronList[index].param[7]);
//		return;
//	}


	int neuron_relative_index = index - CNN_setttings->layer[current_layer].depth_list[0].first_neuron;

	index = index - input_neuron_size;
	if(NeuronList[index].type==4||NeuronList[index].type==5){//if the post-synapse neuron is input-signal-neuron, jump over
		return;
	}
	float start_depth = CNN_setttings->layer[current_layer].first_depth_id - 0.1;
	float end_depth = CNN_setttings->layer[current_layer].last_depth_id + 0.1;
	if(NeuronList[index].param[7]<start_depth||NeuronList[index].param[7]>end_depth){
//		printf("StartDepth:%f_End:%f__current:%f||", start_depth, end_depth, NeuronList[index].param[7]);
		return;
	}

	int number_of_neurons_per_depth = CNN_setttings->layer[current_layer].depth_list[0].total_neuron_num;

	int filter_size_per_depth = CNN_setttings->layer[current_layer].conv_setting.filter_length * CNN_setttings->layer[current_layer].conv_setting.filter_width* CNN_setttings->layer[current_layer].conv_setting.filter_depth;
	int i = 0;
	//int current_depth = neuron_relative_index/number_of_neurons_per_depth + CNN_setttings->layer[current_layer].first_depth_id - 1;
	int current_depth = NeuronList[index].param[7] - CNN_setttings->layer[current_layer].first_depth_id;
//	printf("%d=", index);
	//if(NeuronList[index].state[2]>0.1){//if post-synapse neuron fired
		while(NeuronList[index].connected_in[i] > 0.1){
			int connected_in = NeuronList[index].connected_in[i] - 1;
			int filter_index = current_depth*filter_size_per_depth+i;
			NeuronList[index].connected_weight[i] = filter[current_layer-1][filter_index];
			i++;
			//printf("%d: %f, ",index, filter[current_layer-1][filter_index]);
		}
	//}
}




void write_weight_to_file(CNN_struct *network_config, float **filter_array, string plot_prefix){

	//for(int i=0; i<15; i++) cout<<i<<": "<<filter_array[0][i]<<", ";

	ofstream myfile_2 (plot_prefix+"CNN_WEIGHT_OUT.csv");
	if (myfile_2.is_open()){
		for (int layer_index=1; layer_index<CNN_total_layer_num; layer_index++){
			convolution_param current_conv = network_config->layer[layer_index].conv_setting;
			//cout<<endl<<"check this: "<<current_conv.filter_depth<<" "<<network_config->layer[layer_index].depth<<endl;
			//float filter_mat[current_conv.filter_depth][network_config->layer[layer_index].depth][current_conv.filter_length][current_conv.filter_width];
			int filter_size=current_conv.filter_depth*network_config->layer[layer_index].depth*current_conv.filter_length*current_conv.filter_width;
			printf("filer size: %d\n", filter_size);
			//float *filter_mat = new float[filter_size];
			//for(int i=0;i<filter_size;i++) filter_mat[i]=0;
			//memcpy(filter_mat, filter_array[layer_index-1], sizeof(filter_mat));
			myfile_2 << endl << endl;
			int index_count = 0;
				for (int kernel = 0; kernel < current_conv.filter_depth; ++kernel) {
					for (int channel = 0; channel < network_config->layer[layer_index].depth; ++channel) {
					  for (int row = 0; row < current_conv.filter_length; ++row) {
						for (int column = 0; column < current_conv.filter_width; ++column) {
						  //myfile_2 << filter_mat[kernel][channel][row][column] << ", ";
						  //cout<<index_count<<": "<<filter_mat[index_count]<<" ,";
						  myfile_2 << filter_array[layer_index-1][index_count] << ", ";
						  index_count ++;
						}
					  }
					  //cout<<"Kernel: "<<kernel<<" has depth: "<<network_config->layer[layer_index].depth<<endl;
					  myfile_2 << endl;
					}
					myfile_2 << endl;
				}
			//delete[] filter_mat;
		}

		myfile_2.close();
	}
}

void shared_weight_gen(CNN_struct *network_config, float *filter_array[CNN_total_layer_num-1]){
	for (int layer_index=1; layer_index<CNN_total_layer_num; layer_index++){
		convolution_param current_conv = network_config->layer[layer_index].conv_setting;
		//float *weight = new float[current_conv.filter_depth][network_config->layer[layer_index].depth][current_conv.filter_length][current_conv.filter_width];
		srand (1);
		int mid_conductance = 500;
		const float kernel_template[3][3] = {
		{1, 1, 1},
		{1, 0, 1},
		{1, 1, 1}
		};
//		const int kernel_count = current_conv.filter_depth;
//		const int channel_count = network_config->layer[layer_index].depth;
//		const int row_count = current_conv.filter_length;
//		const int column_count = current_conv.filter_width;
		//float filter_mat[current_conv.filter_depth][network_config->layer[layer_index].depth][current_conv.filter_length][current_conv.filter_width] = {{{{0}}}};
		//float *filter_mat = new float[kernel_count][channel_count][row_count][column_count];

		int filter_size=current_conv.filter_depth*network_config->layer[layer_index].depth*current_conv.filter_length*current_conv.filter_width;
		float *filter_mat = new float[filter_size];
		cout<<"Filter size: "<<filter_size<<endl;
		int index_count = 0;
		for (int kernel = 0; kernel < current_conv.filter_depth; ++kernel) {
			for (int channel = 0; channel < network_config->layer[layer_index].depth; ++channel) {
			  for (int row = 0; row < current_conv.filter_length; ++row) {
				for (int column = 0; column < current_conv.filter_width; ++column) {

				  int fluct = 50 - (rand() % 100);
					if(non_random_weight_init) fluct = 0;
				  //filter_mat[kernel][channel][row][column] = (mid_conductance+fluct)/1000.0;
				  filter_mat[index_count] = (mid_conductance+fluct)/1000.0;
				  index_count++;
				  //filter_mat[kernel][channel][row][column] = kernel_template[row][column];
				}
			  }
			  //printf(" %d, %d||", kernel, channel);
			}
		}

		memcpy(filter_array[layer_index-1], filter_mat, filter_size*sizeof(float));

		//network_config->layer[layer_index].filter = filter;
		delete[] filter_mat;
	}
}

int filter_util(CNN_struct *settings, Neuron *NeuronList, int network_size, int input_neuron_size, float **host_filter_array, float **device_filter_array, string plot_prefix, int function_select){

	if(function_select==0){//whole function, initialize host array and copy to device
		int filter_array_size = CNN_total_layer_num-1;

		//load filter into host array
		float *weight_array[CNN_total_layer_num-1];
		//cout<<sizeof(weight_array[0])<<endl;
		for (int i=0;i<CNN_total_layer_num-1;i++){
			int filter_size = settings->layer[i+1].conv_setting.filter_depth * settings->layer[i+1].conv_setting.filter_width * settings->layer[i+1].conv_setting.filter_length * settings->layer[i+1].depth;
			cout<<"$$Filter_size_is: "<<filter_size<<endl;
			weight_array[i] = new float[filter_size];

//				const int kernel = settings->layer[i+1].depth;
//				const int channel = settings->layer[i+1].conv_setting.filter_depth;
//				const int width = settings->layer[i+1].conv_setting.filter_width;
//				const int length = settings->layer[i+1].conv_setting.filter_length;
//				const int& channel_1 = settings->layer[i+1].conv_setting.filter_depth;
//				weight_array[i] = new float[kernel][channel_1][width][length];



		}

		shared_weight_gen(settings, weight_array);
//		printf("\n filter print: \n");
//		for(int ij=0;ij<784000;ij++) printf(" %1.1f ", weight_array[0][ij]);
//		printf("\n filter print: \n");
//		for(int ij=0;ij<78400;ij++) printf(" %1.1f ", weight_array[0][ij]);
//		float *filter_out;
		//memcpy(filter_out, settings->layer[1].filter, sizeof(filter_mat));
		//cout<<sizeof(weight_array[0])<<endl;

//		float **h_filter_array;
//		h_filter_array = (float**)malloc(filter_array_size* sizeof(float*));
//		host_filter_array
		for (int i=0;i<CNN_total_layer_num-1;i++){
			int filter_size = settings->layer[i+1].conv_setting.filter_depth * settings->layer[i+1].conv_setting.filter_width * settings->layer[i+1].conv_setting.filter_length * settings->layer[i+1].depth;
		    printf("Between layer %d and %d, filter size is: %d\n", i, i+1, filter_size);
			cudaMalloc((void **)&host_filter_array[i], filter_size * sizeof(float));
		    cudaMemcpy(host_filter_array[i], weight_array[i], filter_size * sizeof(float), cudaMemcpyHostToDevice);
			//cudaerr = cudaMemcpy(device_filter_array[i], weight_array[i], filter_size*sizeof(float), cudaMemcpyHostToDevice);
		}
		cudaerr = cudaMemcpy(device_filter_array, host_filter_array, filter_array_size* sizeof(float*), cudaMemcpyHostToDevice);

		CNN_struct *CNN_settings_device;
	    cudaMalloc((void **)&CNN_settings_device, 1*sizeof(CNN_struct));
	    cudaMemcpy(CNN_settings_device,settings,1*sizeof(CNN_struct),cudaMemcpyHostToDevice);

	    dim3 dimBlock(1, 1);
	    dim3 dimGrid(1, 1);
	    //read_filter<<<dimGrid, dimBlock>>>(CNN_settings_device, device_filter_array);
	    //read_filter_one_layer<<<1, 1>>>(CNN_settings_device, host_filter_array[0], 0);
		cudaFree(CNN_settings_device);
		//delete[] weight_array;
	}
	else if(function_select==1){//save filter to file
		float **h_filter_array_temp;
		int filter_array_size = CNN_total_layer_num-1;
		h_filter_array_temp = (float**)malloc(filter_array_size * sizeof(float*));
		for (int i=0;i<CNN_total_layer_num-1;i++){
			int filter_size = settings->layer[i+1].conv_setting.filter_depth * settings->layer[i+1].conv_setting.filter_width * settings->layer[i+1].conv_setting.filter_length * settings->layer[i+1].depth;
			cout<<filter_size<<" sized filter copied"<<endl;
			h_filter_array_temp[i] = (float*)malloc(filter_size * sizeof(float));
			cudaMemcpy(h_filter_array_temp[i], host_filter_array[i], filter_size * sizeof(float), cudaMemcpyDeviceToHost);
			//cudaerr = cudaMemcpy(device_filter_array[i], weight_array[i], filter_size*sizeof(float), cudaMemcpyHostToDevice);
		}
		write_weight_to_file(settings, h_filter_array_temp, plot_prefix);
	}
	else if(function_select==2){//copy filter to NeuronList
		cout<< "copy filter to NeuronList" <<endl;
		CNN_struct *CNN_settings_device;
	    cudaMalloc((void **)&CNN_settings_device, 1*sizeof(CNN_struct));
	    cudaMemcpy(CNN_settings_device,settings,1*sizeof(CNN_struct),cudaMemcpyHostToDevice);

//		int total_neuron_num = 0;
//		for(int i=0;i<CNN_total_layer_num;i++){
//			total_neuron_num += settings->layer[i].neuron_num;
//		}
		int total_neuron_num = network_size;
		int SIZE_PER_SIDE = sqrt(total_neuron_num)+1;
		dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );
		dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));
		for (int i=1;i<CNN_total_layer_num;i++){

			weight_cpy_filter_to_neuronlist<<<dimGrid, dimBlock>>>(CNN_settings_device, NeuronList, network_size, input_neuron_size, device_filter_array, i);
			//cout<<network_size<<", "<<input_neuron_size<<endl;
		}

		cudaFree(CNN_settings_device);
	}


	return 1;
}

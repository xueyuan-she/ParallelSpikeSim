#include <iostream>
#include <time.h>
#include <vector>
#include <string>
#include <stdlib.h>
#include <streambuf>
#include <sstream>
#include <fstream>
#include <math.h>
#include <array>
#include "header.h"
#include <cudnn.h>
using namespace std;

/*
test IZH: 1 0 0.15 0.3 -72.14 2.44 30 -60 -14 0 ; 1 0 .
test LIF: 1 2 -70 -55 -75 20 10 10 -70 0 0 ; 1 0 .
test HH: 1 3 0.01 55.17 -72.14 -49.42 1.2 0.36 0.003 -60 0.0529 0 0.3177 0.5961 ; 1 0 .
test signal input: 1 4 1 1 0 1 ; 0 0 .
*/

void ROI_gen(int network_type, int network_size){

	int neuron_count = 1;
	float param[8];
	float state[8];
	int param_num = 0;
	int state_num = 0;

	if(network_type==0){//this is izh
		float param_temp[8] = {5, 0.273, -56, 2, 30, 0, 0, 0};
		float state_temp[8] = {-60, -14, 0, 0, 0, 0, 0, 0};

		param_num = 5;
		state_num = 3;
		std::copy(param_temp, param_temp + 8, param);
		std::copy(state_temp, state_temp + 8, state);
		//printf("%f",param[4]);
		//state = state_temp;

	}

	if(network_type==2){
		float param_temp[8] = {-70, -55, -75, 20, 10, 10, 0, 0};
		float state_temp[8] = {-70, 0, 0, 0, 0, 0, 0, 0};

		param_num = 6;
		state_num = 3;
		std::copy(param_temp, param_temp + 8, param);
		std::copy(state_temp, state_temp + 8, state);
	}


	ofstream myfile_2 ("visual_lif.txt");
	    if (myfile_2.is_open()){
	    	for(int i=0; i < network_size; i++){
	    		myfile_2 << neuron_count << " " << network_type << " ";

	    		for(int j=0; j<param_num;j++){
	    			myfile_2 << param[j] << " ";
	    		}
	    		for(int j=0; j<state_num;j++){
	    			myfile_2 << state[j] << " ";
	    		}
	    		myfile_2 << "; " << network_size+i+1 << " 0 .\n";

	    		neuron_count++;
	    	}

	    		for(int k=0; k < network_size; k++){
					myfile_2 << neuron_count << " " << "4 " << "1 ";

					for(int j=0; j<3;j++){
						myfile_2 << "0" << " ";
					}
					myfile_2 << "; " << neuron_count << " 0 .\n";
					neuron_count++;
				}

	    	myfile_2.close();
	    }
}

void spike_learning_gen(int neuron_type, int network_size, int* mid_layer, int mid_layer_num, int input_num){

	int neuron_count = 1;
	float param[8];
	float state[8];
	float param_2[8];
	float state_2[8];

	int param_num = 0;
	int state_num = 0;
	int mid_conductance = 500;
	int total_layer_num = 2+mid_layer_num;
	int different_parameter = 0;

	float param_temp_2[8] = {-0.1089, -60.2, -74.7, 20, 0.615, -6.07, 0, 0};
	float state_temp_2[8] = {-70, 0, 0, 0, 0, 0, 0, 0};

	int connection_index[total_layer_num];
	int sum_of_mid_layer_neuron = 0;
	for(int g=0;g<mid_layer_num;g++){
		sum_of_mid_layer_neuron += mid_layer[g];
	}
	int output_layer_neuron_num = network_size - sum_of_mid_layer_neuron;

	for(int h=0;h<total_layer_num;h++){
		if(h==0) {
			connection_index[h] = output_layer_neuron_num;
		}else if(h==1+mid_layer_num){
			connection_index[h] = network_size + input_num;
		}else{
			connection_index[h] = connection_index[h-1] + mid_layer[h-1];
		}
		printf("index_is_%d\n", connection_index[h]);
	}

	if(neuron_type==0){//this is izh
		float param_temp[8] = {5, 0.273, -56, 2, 30, 0, 0, 0};
		float state_temp[8] = {-60, -14, 0, 0, 0, 0, 0, 0};

		param_num = 8;
		state_num = 8;
		std::copy(param_temp, param_temp + 8, param);
		std::copy(state_temp, state_temp + 8, state);
		//printf("%f",param[4]);
		//state = state_temp;

	}

	if(neuron_type==2){//this is LIF
		//type_1 used for 2-20
		float param_temp[8] = {-0.0989, -60.2, -74.7, 20, 0.314, -6.77, 0, 0};
		float state_temp[8] = {-70, 0, 0, 0, 0, 0, 0, 0};

		//type_2 used for 0.2-2


		param_num = 8;
		state_num = 8;
		std::copy(param_temp, param_temp + 8, param);
		std::copy(state_temp, state_temp + 8, state);

		std::copy(param_temp_2, param_temp_2 + 8, param_2);
		std::copy(state_temp_2, state_temp_2 + 8, state_2);

	}

	if(neuron_type==3){//this is HH
		float param_temp[8] = {0.35, 125.17, -122.14, -49.42, 1.2, 0.36, 0.003, 0};
		float state_temp[8] = {-60, 0.0529, 0, 0.3177, 0.5961, 0, 0, 0};

		param_num = 8;
		state_num = 8;
		std::copy(param_temp, param_temp + 8, param);
		std::copy(state_temp, state_temp + 8, state);
	}



/*
	ofstream myfile_2 ("spike_learning.txt");
	    if (myfile_2.is_open()){
	    	for(int i=0; i < network_size; i++){
	    		myfile_2 << neuron_count << " " << neuron_type << " ";

	    		for(int j=0; j<param_num;j++){
	    			myfile_2 << param[j] << " ";
	    		}
	    		for(int j=0; j<state_num;j++){
	    			myfile_2 << state[j] << " ";
	    		}
	    		myfile_2 << "; " ;
	    		for(int connected_in=0;connected_in<input_num;connected_in++){
	    			float cdt = 0;
	    			int fluct = 250 - (rand() % 500);
	    			//fluct = 0;
	    			cdt = (mid_conductance+fluct)/1000.0;

	    			myfile_2 << network_size+connected_in+1 << ' ' << to_string(cdt) << ' ';
	    		}

	    		myfile_2 << ".\n";
	    		neuron_count++;
	    	}

	    		for(int k=0; k < input_num; k++){
					myfile_2 << neuron_count << " " << "4 " << "1 ";

					for(int j=0; j<3;j++){
						myfile_2 << "0" << " ";
					}
					myfile_2 << "; " << neuron_count << " 1 .\n";
					neuron_count++;
				}

	    	myfile_2.close();
	    }

*/
	ofstream myfile_2 ("spike_learning.txt");
	for(int y=0; y<total_layer_num; y++){
		if(y==total_layer_num-1){
			for(int k=0; k < input_num; k++){
				myfile_2 << neuron_count << " " << "4 " << "1 ";

				for(int j=0; j<3;j++){
					myfile_2 << "0" << " ";
				}
				myfile_2 << "; " << neuron_count << " 1 .\n";
				neuron_count++;
			}
		}else{
			int start_num = 0;
			int end_num = connection_index[y];
			if(y!=0){
				start_num = connection_index[y-1];
				//end_num = connection_index[y];
			}

			int connection_start = connection_index[y];
			int connection_end = connection_index[y+1];

			if (myfile_2.is_open()){
				for(int i=start_num; i < end_num; i++){
					myfile_2 << neuron_count << " " << neuron_type << " ";
					if(different_parameter){
						if(neuron_count-1>=output_layer_neuron_num){
							for(int j=0; j<param_num;j++){
								myfile_2 << param[j] << " ";
							}
							for(int j=0; j<state_num;j++){
								myfile_2 << state[j] << " ";
							}
						}else{
							for(int j=0; j<param_num;j++){
								myfile_2 << param_2[j] << " ";
							}
							for(int j=0; j<state_num;j++){
								myfile_2 << state_2[j] << " ";
							}

						}
					}else{
						for(int j=0; j<param_num;j++){
							myfile_2 << param[j] << " ";
						}
						for(int j=0; j<state_num;j++){
							myfile_2 << state[j] << " ";
						}
					}


					myfile_2 << "; " ;
					for(int connected_in=connection_start;connected_in<connection_end;connected_in++){
						float cdt = 0;
						int fluct = 500 - (rand() % 1000);
						//fluct = 0;
						cdt = (mid_conductance+fluct)/1000.0;

						myfile_2 << connected_in+1 << ' ' << to_string(cdt) << ' ';
					}

					myfile_2 << ".\n";
					neuron_count++;
				}
			}
		}
	}

	myfile_2.close();
}

void spike_cnn_gen(CNN_struct *network_config){

	printf("Writing CNN to config file\n");
	int neuron_type = 2;

	float param[8];
	float state[8];
	float param_2[8];
	float state_2[8];

	int param_num = 8;
	int state_num = 8;
	int mid_conductance = 200;

	int different_parameter = 0;

	float param_temp[8] = {-0.1089, -60.2, -74.7, 20, 0.314, -6.07, 0, 0};
	float state_temp[8] = {-70, 0, 0, 0, 0, 0, 0, 0};
	float input_param_temp[8] = {1, 0, 0, 0, 0, 0, 0, 0};
	float input_state_temp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	ofstream myfile_2 ("spike_cnn.txt");

	int neuron_count = 1;
	int total_neuron = 0;
	int depth_count = network_config->layer[0].depth;

	for(int i=0;i<CNN_total_layer_num;i++){
		total_neuron += network_config->layer[i].neuron_num;
	}

	depth_count = network_config->layer[0].depth;
	//first process the input layer
	for(int dep_iter=0; dep_iter<depth_count; dep_iter++){
		input_param_temp[7] = network_config->layer[0].depth_list[dep_iter].id;
		for(int layer_neuron_count=0; layer_neuron_count<network_config->layer[0].width*network_config->layer[0].length; layer_neuron_count++){
//			myfile_2 << neuron_count << " " << "4 ";
			for(int j=0; j<param_num;j++){
//				myfile_2 << input_param_temp[j] << " ";
			}
			for(int j=0; j<state_num;j++){
//				myfile_2 << input_state_temp[j] << " ";
			}
//			myfile_2 << "; " << neuron_count << " 1 .\n";
			neuron_count++;
		}
	}


	for(int first_i=1; first_i<CNN_total_layer_num; first_i++){//first_i start with 1 (0 is input layer)
		//param_temp[6] = first_i;
		convolution_param conv_setting = network_config->layer[first_i].conv_setting;
		CNN_layer current_layer = network_config->layer[first_i];
		cout<<"Working on layer: " << first_i << endl;
		for(int first_j=0; first_j<current_layer.depth; first_j++){
			depth_struct current_depth = current_layer.depth_list[first_j];

			param_temp[7] = current_depth.id;
//			printf("#%d", current_depth.id);
			std::copy(param_temp, param_temp + 8, param);
			std::copy(state_temp, state_temp + 8, state);

			int input_size_x = network_config->layer[first_i-1].width;
			int input_size_y = network_config->layer[first_i-1].length;
			int output_size_x = (input_size_x-conv_setting.filter_width+2*conv_setting.pad_width)/conv_setting.horizontal_stride + 1;
			int output_size_y = (input_size_y-conv_setting.filter_length+2*conv_setting.pad_height)/conv_setting.vertical_stride + 1;

//			cout<<"output_size_x: "<<output_size_x<<"output_size_y: "<<output_size_y<<endl;
//			cout<<"current_depth.width: "<<current_depth.width<<"current_depth.length: "<<current_depth.length<<endl;
			if((output_size_x!=current_depth.width)||(output_size_y!=current_depth.length)){
				cout<<"WARNING: Layer Sizing Problem"<<endl;
			}

			int reverse_mapped_start_x = 0 - conv_setting.pad_width;
			int reverse_mapped_start_y = 0 - conv_setting.pad_height;

			for(int layer_y=0; layer_y<output_size_y; layer_y++){
							for(int layer_x=0; layer_x<output_size_x; layer_x++){
								if (myfile_2.is_open()){

									myfile_2 << neuron_count << " " << neuron_type << " ";
									if(different_parameter){
										for(int j=0; j<param_num;j++){
											myfile_2 << param[j] << " ";
										}
										for(int j=0; j<state_num;j++){
											myfile_2 << state[j] << " ";
										}
									}else{
										for(int j=0; j<param_num;j++){
											myfile_2 << param[j] << " ";
										}
										for(int j=0; j<state_num;j++){
											myfile_2 << state[j] << " ";
										}
									}
									myfile_2 << "; " ;

									//write connections:

									int reverse_mapped_left_x = reverse_mapped_start_x + layer_x*conv_setting.horizontal_stride;
									int reverse_mapped_top_y = reverse_mapped_start_y + layer_y*conv_setting.vertical_stride;

									for(int second_i=0; second_i<conv_setting.filter_depth;second_i++){
										depth_struct input_depth = network_config->layer[current_layer.input_layer].depth_list[second_i];
										int start_neuron_id = input_depth.first_neuron;
										int connected_in_count = 0;//for debug
										for(int second_k=0; second_k<conv_setting.filter_length;second_k++){
											int delta_y = second_k;
											int mapped_y = reverse_mapped_top_y + delta_y;
											if(mapped_y>=0&&mapped_y<input_depth.length){
												for(int second_j=0; second_j<conv_setting.filter_width;second_j++){
													int delta_x = second_j;
													int mapped_x = reverse_mapped_left_x + delta_x;
													if(mapped_x>=0&&mapped_x<input_depth.width){
														int mapped_index = start_neuron_id + mapped_y*input_depth.width + mapped_x;
														//if(neuron_count==7501)cout<<start_neuron_id<<" "<<mapped_y<<" "<<mapped_x<<endl;
														if(mapped_index<=input_depth.last_neuron){
															float cdt = 0;
															int fluct = 100 - (rand() % 200);
															//fluct = 0;
															cdt = (mid_conductance+fluct)/1000.0;
															myfile_2 << mapped_index + 1 << ' ' << to_string(cdt) << ' ';
															connected_in_count ++;
														}
													}else{
//														myfile_2 << (total_neuron+1) << ' ' << mapped_x << "|" << mapped_y << ' ';
														myfile_2 << (total_neuron+1) << ' ' << 0 << ' ';
													}
												}
											}else{
												for(int second_j=0; second_j<conv_setting.filter_width;second_j++){
													int delta_x = second_j;
													int mapped_x = reverse_mapped_left_x + delta_x;
//													myfile_2 << (total_neuron+1) << ' ' << mapped_x << "|" << mapped_y  << ' ';
													myfile_2 << (total_neuron+1) << ' ' << 0 << ' ';
												}
											}
										}
										//if(connected_in_count%2!=0&&connected_in_count!=9) cout<<"Wrong connected in number at depth: "<<current_depth.id<<endl;
									}

									myfile_2 << ".\n";
									neuron_count++;

								}
							}
						}

//									int reverse_mapped_center_x = layer_x*conv_setting.horizontal_stride;
//									int reverse_mapped_center_y = layer_y*conv_setting.vertical_stride;
//			for(int layer_y=0; layer_y<current_layer.length; layer_y++){
//				for(int layer_x=0; layer_x<current_layer.width; layer_x++){
//					if (myfile_2.is_open()){
//
//						myfile_2 << neuron_count << " " << neuron_type << " ";
//						if(different_parameter){
//							for(int j=0; j<param_num;j++){
//								myfile_2 << param[j] << " ";
//							}
//							for(int j=0; j<state_num;j++){
//								myfile_2 << state[j] << " ";
//							}
//						}else{
//							for(int j=0; j<param_num;j++){
//								myfile_2 << param[j] << " ";
//							}
//							for(int j=0; j<state_num;j++){
//								myfile_2 << state[j] << " ";
//							}
//						}
//						myfile_2 << "; " ;
//
//						//write connections:
//
//
//						int reverse_mapped_center_x = layer_x*conv_setting.horizontal_stride;
//						int reverse_mapped_center_y = layer_y*conv_setting.vertical_stride;
//
//
//						for(int second_i=0; second_i<conv_setting.filter_depth;second_i++){
//							depth_struct input_depth = network_config->layer[current_layer.input_layer].depth_list[second_i];
//							int start_neuron_id = input_depth.first_neuron;
//							int connected_in_count = 0;//for debug
//							for(int second_k=0; second_k<conv_setting.filter_length;second_k++){
//								int delta_y = second_k - (int) (conv_setting.filter_length/2);
//								int mapped_y = reverse_mapped_center_y + delta_y;
//								if(mapped_y>=0&&mapped_y<input_depth.length){
//									for(int second_j=0; second_j<conv_setting.filter_width;second_j++){
//										int delta_x = second_j - (int) (conv_setting.filter_width/2);
//										int mapped_x = reverse_mapped_center_x + delta_x;
//										if(mapped_x>=0&&mapped_x<input_depth.width){
//											int mapped_index = start_neuron_id + mapped_y*input_depth.width + mapped_x;
//											//if(neuron_count==7501)cout<<start_neuron_id<<" "<<mapped_y<<" "<<mapped_x<<endl;
//											if(mapped_index<=input_depth.last_neuron){
//												float cdt = 0;
//												//int fluct = 500 - (rand() % 1000);
//												//fluct = 0;
//												//cdt = (mid_conductance+fluct)/1000.0;
//												myfile_2 << mapped_index + 1 << ' ' << to_string(cdt) << ' ';
//												connected_in_count ++;
//											}
//										}else{
//											//myfile_2 << (total_neuron+1) << ' ' << mapped_x << "|" << mapped_y << ' ';
//											myfile_2 << (total_neuron+1) << ' ' << 0 << ' ';
//										}
//									}
//								}else{
//									for(int second_j=0; second_j<conv_setting.filter_width;second_j++){
//										int delta_x = second_j - (int) (conv_setting.filter_width/2);
//										int mapped_x = reverse_mapped_center_x + delta_x;
//										//myfile_2 << (total_neuron+1) << ' ' << mapped_x << "|" << mapped_y  << ' ';
//										myfile_2 << (total_neuron+1) << ' ' << 0 << ' ';
//									}
//								}
//							}
//							if(connected_in_count%2!=0&&connected_in_count!=9) cout<<"Wrong connected in number at depth: "<<current_depth.id<<endl;
//						}
//
//						myfile_2 << ".\n";
//						neuron_count++;
//
//					}
//				}
//			}

		}
	}
	//make the final neuron: padding
	myfile_2 << neuron_count << " " << "5 ";
	for(int j=0; j<param_num;j++){
		myfile_2 << input_param_temp[j] << " ";
	}
	for(int j=0; j<state_num;j++){
		myfile_2 << input_state_temp[j] << " ";
	}
	myfile_2 << "; " << neuron_count << " 1 .\n";


	myfile_2.close();
}

//void shared_weight_gen(CNN_struct *network_config, float *filter_array[CNN_total_layer_num-1]){
//	for (int layer_index=1; layer_index<CNN_total_layer_num; layer_index++){
//		convolution_param current_conv = network_config->layer[layer_index].conv_setting;
//		//float *weight = new float[current_conv.filter_depth][network_config->layer[layer_index].depth][current_conv.filter_length][current_conv.filter_width];
//		const float kernel_template[3][3] = {
//		{1, 1, 1},
//		{1, -8, 1},
//		{1, 1, 1}
//		};
//
//		float filter_mat[current_conv.filter_depth][network_config->layer[layer_index].depth][current_conv.filter_length][current_conv.filter_width];
//		int filter_size=current_conv.filter_depth*network_config->layer[layer_index].depth*current_conv.filter_length*current_conv.filter_width;
//		cout<<filter_size<<endl;
//		for (int kernel = 0; kernel < network_config->layer[layer_index].depth; ++kernel) {
//			for (int channel = 0; channel < current_conv.filter_depth; ++channel) {
//			  for (int row = 0; row < current_conv.filter_length; ++row) {
//				for (int column = 0; column < current_conv.filter_width; ++column) {
//				  filter_mat[kernel][channel][row][column] = kernel_template[row][column];
//				}
//			  }
//			}
//		}
//		memcpy(filter_array[layer_index-1], filter_mat, sizeof(filter_mat));
//
//		//network_config->layer[layer_index].filter = filter;
//
//	}
//}

void CNN_get_dimension(CNN_struct *settings, int layer_index, int *width_result, int *length_result){

	int convolution_result_index = layer_index - 1;
	if (layer_index==0) convolution_result_index = 0;

//	float *d_input;
//	float *filter;
//	float *output;
//
//    dim3 dimBlock(1, 1 );
//    dim3 dimGrid(1, 1);
//    copy_pointer<<<dimGrid, dimBlock>>>(d_input_2d, d_input, layer_index);
//    copy_pointer<<<dimGrid, dimBlock>>>(filter_2d, filter, convolution_result_index);
//    copy_pointer<<<dimGrid, dimBlock>>>(output_2d, output, convolution_result_index);
//	float **add = &output_2d[0];
//	printf("Address On GPU: %p\n", add);
//	read_data<<<1, 1>>>(output_2d[0]);

	int filter_in_channel;
	int filter_out_channel;
	int filter_height;
	int filter_width;

	int input_batch_size = 1;
	int input_channel;
	int input_height;
	int input_width;

	int output_channel;
	int output_batch_size = 1;
	int output_height;
	int output_width;



//	if(layer_index==0){
//		filter_in_channel = input_image_channel;
//		filter_out_channel = settings->layer[layer_index].depth;
//		filter_height = settings->layer[layer_index].conv_setting.filter_length;
//		filter_width = settings->layer[layer_index].conv_setting.filter_width;
//
//		input_batch_size = 1;
//		input_channel = input_image_channel;
//		input_height = input_image_l;
//		input_width = input_image_w;
//
//	}else{
//		filter_in_channel = settings->layer[layer_index+1].conv_setting.filter_depth;
//		filter_out_channel = settings->layer[layer_index+1].depth;
//		filter_height = settings->layer[layer_index+1].conv_setting.filter_length;
//		filter_width = settings->layer[layer_index+1].conv_setting.filter_width;
//
//		input_batch_size = 1;
//		input_channel = settings->layer[layer_index+1].conv_setting.filter_depth;
//		input_height = settings->layer[layer_index].depth_list[0].length;
//		input_width = settings->layer[layer_index].depth_list[0].width;
//	}



	filter_in_channel = settings->layer[layer_index].conv_setting.filter_depth;
	filter_out_channel = settings->layer[layer_index].depth;
	filter_height = settings->layer[layer_index].conv_setting.filter_length;
	filter_width = settings->layer[layer_index].conv_setting.filter_width;

	input_batch_size = 1;
	input_channel = settings->layer[layer_index].conv_setting.filter_depth;
	input_height = settings->layer[layer_index-1].length;
	input_width = settings->layer[layer_index-1].width;

	cudnnHandle_t cudnn;
	cudnnCreate(&cudnn);



	cudnnTensorDescriptor_t input_descriptor;
	cudnnCreateTensorDescriptor(&input_descriptor);
	cudnnSetTensor4dDescriptor(input_descriptor,
										/*format=*/CUDNN_TENSOR_NHWC,
										/*dataType=*/CUDNN_DATA_FLOAT,
										/*batch_size=*/input_batch_size,
										/*channels=*/input_channel,
										/*image_height=*/input_height,
										/*image_width=*/input_width);




	cudnnFilterDescriptor_t kernel_descriptor;
	cudnnCreateFilterDescriptor(&kernel_descriptor);
	cudnnSetFilter4dDescriptor(kernel_descriptor,
										/*dataType=*/CUDNN_DATA_FLOAT,
										/*format=*/CUDNN_TENSOR_NCHW,
										/*out_channels=*/filter_out_channel,
										/*in_channels=*/filter_in_channel,
										/*kernel_height=*/filter_height,
										/*kernel_width=*/filter_width);



	cudnnConvolutionDescriptor_t convolution_descriptor;
	cudnnCreateConvolutionDescriptor(&convolution_descriptor);
	cudnnSetConvolution2dDescriptor(convolution_descriptor,
											 /*pad_height=*/settings->layer[layer_index].conv_setting.pad_height,
											 /*pad_width=*/settings->layer[layer_index].conv_setting.pad_width,
											 /*vertical_stride=*/settings->layer[layer_index].conv_setting.vertical_stride,
											 /*horizontal_stride=*/settings->layer[layer_index].conv_setting.horizontal_stride,
											 /*dilation_height=*/settings->layer[layer_index].conv_setting.dilation_height,
											 /*dilation_width=*/settings->layer[layer_index].conv_setting.dilation_width,
											 /*mode=*/CUDNN_CROSS_CORRELATION,
											 /*computeType=*/CUDNN_DATA_FLOAT);




	int batch_size{0}, channels{0}, height{0}, width{0};
	cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
												   input_descriptor,
												   kernel_descriptor,
												   &batch_size,
												   &channels,
												   &height,
												   &width);

	width_result[0] = width;
	length_result[0] = height;

	cout<<"height(from cuDNN calculate): "<<height<<endl;
	cout<<"width(from cuDNN calculate): "<<width<<endl;

	cudnnDestroyTensorDescriptor(input_descriptor);
	cudnnDestroyFilterDescriptor(kernel_descriptor);
	cudnnDestroyConvolutionDescriptor(convolution_descriptor);

	cudnnDestroy(cudnn);

}

void CNN_sturct_build(CNN_struct *network_config){
	printf("Building CNN struct\n");
	int depth_id_count = 0;
	//define the network here
	for (int layer_index=0; layer_index<CNN_total_layer_num; layer_index++){
			convolution_param conv_build;
			if(layer_index==0){//this is the input layer
				conv_build.dilation_height = 1;
				conv_build.dilation_width = 1;
				conv_build.filter_depth = 1;
				conv_build.filter_length = 0;
				conv_build.filter_width = 0;
				conv_build.horizontal_stride = 1;
				conv_build.vertical_stride = 1;
				conv_build.pad_height = 1;
				conv_build.pad_width = 1;

				network_config->layer[layer_index].depth = input_image_channel;
				network_config->layer[layer_index].conv_setting = conv_build;
				network_config->layer[layer_index].layer_id = layer_index;
				network_config->layer[layer_index].input_layer = layer_index - 1;
				network_config->layer[layer_index].width = input_image_w; //stride = 1
				network_config->layer[layer_index].length = input_image_l;
			}else if(layer_index==1){
				conv_build.dilation_height = 1;
				conv_build.dilation_width = 1;											//If all connect, use:
				conv_build.filter_depth = network_config->layer[layer_index-1].depth;	//network_config->layer[layer_index-1].depth;
				conv_build.filter_length = 	network_config->layer[layer_index-1].length;	//network_config->layer[layer_index-1].length;
				conv_build.filter_width = network_config->layer[layer_index-1].width;		//network_config->layer[layer_index-1].width;
				conv_build.horizontal_stride = 1;
				conv_build.vertical_stride = 1;
				conv_build.pad_height = 0;
				conv_build.pad_width = 0;

				network_config->layer[layer_index].depth = 100;
				network_config->layer[layer_index].conv_setting = conv_build;
				network_config->layer[layer_index].layer_id = layer_index;
				network_config->layer[layer_index].input_layer = layer_index - 1;

			}else if(layer_index==2){
				conv_build.dilation_height = 1;
				conv_build.dilation_width = 1;
				conv_build.filter_depth = network_config->layer[layer_index-1].depth;
				conv_build.filter_length = 3;
				conv_build.filter_width = 3;
				conv_build.horizontal_stride = 3;
				conv_build.vertical_stride = 3;
				conv_build.pad_height = 1;
				conv_build.pad_width = 1;
				network_config->layer[layer_index].depth = 16;
				network_config->layer[layer_index].conv_setting = conv_build;

				network_config->layer[layer_index].layer_id = layer_index;
				network_config->layer[layer_index].input_layer = layer_index - 1;

			}else if(layer_index==3){
				conv_build.dilation_height = 1;
				conv_build.dilation_width = 1;
				conv_build.filter_depth = network_config->layer[layer_index-1].depth;
				conv_build.filter_length = 3;
				conv_build.filter_width = 3;
				conv_build.horizontal_stride = 1;
				conv_build.vertical_stride = 1;
				conv_build.pad_height = 1;
				conv_build.pad_width = 1;

				network_config->layer[layer_index].depth = 1;
				network_config->layer[layer_index].conv_setting = conv_build;
				network_config->layer[layer_index].layer_id = layer_index;
				network_config->layer[layer_index].input_layer = layer_index - 1;

			}
	}

	for (int layer_index=0; layer_index<CNN_total_layer_num; layer_index++){
		printf("Layer:%d\n", layer_index);
		convolution_param conv_build;
		if(layer_index==0){//this is the input layer

		}else{
			int *width_result = new int[1];
			int *length_result = new int[1];
			CNN_get_dimension(network_config, layer_index, width_result, length_result);

			network_config->layer[layer_index].width = width_result[0]; //stride = 1
			network_config->layer[layer_index].length = length_result[0];
		}

		//cout<<"current depth is:"<<network_config->layer[layer_index].depth<<endl;
		network_config->layer[layer_index].neuron_num = network_config->layer[layer_index].width * network_config->layer[layer_index].length * network_config->layer[layer_index].depth;
		if(layer_index==0){
			network_config->layer[layer_index].first_depth_id = 0;
			network_config->layer[layer_index].last_depth_id = network_config->layer[0].depth - 1;
		}else{
			network_config->layer[layer_index].first_depth_id = network_config->layer[layer_index-1].last_depth_id + 1;
			network_config->layer[layer_index].last_depth_id = network_config->layer[layer_index-1].last_depth_id + network_config->layer[layer_index].depth;
		}

		for(int i=0; i<network_config->layer[layer_index].depth; i++){
			depth_struct current_depth;
			current_depth.id = depth_id_count;
//			printf("#%d", current_depth.id);
			current_depth.total_neuron_num = network_config->layer[layer_index].width * network_config->layer[layer_index].length;
			current_depth.width = network_config->layer[layer_index].width;
			current_depth.length = network_config->layer[layer_index].length;
			if(i==0){
				if(layer_index==0){
					current_depth.first_neuron = 0;
				}else{
					current_depth.first_neuron = network_config->layer[layer_index-1].depth_list[(network_config->layer[layer_index-1].depth-1)].last_neuron + 1;
				}
			}else{
				current_depth.first_neuron = network_config->layer[layer_index].depth_list[i-1].last_neuron + 1;
			}
			current_depth.last_neuron = current_depth.first_neuron + current_depth.total_neuron_num - 1;
			network_config->layer[layer_index].depth_list[i] = current_depth;
			depth_id_count ++;
		}
	}
}

int network_config_generator(int function_select, CNN_struct *settings){

	switch(function_select){
		case 1:{
			//ROI
			int network_size = 64*64*3; //this only includes the main neuron(no signal neurons)
			int network_type = 0;
			ROI_gen(network_type, network_size);
		}
		break;
		case 2:{
			//Spiking Simple
			int network_size = 1000;
			int mid_layer_num = 0;
			int mid_layer[mid_layer_num];

			mid_layer[0] = 500;
			//mid_layer[1] = 600;

			int input_size = 32*32*3;

			spike_learning_gen(2, network_size, (int *)mid_layer, mid_layer_num, input_size);
		}
		break;
		case 3:{
			//Spiking CNN generator, first build CNN struct
			CNN_sturct_build(settings);
			//Then, write to file
			spike_cnn_gen(settings);
			/*
			float *weight_array[3];
			for (int i=0;i<3;i++){
				int filter_size = i*2;
				//cout<<filter_size<<endl;
				weight_array[i] = new float[filter_size];
				for (int j=0; j<filter_size; j++){
					weight_array[i][j] = i+j;

				}
				//cout<<sizeof(weight_array)<<endl;
			}
			for (int i=0;i<3;i++){
				for (int j=0; j<8; j++){
					cout<<weight_array[i][j]<<"|";

				}
				cout<<endl;
			}
			*/
		}

		break;
	}
	return 0;
}

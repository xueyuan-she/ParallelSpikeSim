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


//currently using LIF for spike learning

__global__ void check_weight (Neuron *NeuronList, int network_size){
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	if(index>network_size){
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

__global__ void check_total_spike (float *log_total_spike, int network_size){
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	if(index>network_size){
		return;
	}
	//printf("|");

	printf("spikeNo_ofNeuronNo%d_is_%f\n",index,log_total_spike[index]);

}

void print_connected_in_weight_old(Neuron *NeuronList, int output_index_start, int output_index_stop, int plot){

	float weight_max = 0;
	float weight_min = 10000;

	for(int i=0;i<(output_index_stop-output_index_start);i++){

		for (int y=0; y<28; ++y) {
				for (int x=0; x<28; ++x) {
				  //std::cout << ((image[y*28+x] == 0.0)? ' ' : '*');
				  //std::cout << std::to_string(int(100*(NeuronList[output_index_start+i].connected_weight[y*28+x]))) << ' ';

				  if(NeuronList[output_index_start+i].connected_weight[y*28+x]>weight_max){
					  weight_max = NeuronList[output_index_start+i].connected_weight[y*28+x];
				  }
				  if(NeuronList[output_index_start+i].connected_weight[y*28+x]<weight_min){
					  weight_min = NeuronList[output_index_start+i].connected_weight[y*28+x];
				  }

				}
				//std::cout << std::endl;
		}
		//cout<<"\n\n\n";
	}
	//printf("weightMaxIs%f;weightMinIs%f\n", weight_max, weight_min);

	if(plot){
		float weight_diff = weight_max - weight_min;

		cimg_library::CImg<unsigned char> image("color_small.jpg");
		/*
		cimg_library::CImgDisplay display(image, "Click a point");
		while (!display.is_closed())
		    {
		        display.wait();
		        if (display.button() && display.mouse_y() >= 0 && display.mouse_x() >= 0)
		        {
		            const int y = display.mouse_y();
		            const int x = display.mouse_x();

		            unsigned char randomColor[3];
		            randomColor[0] = rand() % 256;
		            randomColor[1] = rand() % 256;
		            randomColor[2] = rand() % 256;

		            image.draw_point(x, y, randomColor);
		        }
		        image.display(display);
		    }
		*/
		for(int i=0;i<(output_index_stop-output_index_start);i++){
			int current_index = i+output_index_start;
			int img_i;
			int img_j;
			int img_k;
			for (img_i=0;img_i<input_image_w;img_i++){
				for (img_j=0;img_j<input_image_l;img_j++){
					for(img_k=0;img_k<3;img_k++){
						float weight_raw = NeuronList[current_index].connected_weight[img_i*input_image_w+img_j];
						image(img_j, img_i, 0, img_k) = 255*(weight_raw-weight_min)/weight_diff;

						//printf("pixel%d, %d, signal is: %f \n",img_i, img_j, img_temp);
					}
				}
			}
			/*
			cimg_library::CImgDisplay main_disp(image,"Synapse_Conductance");
			while (!main_disp.is_closed()) {
					main_disp.wait();
			}
			*/
			string out_file_name = "weight_out_index_"+to_string(current_index)+".jpg";
			image.save(out_file_name.c_str());
		}
	}


}


void print_connected_in_weight(Neuron *NeuronList, int output_index_start, int output_index_stop, int plot, string plot_prefix){

	float weight_max = 0;
	float weight_min = 10000;

	for(int i=0;i<(output_index_stop-output_index_start);i++){

		for (int y=0; y<input_image_w; ++y) {
				for (int x=0; x<input_image_l; ++x) {
				  //std::cout << ((image[y*28+x] == 0.0)? ' ' : '*');
				  //std::cout << std::to_string(int(100*(NeuronList[output_index_start+i].connected_weight[y*28+x]))) << ' ';

				  if(NeuronList[output_index_start+i].connected_weight[y*input_image_l+x]>weight_max){
					  weight_max = NeuronList[output_index_start+i].connected_weight[y*input_image_l+x];
				  }
				  if(NeuronList[output_index_start+i].connected_weight[y*input_image_l+x]<weight_min){
					  weight_min = NeuronList[output_index_start+i].connected_weight[y*input_image_l+x];
				  }

				}
				//std::cout << std::endl;
		}
		//cout<<"\n\n\n";
	}
	//printf("weightMaxIs%f;weightMinIs%f\n", weight_max, weight_min);

	if(plot){
		float weight_diff = weight_max - weight_min;

		cimg_library::CImg<unsigned char> image("color.jpg");
		image.resize(input_image_w, input_image_l);
		/*
		cimg_library::CImgDisplay display(image, "Click a point");
		while (!display.is_closed())
		    {
		        display.wait();
		        if (display.button() && display.mouse_y() >= 0 && display.mouse_x() >= 0)
		        {
		            const int y = display.mouse_y();
		            const int x = display.mouse_x();

		            unsigned char randomColor[3];
		            randomColor[0] = rand() % 256;
		            randomColor[1] = rand() % 256;
		            randomColor[2] = rand() % 256;

		            image.draw_point(x, y, randomColor);
		        }
		        image.display(display);
		    }
		*/
		for(int i=0;i<(output_index_stop-output_index_start);i++){

			int current_index = i+output_index_start;
			int img_i;
			int img_j;
			int img_k;
			weight_max = 0;
			weight_min = 10000;
			for (int y=0; y<input_image_w; ++y) {
				for (int x=0; x<input_image_l; ++x) {
				  //std::cout << ((image[y*28+x] == 0.0)? ' ' : '*');
				  //std::cout << std::to_string(int(100*(NeuronList[output_index_start+i].connected_weight[y*28+x]))) << ' ';

				  if(NeuronList[output_index_start+i].connected_weight[y*input_image_l+x]>1){
					  NeuronList[output_index_start+i].connected_weight[y*input_image_l+x] = 1;
				  }
				  if(NeuronList[output_index_start+i].connected_weight[y*input_image_l+x]<-1){
					  NeuronList[output_index_start+i].connected_weight[y*input_image_l+x] = -1;
				  }

				}
				//std::cout << std::endl;
			}
			for (int y=0; y<input_image_w; ++y) {
				for (int x=0; x<input_image_l; ++x) {
				  //std::cout << ((image[y*28+x] == 0.0)? ' ' : '*');
				  //std::cout << std::to_string(int(100*(NeuronList[output_index_start+i].connected_weight[y*28+x]))) << ' ';

				  if(NeuronList[output_index_start+i].connected_weight[y*input_image_l+x]>weight_max){
					  weight_max = NeuronList[output_index_start+i].connected_weight[y*input_image_l+x];
				  }
				  if(NeuronList[output_index_start+i].connected_weight[y*input_image_l+x]<weight_min){
					  weight_min = NeuronList[output_index_start+i].connected_weight[y*input_image_l+x];
				  }

				}
				//std::cout << std::endl;
			}

			weight_diff = weight_max - weight_min;
			int pix_count = 0;
			bool plot_three_channel = true;
			if(input_image_channel==1) plot_three_channel = false;

			if(input_image_channel==1)	bool plot_three_channel = false;
			for(img_k=0;img_k<3;img_k++){
				for (img_i=0;img_i<input_image_w;img_i++){
					for (img_j=0;img_j<input_image_l;img_j++){
						pix_count = img_k*input_image_w*input_image_l + img_i*input_image_l + img_j;
						float weight_raw = NeuronList[current_index].connected_weight[img_i*input_image_l+img_j];
						if(plot_three_channel)  weight_raw = NeuronList[current_index].connected_weight[pix_count];

						image(img_j, img_i, 0, img_k) = 255*(weight_raw-weight_min)/weight_diff;
						//pix_count ++;
						//printf("pixel%d, %d, signal is: %f \n",img_i, img_j, img_temp);
					}
				}
			}
			/*
			cimg_library::CImgDisplay main_disp(image,"Synapse_Conductance");
			while (!main_disp.is_closed()) {
					main_disp.wait();
			}
			*/
			string out_file_name = plot_prefix + "weight_out_index_"+to_string(current_index)+".jpg";
			image.save(out_file_name.c_str());
		}
	}


}


void print_signal(Neuron *NeuronList, int start, int end){
	for (int y=0; y<input_image_l; ++y) {
		for (int x=0; x<input_image_w; ++x) {
				//std::cout << ((image[y*28+x] == 0.0)? ' ' : '*');
				int index = start + x + y*28;
				if(index>end) return;
				std::cout << std::to_string(int(NeuronList[index].state[1])) << ' ';
			}
			std::cout << std::endl;
	}
}



void data_check(Neuron *NeuronList, float *log_total_spike, int network_size, int mnist_start_index, int mnist_end_index, int function_select, string plot_prefix){

	int SIZE_PER_SIDE = sqrt(network_size)+1;
	dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );
	dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));

	int plot = 1;
	int start_index = 0;
	int end_index = start_index+30;

	if(function_select==0){
		check_weight<<<dimGrid, dimBlock>>>(NeuronList, network_size);
	}
	if(function_select==1){
		check_total_spike<<<dimGrid, dimBlock>>>(log_total_spike, network_size);
	}
	if(function_select==2){

		if(plot>0){
			cout<<"Saving conductance visualization"<<endl;
			print_connected_in_weight(NeuronList, start_index, end_index, 1, plot_prefix);
		}else{
			print_connected_in_weight(NeuronList, start_index, end_index, 0, plot_prefix);
		}
	}
	if(function_select==3){
		print_signal(NeuronList, mnist_start_index, mnist_end_index);
	}


}

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


void find_max(){

}

void read_by_type(float *mnist_img, int *mnist_label, float *output_array, int type, int total_num, int *result_size){
	int count = 0;
	int total_pixel = 28*28;

	for(int i=0;i<total_num;i++){
		if (mnist_label[i]==type){
    		for(int j=0;j<total_pixel;j++){
    			output_array[count*total_pixel+j] = mnist_img[i*total_pixel+j];
    		}
			count ++;
		}

	}

	result_size[0] = count;
	printf("number_of_%d_img_is:%d\n",type, count);
}




void MNIST_labeling(string input_file_starter, int size, float *input_array_1, int *input_array_2, float *output_array_1, int *output_array_2, int main_neuron_num, int function_select, int function_select_2){

	if (function_select == 0){
		int *fire_count = new int[main_neuron_num];
		string output_file_name = "MNIST_labeled_data.csv";
	}else if (function_select == 1){
		read_by_type(input_array_1, input_array_2, output_array_1, function_select_2, size, output_array_2);
	}

}

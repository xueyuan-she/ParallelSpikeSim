#include "header.h"

__global__ void read_array(float **d_instance_matrix_array, float **d_convolution_result_array){

	printf("Reading array on GPU: ");
	printf("%f \n", d_instance_matrix_array[1][1]);
	printf("%f \n", d_convolution_result_array[0][0]);
}

int CNN_util(CNN_struct *settings, float **d_instance_matrix_array, float **d_convolution_result_array, float **h_instance_matrix_array, float **h_convolution_result_array, int function_select){


	if (function_select==0){//initialize arrays
		//first initialize instance_matrix, this is the input to convolution kernel
		int instance_array_size = CNN_total_layer_num;

		for (int i=0;i<instance_array_size;i++){
			int matrix_size = settings->layer[i].neuron_num;
			float *temp = new float[matrix_size];
			for(int j=0;j<matrix_size; j++) temp[j] = 0;
		    cudaMalloc((void **)&h_instance_matrix_array[i], matrix_size * sizeof(float));
		    printf("For layer %d Matrix size for h_instance_matrix_array is %d\n", i, matrix_size);
		    cudaMemcpy(h_instance_matrix_array[i], temp, matrix_size * sizeof(float), cudaMemcpyHostToDevice);
		}
		cudaMemcpy(d_instance_matrix_array, h_instance_matrix_array, instance_array_size* sizeof(float*), cudaMemcpyHostToDevice);


		int convolution_result_size = CNN_total_layer_num - 1;

		for (int i=0;i<convolution_result_size;i++){
			//this is the output of convolution kernel
			int matrix_size = settings->layer[i+1].neuron_num;
			float *temp = new float[matrix_size];
			for(int j=0;j<matrix_size; j++) temp[j] = 0;
			cudaMalloc((void **)&h_convolution_result_array[i], matrix_size * sizeof(float));
		    printf("Between layer %d and %d Matrix size for h_convolution_result_array is %d\n", i, i+1, matrix_size);
			cudaMemcpy(h_convolution_result_array[i], temp, matrix_size * sizeof(float), cudaMemcpyHostToDevice);
		}
		cudaMemcpy(d_convolution_result_array, h_convolution_result_array, convolution_result_size*sizeof(float*), cudaMemcpyHostToDevice);


	}

    dim3 dimBlock(1, 1);
    dim3 dimGrid(1, 1);
    read_array<<<dimGrid, dimBlock>>>(d_instance_matrix_array, d_convolution_result_array);


	return 0;
}

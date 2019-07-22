#include <iostream>
#include <time.h>
#include <vector>
#include <string>
#include "header.h"
#include <stdlib.h>
#include <streambuf>
#include <sstream>
#include <fstream>
#include <math.h>
#include "CImg.h"
#include <curand.h>
#include <curand_kernel.h>
#include <assert.h>
#include "cifar10_reader.hpp"
#include <boost/filesystem.hpp>

#define tau 10
#define exp_coeff 1.442695
#define SIZE 50000  //for ROI, use 30000
#define MAX_TIME 2500000 //in ms
#define TEST_TIME 1000

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      //if (abort) exit(code);
   }
}


#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return;}} while(0)

__global__ void print_log (float *log_v, int i){
	printf("_%f_", log_v[i]);
	//printf("time of %d: _%f_",i,log_v[i]);
}

__global__ void random (float *random_number, int rand_number_size, curandState_t *state){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    if (index>=rand_number_size) return;
	random_number[index] = (curand(&state[index])%1000)/1000.0;
    //printf("rand_gen_complete\n");
	//if(index==31)printf("The no.%d of random nubmer is %f\n", index, random_number[index]);
	//if(index==31)printf("%f|", random_number[index]);

}

__global__ void rand_init (unsigned int seed, int size, curandState_t *states){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    if (index>=size) return;
	curand_init(seed, index, 0, &states[index]);
    //printf("rand_init_complete\n");
}

__global__ void read_filter_GPU_one_layer (CNN_struct *settings, float *device_filter_array, int layer_num){
	int counter = 0;
	printf("Printing filter array on GPU\n");

	int filter_size = settings->layer[layer_num].conv_setting.filter_depth * settings->layer[layer_num].conv_setting.filter_width * settings->layer[layer_num].conv_setting.filter_length * settings->layer[layer_num].depth;
	for(int j=0;j<filter_size;j++){
		printf("%f ", device_filter_array[j]);
		counter ++;
	}
	printf("\n");
}

__global__ void read_filter_GPU (CNN_struct *settings, float **device_filter_array){
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


__global__ void lateral_inhibition_child (Neuron *NeuronList, int network_size, int inhibit_time, float start_depth, float end_depth){
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;


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
	//printf("%d %d| ", index, network_size);
	//printf("%d | ", index);
	NeuronList[index].state[7] = inhibit_time;	//
	NeuronList[index].state[0] = NeuronList[index].state[0] - 7;//NeuronList[index].param[2];				//change mem potential to reset_value
	//float *result = std::find(std::begin(NeuronList[index].state), std::end(NeuronList[index].state), 123);

}

__global__ void lateral_inhibition_mother_thread (Neuron *NeuronList, int network_size, int inhibit_time, CNN_struct *CNN_setttings, int *spike_flag){
	if (threadIdx.x==0){
    	for(int layer_iter=0;layer_iter<CNN_total_layer_num;layer_iter++){
    		//if (layer_iter==0) printf("Layer[%d] SpikeFlag: %d\n", layer_iter, spike_flag[layer_iter]);
            if(spike_flag[layer_iter]>0){//use lateral inhibition
            	int SIZE_PER_SIDE = sqrt((float)network_size)+1;
            	float start_depth = CNN_setttings->layer[layer_iter].first_depth_id - 0.1;
            	float end_depth = CNN_setttings->layer[layer_iter].last_depth_id + 0.1;
            	dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );
            	dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));
            	//printf("Start_depth: %f, end_depth: %f||", start_depth, end_depth);
        		lateral_inhibition_child<<<dimGrid, dimBlock>>>(NeuronList, network_size, inhibit_time, start_depth, end_depth);
            }
    		spike_flag[layer_iter] = 0;
    	}

	}
	__syncthreads();

}

int find_max_potential(Neuron *NeuronList){
	int max_index = 0;
	float max_v = -100;

	for(int i=0; i<SPIKING_NEURON_NUM; i++){
		printf("v_of_%d_is%f\n", i, NeuronList[i].state[0]);
		if (NeuronList[i].state[0]>max_v){

			max_index = i;
			max_v = NeuronList[i].state[0];
		}
	}
	return max_index;
}

void neuron_list_init(Neuron *NeuronList){

	int i = 0;
	while (i<SIZE){
		int j = 0;
		NeuronList[i].index = -1;
		NeuronList[i].type = -1;

		while (j<MAX_CONNECTION){
			NeuronList[i].connected_in[j] = 0;
			NeuronList[i].connected_weight[j] = 0;
			j++;
		}
		j = 0;
		while(j<8){
			NeuronList[i].param[j] = -1;
			NeuronList[i].state[j] = 0;
			j++;
		}
	i++;
	}
}

void input_neuron_list_init(Input_neuron *NeuronList, int network_size){

	int i = 0;
	while (i<network_size){
		int j = 0;
		NeuronList[i].index = i+1;
		NeuronList[i].type = 4;

		while(j<8){
			NeuronList[i].param[j] = -1;
			NeuronList[i].state[j] = 0;
			j++;
		}
	i++;
	}
}
void neuron_list_init(Neuron *NeuronList, int network_size){

	int i = 0;
	while (i<network_size){
		int j = 0;
		NeuronList[i].index = -1;
		NeuronList[i].type = -1;

		while (j<MAX_CONNECTION){
			NeuronList[i].connected_in[j] = 0;
			NeuronList[i].connected_weight[j] = 0;
			j++;
		}
		j = 0;
		while(j<8){
			NeuronList[i].param[j] = -1;
			NeuronList[i].state[j] = 0;
			j++;
		}
	i++;
	}
}

void find_fired(Neuron *NeuronList, int *fire_list, int *fired_no){
	int i = 0;
	int fired_count = 0;
	//int fire_list [SIZE] = { };
	while(i<SIZE){
		if(i<5){
			//printf("The no. %d neuron is timed: %f\n", i, NeuronList[i].state.head->data);
			//NeuronList[2].state.display();
		}
		if (NeuronList[i].state[0] > 0.1){
			//printf("state of %d is %f\n", i, NeuronList[i].state[0]);
			fire_list[fired_count] = i+1;
			fired_count ++;
			//printf("The no. of fired neuron is %d\n", i);
		}
		i++;
	}
	*fired_no = fired_count;
}

void check_neuron(Neuron *NeuronList, int start_index, int end_index){
	cout<<"===check_neuron==="<<endl;
	for(int i=start_index; i<=end_index; i++){
		cout<<NeuronList[i].index<<" "<<NeuronList[i].type<<" ";
		for(int j=0; j<8; j++){
			cout<<NeuronList[i].param[j]<<" ";
		}
		for(int j=0; j<8; j++){
			cout<<NeuronList[i].state[j]<<" ";
		}
	cout<<endl;
	}
}

void izh_parameter_init(float *izh_parameters){
	izh_parameters[0] = SIZE;
}

void init_log_v (float *log_v){
	int i = 0;
	while (i<MAX_TIME){
		log_v[i] = 0;
		i++;
	}
}

void init_data_log (float *log_v_host, float *log_spike_host, float *log_total_spike_host, int inter){
	int i=0;
	while(i<inter){
		log_v_host[i] = 0;
		log_spike_host[i] = 0;
		i++;
	}

	int j=0;
	while(j<SIZE){
		log_total_spike_host[j] = 0;
		j++;
	}


}

void spiking_learning_label(){
	int total_neuron_num = 4300;

    Neuron *NeuronList = new Neuron[total_neuron_num];
	neuron_list_init(NeuronList, total_neuron_num);
	cout<<"1"<<endl;
	read_neuron_list(NeuronList, 1, "device2_output_network.txt");
	float *log_total_spike_host = new float[total_neuron_num];
	for(int i=0; i < total_neuron_num; i++){
		log_total_spike_host[i] = 0;
	}
	float *log_total_spike;
    gpuErrchk( cudaMalloc((void **)&log_total_spike, total_neuron_num * sizeof(float)) );
    gpuErrchk( cudaMemcpy(log_total_spike,log_total_spike_host,total_neuron_num*sizeof(float),cudaMemcpyHostToDevice) );
	cout<<"2"<<endl;

    int mnist_start_index = 0;
    int mnist_end_index = input_image_w*input_image_l*input_image_channel;

    data_check(NeuronList,log_total_spike,total_neuron_num, mnist_start_index, mnist_end_index, 2, "");
}

float spiking_learning_label(string network_data, string flag_file, int input_index, int num_test, int function_select, int data_set_select){
	cudaSetDevice(1);
	int spiking_neuron_num = SPIKING_NEURON_NUM;
	int output_layer_neuron_num = OUTPUT_LAYER_NEURON_NUM;
	//spiking_neuron_num = 1000;
	int training_set_number = 10000;
	int input_neuron_num = input_image_w*input_image_l*input_image_channel;
	int *mnist_label = new int[training_set_number];
	Neuron *NeuronList = new Neuron[SIZE];
	float *mnist_img = new float[28*28*training_set_number];
	int *test_log = new int[num_test];
	int wrong_class_log[10];
	float label_sum_v[output_layer_neuron_num][10];

	for(int li=0; li<10; li++){
		wrong_class_log[li] = 0;
		for(int lj=0;lj<output_layer_neuron_num;lj++){
			label_sum_v[lj][li] = 0;
		}
	}

	for(int ti=0; ti<num_test; ti++){
		test_log[ti] = 0;
	}
	//cout<<"In Spiking Labeling"<<endl;
	//

	//defining all data sets 0 is MNIST, 1 is fashion_mnist
	string dataset_train;
	string dataset_train_label;
	string dataset_test;
	string dataset_test_label;

	if (data_set_select==0){
		dataset_train = "train-images-idx3-ubyte";
		dataset_train_label = "train-labels-idx1-ubyte";
		dataset_test = "t10k-images-idx3-ubyte";
		dataset_test_label = "t10k-labels-idx1-ubyte";
	} else if (data_set_select==1){
		dataset_train = "fashion_train-images-idx3-ubyte";
		dataset_train_label = "fashion_train-labels-idx1-ubyte";
		dataset_test = "fashion_t10k-images-idx3-ubyte";
		dataset_test_label = "fashion_t10k-labels-idx1-ubyte";
	}else if (data_set_select==2){
		dataset_train = "SVHN_train_data";
		dataset_train_label = "SVHN_train_label";
		dataset_test = "SVHN_test_data";
		dataset_test_label = "SVHN_test_label";
	}else if (data_set_select==3){
		dataset_train = "CIFAR_train_data";
		dataset_train_label = "CIFAR_train_label";
		dataset_test = "CIFAR_test_data";
		dataset_test_label = "CIFAR_test_label";
	}
	else if (data_set_select==4){
		dataset_train = "train_dataset_noisy";
		dataset_train_label = "train-labels-idx1-ubyte";
		dataset_test = "test_dataset_noisy";
		dataset_test_label = "t10k-labels-idx1-ubyte";
	}


	else{
		cout<<"No data set specified"<<endl;
		return 1;
	}


	string image_file = dataset_test;
	if(function_select==2) image_file = dataset_train;
	string image_label_file = dataset_test_label;
	if(function_select==2) image_label_file = dataset_train_label;
	MNIST_read_label(image_label_file, mnist_label, training_set_number);
	int flag[output_layer_neuron_num][10];
	string file_name = flag_file;
	//read_csv_label(file_name, flag);

	ifstream file (file_name);
	string value;

	for(int i1=0;i1<10;i1++){//read flag_csv

			string line;
			getline(file, line);
			if(!file.good()){
				break;
			}
			stringstream iss(line);

			for(int i2=0;i2<output_layer_neuron_num;i2++){
				string val;

				getline(iss, val, ',');

				//if(!iss.good()) break;
				flag[i2][i1] = atoi(val.c_str());
				//cout<<" ["<<to_string(i2)<<","<<to_string(i1)<<"] is "<<to_string(flag[i2][i1])<< " \n";
		}
	}
	file.close();


	neuron_list_init(NeuronList);
	read_neuron_list(NeuronList, 1, network_data);

	//print all weight to file;
	string weight_file_name = "all_weight.csv";
    ofstream myfile_weight (weight_file_name);
    if (myfile_weight.is_open()){
    	//myfile << "This is a new test\n";
    	for(int i=input_neuron_num; i < input_neuron_num+1000; i++){
    		//printf("_%f_", log_v_host[i]);
    		int j = 0;
    		while(NeuronList[i].connected_in[j] > 0.1){
    			myfile_weight << NeuronList[i].connected_weight[j] << ", ";
    			j++;
    		}
    		myfile_weight<<endl;
    	}
    	myfile_weight.close();
    }
	float *log_total_spike = new float[SIZE];
    data_check(NeuronList,log_total_spike,SIZE, 0, input_neuron_num, 2, "");

}
//
//void run_test(){
//	//1. Test of single neuron reaction under constant input current
//
//	Neuron *NeuronList = new Neuron[SIZE];
//	curandState_t *states;
//	float *random_number_list = new float[SIZE];
//	float *log_v_host = new float[MAX_TIME];
//	float *log_spike_host = new float[MAX_TIME];
//	float *log_total_spike_host = new float[SIZE];
//	init_log_v(log_v_host);
//
//	neuron_list_init(NeuronList);
//	read_neuron_list(NeuronList, 1, "data_test.txt");
//	for(int z=0;z<10;z++){
//		//printf("=%d=",NeuronList[z].type);
//	}
//	Neuron *Neuron_list_device;
//	Neuron *old_device_neurons;
//	float *random_number_list_device;
//	float *log_v;
//	float *log_spike;
//	float *log_total_spike;
//
//	int SIZE_PER_SIDE = sqrt(SIZE)+1;
//    dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );
//    dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));
//	dim3 print_grid(1);
//	dim3 print_block(1);
//
//    cudaMalloc((void **)&Neuron_list_device, SIZE*sizeof(Neuron));
//    cudaMalloc((void **)&old_device_neurons, SIZE*sizeof(Neuron));
//    cudaMalloc((void **)&random_number_list_device,SIZE*sizeof(float));
//    cudaMalloc((void **)&states, SIZE * sizeof(curandState_t));
//    cudaMalloc((void **)&log_v, MAX_TIME * sizeof(float));
//    cudaMalloc((void **)&log_spike, MAX_TIME * sizeof(float));
//    cudaMalloc((void **)&log_total_spike, SIZE * sizeof(float));
//
//
//    rand_init<<<dimGrid,dimBlock>>>(time(0), SIZE, states);
//
//    cudaMemcpy(Neuron_list_device,NeuronList,SIZE*sizeof(Neuron),cudaMemcpyHostToDevice);
//    cudaMemcpy(old_device_neurons,NeuronList,SIZE*sizeof(Neuron),cudaMemcpyHostToDevice);
//    cudaMemcpy(random_number_list_device, random_number_list, SIZE*sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(log_v,log_v_host,MAX_TIME*sizeof(float),cudaMemcpyHostToDevice);
//    cudaMemcpy(log_total_spike,log_total_spike_host,SIZE*sizeof(float),cudaMemcpyHostToDevice);
//
//    int network_size = SIZE;
//    int time = 0;
//    int max_time = MAX_TIME;
//    while (time<max_time){
//        //random<<<dimGrid,dimBlock>>>(random_number_list_device, states);
//        neuron_test(Neuron_list_device, old_device_neurons, random_number_list_device, network_size, log_v, log_spike, log_total_spike, time);
//
//        cudaMemcpy(old_device_neurons,Neuron_list_device,sizeof(Neuron)*SIZE,cudaMemcpyDeviceToDevice);
//
//        //print_log<<<print_grid,print_block>>>(log_v, 34);//correct this point
//    	time ++;
//    }
//
//	int j;
//	for(j=0;j<MAX_TIME;j++){
//		//print_log<<<print_grid,print_block>>>(log_v, j);
//	}
//	//cudaDeviceSynchronize();
//    cudaMemcpy(NeuronList,Neuron_list_device,SIZE*sizeof(Neuron),cudaMemcpyDeviceToHost);
//    cudaMemcpy(log_v_host,log_v,MAX_TIME*sizeof(float),cudaMemcpyDeviceToHost);
//    cudaMemcpy(log_spike_host,log_spike,MAX_TIME*sizeof(float),cudaMemcpyDeviceToHost);
//    cudaMemcpy(log_total_spike_host,log_total_spike,SIZE*sizeof(float),cudaMemcpyDeviceToHost);
//    //printf("cpy Done");
//
//    //====write to file=====
//    ofstream myfile ("out_v.csv");
//    if (myfile.is_open()){
//    	//myfile << "This is a new test\n";
//    	for(int i=0; i < MAX_TIME; i++){
//    		//printf("_%f_", log_v_host[i]);
//    		myfile << log_v_host[i] << ", ";
//    	}
//    	myfile.close();
//    }
//
//    ofstream myfile_2 ("out_spike.csv");
//    if (myfile_2.is_open()){
//    	//myfile << "This is a new test\n";
//    	for(int i=0; i < MAX_TIME; i++){
//    		//printf("_%f_", log_v_host[i]);
//    		myfile_2 << log_spike_host[i] << ", ";
//    	}
//    	myfile_2.close();
//    }
//
//    ofstream myfile_3 ("out_total_spike.csv");
//    if (myfile_3.is_open()){
//    	//myfile << "This is a new test\n";
//    	for(int i=0; i < SIZE; i++){
//    		//printf("_%f_", log_v_host[i]);
//    		myfile_3 << log_total_spike_host[i] << ", ";
//    	}
//    	myfile_3.close();
//    }
//
//    //===clean up===
//    delete[] random_number_list;
//    delete[] log_v_host;
//	delete[] NeuronList;
//	delete[] log_spike_host;
//
//
//	cudaFree(states);
//	cudaFree(log_v);
//	cudaFree(log_spike);
//	cudaFree(Neuron_list_device);
//	cudaFree(old_device_neurons);
//	cudaFree(random_number_list_device);
//
//}
//
//void run_Spiking(){
//	Neuron *NeuronList = new Neuron[SIZE];
//
//	neuron_list_init(NeuronList);
//	read_neuron_list(NeuronList, 1, "data_tsp.txt");
//
//	Neuron *Neuron_list_device;
//	Neuron *old_device_neurons;
//
//	int SIZE_PER_SIDE = sqrt(SIZE)+1;
//    dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );
//    dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));
//
//
//    cudaMalloc((void **)&Neuron_list_device, SIZE*sizeof(Neuron));
//    cudaMalloc((void **)&old_device_neurons, SIZE*sizeof(Neuron));
//    //random<<<dimGrid,dimBlock>>>(random_number_list_device,states,number_of_block_y, number_of_threads_y);
//
//    cudaMemcpy(Neuron_list_device,NeuronList,SIZE*sizeof(Neuron),cudaMemcpyHostToDevice);
//    cudaMemcpy(old_device_neurons,NeuronList,SIZE*sizeof(Neuron),cudaMemcpyHostToDevice);
//
//    //int network_size = SIZE;
//    int time = 0;
//    int max_time = MAX_TIME;
//
//    while (time<max_time){
//    	//printf("|||||time is %d|||||\n", time);
//        //kernel_spiking(Neuron_list_device, network_size);
//        cudaMemcpy(old_device_neurons,Neuron_list_device,sizeof(Neuron)*SIZE,cudaMemcpyDeviceToDevice);
//    	time ++;
//    }
//    cudaMemcpy(NeuronList,Neuron_list_device,SIZE*sizeof(Neuron),cudaMemcpyDeviceToHost);
//
//	delete[] NeuronList;
//
//	cudaFree(Neuron_list_device);
//}
//
//void run_Stoc(){
//
//	Neuron *NeuronList = new Neuron[SIZE];
//	curandState_t *states;
//	float *random_number_list = new float[SIZE];
//
//	neuron_list_init(NeuronList);
//	read_neuron_list(NeuronList, 1, "data_tsp.txt");
//
//	Neuron *Neuron_list_device;
//	Neuron *old_device_neurons;
//	float *random_number_list_device;
//
//	int SIZE_PER_SIDE = sqrt(SIZE)+1;
//    dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );
//    dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));
//
//
//    cudaMalloc((void **)&Neuron_list_device, SIZE*sizeof(Neuron));
//    cudaMalloc((void **)&old_device_neurons, SIZE*sizeof(Neuron));
//    cudaMalloc((void **)&random_number_list_device,SIZE*sizeof(float));
//    cudaMalloc((void **)&states, SIZE * sizeof(curandState_t));
//
//    rand_init<<<dimGrid,dimBlock>>>(time(0), SIZE, states);
//    //random<<<dimGrid,dimBlock>>>(random_number_list_device,states,number_of_block_y, number_of_threads_y);
//
//    cudaMemcpy(Neuron_list_device,NeuronList,SIZE*sizeof(Neuron),cudaMemcpyHostToDevice);
//    cudaMemcpy(old_device_neurons,NeuronList,SIZE*sizeof(Neuron),cudaMemcpyHostToDevice);
//    cudaMemcpy(random_number_list_device,random_number_list,SIZE*sizeof(float),cudaMemcpyHostToDevice);
//
//
//    int network_size = SIZE;
//    int time = 0;
//    int max_time = MAX_TIME;
//    while (time<max_time){
//    	//printf("|||||time is %d|||||\n", time);
//
//        random<<<dimGrid,dimBlock>>>(random_number_list_device, SIZE,states);
//        kernel_neuron(Neuron_list_device, old_device_neurons, random_number_list_device, network_size);
//        cudaMemcpy(old_device_neurons,Neuron_list_device,sizeof(Neuron)*SIZE,cudaMemcpyDeviceToDevice);
//    	time ++;
//    }
//    cudaMemcpy(NeuronList,Neuron_list_device,SIZE*sizeof(Neuron),cudaMemcpyDeviceToHost);
//    //printf("cpy Done");
//    int *fired_list = new int[SIZE];
//    int fired_no;
//    find_fired(NeuronList, fired_list, &fired_no);
//	printf("fired neuron no. is %d\n", fired_no);
//
//    for (int i = 0; i<fired_no; i++){
//    	//printf("fired neuron is %d\n", fired_list[i]);
//    }
//
//
//    delete[] random_number_list;
//    delete[] fired_list;
//	delete[] NeuronList;
//
//	cudaFree(Neuron_list_device);
//	cudaFree(old_device_neurons);
//	cudaFree(random_number_list_device);
//}
//
//void run_ROI(){
//	//ROI
//	//First read image
//	cimg_library::CImg<unsigned char> image("color_small.jpg");
//	float signal_max = 0.6;
//	float signal_min = 1.8;
//	float img_signal [img_width][img_len][3];
//
//	//unsigned char* ptr = image.data(10,10, 0, 1); // get pointer to pixel @ 10,10
//	//unsigned char pixel = *ptr;
//
//	int img_i;
//	int img_j;
//	int img_k;
//	for (img_i=0;img_i<img_width;img_i++){
//		for (img_j=0;img_j<img_len;img_j++){
//			for(img_k=0;img_k<3;img_k++){
//				float img_temp = (float)image(img_i, img_j, 0, img_k)/255;
//				img_temp = img_temp*(signal_max-signal_min)+signal_min; //
//				img_signal[img_i][img_j][img_k] = img_temp;
//				//printf("pixel%d, %d, signal is: %f \n",img_i, img_j, img_temp);
//			}
//		}
//	}
//	//finish reading image
//	int signal_start_1 = img_width*img_len*3;
//	int signal_end_1 = img_width*img_len*6;
//
//	Neuron *NeuronList = new Neuron[SIZE];
//	curandState_t *states;
//	float *random_number_list = new float[SIZE];
//	float *log_v_host = new float[MAX_TIME];
//	float *log_spike_host = new float[MAX_TIME];
//	float *log_total_spike_host = new float[SIZE];
//
//	init_log_v(log_v_host);
//	neuron_list_init(NeuronList);
//	//printf("=0=\n");
//	read_neuron_list(NeuronList, 1, "visual_IZH.txt");
//
//	Neuron *Neuron_list_device;
//	Neuron *old_device_neurons;
//	float *random_number_list_device;
//	float *log_v;
//	float *log_spike;
//	float *log_total_spike;
//	//printf("=1=\n");
//	int SIZE_PER_SIDE = sqrt(SIZE)+1;
//    dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );
//    dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));
//	dim3 print_grid(1);
//	dim3 print_block(1);
//
//    cudaMalloc((void **)&Neuron_list_device, SIZE*sizeof(Neuron));
//    cudaMalloc((void **)&old_device_neurons, SIZE*sizeof(Neuron));
//    cudaMalloc((void **)&random_number_list_device,SIZE*sizeof(float));
//    cudaMalloc((void **)&states, SIZE * sizeof(curandState_t));
//    cudaMalloc((void **)&log_v, MAX_TIME * sizeof(float));
//    cudaMalloc((void **)&log_spike, MAX_TIME * sizeof(float));
//    cudaMalloc((void **)&log_total_spike, SIZE * sizeof(float));
//
//    rand_init<<<dimGrid,dimBlock>>>(time(0), SIZE, states);
//
//    cudaMemcpy(Neuron_list_device,NeuronList,SIZE*sizeof(Neuron),cudaMemcpyHostToDevice);
//    cudaMemcpy(old_device_neurons,NeuronList,SIZE*sizeof(Neuron),cudaMemcpyHostToDevice);
//    cudaMemcpy(random_number_list_device, random_number_list, SIZE*sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(log_v,log_v_host,MAX_TIME*sizeof(float),cudaMemcpyHostToDevice);
//    cudaMemcpy(log_total_spike,log_total_spike_host,SIZE*sizeof(float),cudaMemcpyHostToDevice);
//
//    int network_size = SIZE;
//    int time = 0;
//    int max_time = MAX_TIME;
//
//    ROI_drive(old_device_neurons, (float *)img_signal, network_size, signal_start_1, signal_end_1, 1);
//    ROI_drive(old_device_neurons, (float *)img_signal, network_size, signal_start_1, signal_end_1, 0);
//    cudaMemcpy(Neuron_list_device,old_device_neurons,sizeof(Neuron)*SIZE,cudaMemcpyDeviceToDevice);
//
//    cudaDeviceSynchronize();
//
//    while (time<max_time){
//        //random<<<dimGrid,dimBlock>>>(random_number_list_device, states);
//        neuron_test(Neuron_list_device, old_device_neurons, random_number_list_device, network_size, log_v, log_spike, log_total_spike, time);
//        cudaDeviceSynchronize();
//        cudaMemcpy(old_device_neurons,Neuron_list_device,sizeof(Neuron)*SIZE,cudaMemcpyDeviceToDevice);
//        //cudaDeviceSynchronize();
//        //print_log<<<print_grid,print_block>>>(log_v, 34);//correct this point
//    	time ++;
//    }
//
//	int j;
//	for(j=0;j<MAX_TIME;j++){
//		//print_log<<<print_grid,print_block>>>(log_v, j);
//	}
//	//cudaDeviceSynchronize();
//    cudaMemcpy(NeuronList,Neuron_list_device,SIZE*sizeof(Neuron),cudaMemcpyDeviceToHost);
//    cudaMemcpy(log_v_host,log_v,MAX_TIME*sizeof(float),cudaMemcpyDeviceToHost);
//    cudaMemcpy(log_spike_host,log_spike,MAX_TIME*sizeof(float),cudaMemcpyDeviceToHost);
//    cudaMemcpy(log_total_spike_host,log_total_spike,SIZE*sizeof(float),cudaMemcpyDeviceToHost);
//    //printf("cpy Done");
//
//    ofstream myfile ("ROI_out.csv");
//    if (myfile.is_open()){
//    	//myfile << "This is a new test\n";
//    	for(int i=0; i < SIZE; i++){
//    		//printf("_%f_", log_v_host[i]);
//    		myfile << log_total_spike_host[i] << ", ";
//    	}
//    	myfile.close();
//    }
//
//    ofstream myfile_0 ("out_v.csv");
//    if (myfile_0.is_open()){
//    	//myfile << "This is a new test\n";
//    	for(int i=0; i < MAX_TIME; i++){
//    		//printf("_%f_", log_v_host[i]);
//    		myfile_0 << log_v_host[i] << ", ";
//    	}
//    	myfile.close();
//    }
//
//    ofstream myfile_2 ("out_spike.csv");
//    if (myfile_2.is_open()){
//    	//myfile << "This is a new test\n";
//    	for(int i=0; i < MAX_TIME; i++){
//    		//printf("_%f_", log_v_host[i]);
//    		myfile_2 << log_spike_host[i] << ", ";
//    	}
//    	myfile_2.close();
//    }
//
//    //===clean up===
//    delete[] random_number_list;
//    delete[] log_v_host;
//	delete[] NeuronList;
//	delete[] log_spike_host;
//
//
//	cudaFree(states);
//	cudaFree(log_v);
//	cudaFree(log_spike);
//	cudaFree(log_total_spike);
//	cudaFree(Neuron_list_device);
//	cudaFree(old_device_neurons);
//	cudaFree(random_number_list_device);
//
//}
//
//void run_Spiking_learn(string index_prefix, float input_float, float input_float_2, int input_int, int input_int_2, string input_img){
//		cudaSetDevice(0);
//		//set parameters
//
//		int training_time_each_img = input_int;
//		int calculated_total_time = training_time_each_img*50;
//		#undef MAX_TIME
//		#define MAX_TIME calculated_total_time
//		printf("==Training Total Iter: %d==", MAX_TIME);
//
//		float max_frequency = 22; //in Hz default 22
//		float min_frequency = 1;
//		int training_set_number = 60000;
//		int input_neuron_num = input_image_w * input_image_l*input_image_channel;
//		int spiking_neuron_num = SPIKING_NEURON_NUM;
//		int output_layer_neuron_num = OUTPUT_LAYER_NEURON_NUM;
//		int tenpercent_iter = MAX_TIME/10;
//
//		int connection_size = 900;
//		int syn_timer_max = 25;
//		int input_signal_width = 25;	//default 25
//		int inhibition_time = input_int_2;	//default 10
//
//		float target_frequency_param = 0.5;
//		float target_frequency = target_frequency_param*(1/(SPIKING_NEURON_NUM*inhibition_time));
//
//		float *mnist_img = new float[input_neuron_num*training_set_number];
//		string image_file = input_img; //"train-images-idx3-ubyte";
//		MNIST_read_image(image_file, mnist_img, training_set_number);
//		int *mnist_label = new int[training_set_number];
//		string image_label_file = "train-labels-idx1-ubyte";
//		MNIST_read_label(image_label_file, mnist_label, training_set_number);
//		//special_function: learn one category
//		int learn_one_digit = 0;
//		int *num_one_digit_img = new int[1];
//		if(learn_one_digit){
//			//
//			MNIST_labeling("abc", 60000, mnist_img, mnist_label, mnist_img, num_one_digit_img, spiking_neuron_num, 1, 5);
//		}
//
//		//int synapse_size = SIZE*SIZE;
//		Neuron *NeuronList = new Neuron[SIZE];
//		//unsigned char *synapse_timer = new unsigned char[synapse_size];  //this is the array that stores timer used in STPD. e.g Neuron x --->  Neuron y Spike! In the array index [(x-1)*SIZE+(y-1)]  => 1
//		//curandState_t *states;
//		//float *random_number_list = new float[SIZE];
//		float *log_v_host = new float[MAX_TIME];
//		float *log_spike_host = new float[MAX_TIME];
//		float *log_total_spike_host = new float[SIZE];
//		int *spike_flag = new int[1];
//		spike_flag[0] = 0;
//
//		//init_log_v(log_v_host);
//		init_data_log(log_v_host,log_spike_host,log_total_spike_host, MAX_TIME);
//		neuron_list_init(NeuronList);
//		//printf("=0=\n");
//		read_neuron_list(NeuronList, 1, "spike_learning_1000_v1.txt");
//	    //write_neuron_list(NeuronList, "learning_output_confirm.txt", SIZE);
//
//		Neuron *Neuron_list_device;
//		Neuron *old_device_neurons;
//		//unsigned char *snapse_timer_device;
//		float *log_v;
//		float *log_spike;
//		float *log_total_spike;
//		int *spike_flag_device;
//
//		//printf("=1=\n");
//		curandState_t *states;
//		cudaMalloc((void **)&states, SIZE * sizeof(curandState_t));
//		int SIZE_PER_SIDE = sqrt(SIZE)+1;
//	    dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );
//	    dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));
//		dim3 print_grid(1);
//		dim3 print_block(1);
//		rand_init<<<dimGrid,dimBlock>>>(time(0), SIZE, states);
//
//		int rand_numb_size = SPIKING_NEURON_NUM*MAX_CONNECTION;
//
//		float *random_number_list = new float[rand_numb_size];
//		float *random_number_list_device;
//		SIZE_PER_SIDE = sqrt(rand_numb_size)+1;
//		dim3 dimBlock_synapse( ThreadsPerBlock, ThreadsPerBlock );
//		dim3 dimGrid_synapse( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));
//		cudaMalloc((void **)&random_number_list_device,rand_numb_size*sizeof(float));
//		cudaMemcpy(random_number_list_device,random_number_list,rand_numb_size*sizeof(float),cudaMemcpyHostToDevice);
//        random<<<dimGrid_synapse,dimBlock_synapse>>>(random_number_list_device, rand_numb_size, states);
//
//	    cudaMalloc((void **)&Neuron_list_device, SIZE*sizeof(Neuron));
//	    cudaMalloc((void **)&old_device_neurons, SIZE*sizeof(Neuron));
//
//	    //cudaMalloc((void **)&states, SIZE * sizeof(curandState_t));
//	    cudaMalloc((void **)&log_v, MAX_TIME * sizeof(float));
//	    cudaMalloc((void **)&log_spike, MAX_TIME * sizeof(float));
//	    //cudaMalloc((void **)&log_total_spike, SIZE * sizeof(float));
//	    cudaMalloc((void **)&log_total_spike, SIZE * sizeof(float));
//	    cudaMalloc((void **)&spike_flag_device, sizeof(int));
//	    //rand_init<<<dimGrid,dimBlock>>>(time(0), states);
//
//	    cudaMemcpy(Neuron_list_device,NeuronList,SIZE*sizeof(Neuron),cudaMemcpyHostToDevice);
//	    cudaMemcpy(old_device_neurons,NeuronList,SIZE*sizeof(Neuron),cudaMemcpyHostToDevice);
//	    //cudaMemcpy(random_number_list_device, random_number_list, SIZE*sizeof(float), cudaMemcpyHostToDevice);
//	    cudaMemcpy(log_v,log_v_host,MAX_TIME*sizeof(float),cudaMemcpyHostToDevice);
//	    cudaMemcpy(log_spike,log_spike_host,MAX_TIME*sizeof(float),cudaMemcpyHostToDevice);
//	    cudaMemcpy(log_total_spike,log_total_spike_host,SIZE*sizeof(float),cudaMemcpyHostToDevice);
//	    cudaMemcpy(spike_flag_device,spike_flag,sizeof(int),cudaMemcpyHostToDevice);
//
//	    int network_size = SIZE;
//	    int time = 0;
//	    int max_time = MAX_TIME;
//
//	    cudaMemcpy(Neuron_list_device,old_device_neurons,sizeof(Neuron)*SIZE,cudaMemcpyDeviceToDevice);
//	    //first change raw img data into frequency
//	    int mnist_start_index = spiking_neuron_num;
//	    int mnist_end_index = spiking_neuron_num + input_neuron_num;
//	    MNIST_drive(NeuronList, mnist_img, network_size, training_set_number, mnist_start_index, mnist_end_index, max_frequency, min_frequency, 1);
//
//
//	    cudaDeviceSynchronize();
//
//	    //data_check(Neuron_list_device,log_total_spike,SIZE,1);
//	    float *one_mnist_img = new float[input_neuron_num];
//	    int training_img_index = 0;
//	    clock_t iter_start, iter_log;
//	    iter_start = clock();
//	    int log_interval = MAX_TIME/25;
//	    while (time<max_time){
//	        //random<<<dimGrid,dimBlock>>>(random_number_list_device, states);
//	    	//first create an array of 1 MNIST image
//	    	printf("\n iter_%d\n",time);
//	    	if(STOCHASTIC_STDP || STOCHASTIC_ROUNDING){
//	            random<<<dimGrid,dimBlock>>>(random_number_list_device, rand_numb_size, states);
//	    	}
//	    	if(time%log_interval == 0){
//	    		cudaMemcpy(NeuronList,Neuron_list_device,SIZE*sizeof(Neuron),cudaMemcpyDeviceToHost);
//	    		string interval_file_name = "device2_output_at_iter_" + to_string(time) + ".txt";
//	    		write_neuron_list(NeuronList, interval_file_name, network_size);
//	    		//printf("%");
//	    	}
//
//	    	if(time%tenpercent_iter == 0){
//	    		iter_log = clock();
//	    		cout<<to_string(10*(time/tenpercent_iter))<<"% done, time used is: " << (iter_log - iter_start)/1000 << " (ms)" << endl;
//	    	}
//
//	    	if(time%training_time_each_img==0){//at the beginning of each img's training, load into
//	    		for(int i=0;i<input_neuron_num;i++){
//	    			one_mnist_img[i] = mnist_img[training_img_index*input_neuron_num+i];
//	    		}
//	    	    for (int y=0; y<28; ++y) {
//	    	    	    for (int x=0; x<28; ++x) {
//	    	    	      //std::cout << ((one_mnist_img[y*28+x] == 0.0)? ' ' : '*');
//	    	    	      std::cout << std::to_string(int(one_mnist_img[y*28+x])) << ' ';
//	    	    	    }
//	    	    	    std::cout << std::endl;
//	    	    }
//	    		MNIST_drive(Neuron_list_device, one_mnist_img, network_size, training_set_number, mnist_start_index, mnist_end_index, max_frequency, min_frequency, 0);
//	    		MNIST_drive(old_device_neurons, one_mnist_img, network_size,training_set_number, mnist_start_index, mnist_end_index, max_frequency, min_frequency, 0);
//	    		training_img_index ++;
//	    		//confirm the data in signal neuron
//	    		//cudaMemcpy(NeuronList,Neuron_list_device,SIZE*sizeof(Neuron),cudaMemcpyDeviceToHost);
//	    		//data_check(NeuronList,log_total_spike,SIZE, mnist_start_index, mnist_end_index, 3);
//	    		//printf("\n\n\n************************\n\n\n\n");
//	    	}
//	    	spiking_learning_main(Neuron_list_device, old_device_neurons, random_number_list_device, network_size, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, time);
//
//	    	synapse_drive_v1(Neuron_list_device, network_size, syn_timer_max, connection_size, random_number_list_device, input_float, input_float_2);
//	    	if(HOMEOSTASIS_ENABLE){
//				if(time%HOMEOSTASIS_UPDATE_FREQUENCY == 0 && time != 0){
//					spiking_learning_drive(Neuron_list_device, network_size, inhibition_time, log_total_spike, target_frequency, time, log_spike, 0, 1);
//				}
//	    	}
//	        cudaDeviceSynchronize();
//
//	        //if any neuron spikes, run inhibition
//		    cudaMemcpy(spike_flag, spike_flag_device, sizeof(int),cudaMemcpyDeviceToHost);
//		    //printf("AtTime:%d_spike_flag_is:%d\n",time,spike_flag[0]);
//	        if(spike_flag[0]>0){//use lateral inhibition
//	        	//printf("inInhibit\n");
//	        	spiking_learning_drive(Neuron_list_device, network_size, inhibition_time, log_total_spike, target_frequency, time, log_spike, 0, 0);
//	        	spike_flag[0] = 0;
//
//	    	    cudaMemcpy(spike_flag_device,spike_flag,sizeof(int),cudaMemcpyHostToDevice);
//	        }
//
//	        cudaMemcpy(old_device_neurons,Neuron_list_device,sizeof(Neuron)*SIZE,cudaMemcpyDeviceToDevice);
//	        //cudaDeviceSynchronize();
//	    	time ++;
//	    }
//	    //spiking_learning_drive(Neuron_list_device, network_size, inhibition_time, 2);
//		//cudaDeviceSynchronize();
//	    cudaMemcpy(NeuronList,Neuron_list_device,SIZE*sizeof(Neuron),cudaMemcpyDeviceToHost);
//	    cudaMemcpy(log_v_host,log_v,MAX_TIME*sizeof(float),cudaMemcpyDeviceToHost);
//	    cudaMemcpy(log_spike_host,log_spike,MAX_TIME*sizeof(float),cudaMemcpyDeviceToHost);
//	    cudaMemcpy(log_total_spike_host,log_total_spike,SIZE*sizeof(float),cudaMemcpyDeviceToHost);
//
//	    //print out the synapse conductance data
//	    data_check(NeuronList,log_total_spike,SIZE, mnist_start_index, mnist_end_index, 2, "");
//
//	    ofstream myfile ((index_prefix+"device2_spike_of_neuron_out.csv"));
//	    if (myfile.is_open()){
//	    	//myfile << "This is a new test\n";
//	    	for(int i=0; i < SIZE; i++){
//	    		//printf("_%f_", log_v_host[i]);
//	    		myfile << log_total_spike_host[i] << ", ";
//	    	}
//	    	myfile.close();
//	    }
//
//	    ofstream myfile_0 ((index_prefix+"device2_out_v.csv"));
//	    if (myfile_0.is_open()){
//	    	//myfile << "This is a new test\n";
//	    	for(int i=0; i < MAX_TIME; i++){
//	    		//printf("_%f_", log_v_host[i]);
//	    		myfile_0 << log_v_host[i] << ", ";
//	    	}
//	    	myfile.close();
//	    }
//
//	    ofstream myfile_2 ((index_prefix+"device2_spike_of_one.csv"));
//	    if (myfile_2.is_open()){
//	    	//myfile << "This is a new test\n";
//	    	for(int i=0; i < MAX_TIME; i++){
//	    		//printf("_%f_", log_v_host[i]);
//	    		myfile_2 << log_spike_host[i] << ", ";
//	    	}
//	    	myfile_2.close();
//	    }
//
//	    write_neuron_list(NeuronList, (index_prefix+"device2_output_network.txt"), network_size);
//
//	    //===clean up===
//	    //delete[] random_number_list;
//	    delete[] log_v_host;
//		delete[] NeuronList;
//		delete[] log_spike_host;
//
//
//		//cudaFree(states);
//		cudaFree(log_v);
//		cudaFree(log_spike);
//		cudaFree(log_total_spike);
//		cudaFree(Neuron_list_device);
//		cudaFree(old_device_neurons);
//		cudaFree(random_number_list_device);
//}

//void last_layer_learn(string index_prefix, float input_float, float input_float_2, int input_int, int input_int_2, string input_img){
//	/*
//	int training_set_number = 1;
//	int size_per_img = input_image_w * input_image_l*input_image_channel;
//	float *mnist_img = new float[size_per_img*training_set_number];
//	string image_file = "train-images-idx3-ubyte"; //"train-images-idx3-ubyte";
//	MNIST_read_image(image_file, mnist_img, training_set_number);
//	int *mnist_label = new int[training_set_number];
//	string image_label_file = "train-labels-idx1-ubyte";
//	MNIST_read_label(image_label_file, mnist_label, training_set_number);
//
//	float *filter;
//	float *output = new float[size_per_img*training_set_number];
//
//	convolution_kernel(mnist_img, filter, output);
//	img_util(output, "test_output.jpg", 0);
//	*/
//
//
//	CNN_struct *network_config = new CNN_struct;
//	network_config_generator(3, network_config);
//	Neuron *NeuronList_temp = new Neuron[1];
//	CNN_struct *d_network_config;
//	cudaMalloc((void **)&d_network_config,sizeof(CNN_struct));
//	cudaMemcpy(d_network_config,network_config,sizeof(CNN_struct),cudaMemcpyHostToDevice);
//	int total_depth_number = 0;
//	for(int i=0;i<CNN_total_layer_num; i++){
//		total_depth_number = total_depth_number + network_config->layer[i].depth;
//		cout<<"depth number: "<<network_config->layer[i].depth<<endl;
//	}
//
//	cout<<endl<<"Total depth number: "<<total_depth_number<<endl;
//	float **h_filter_array;
//	float **d_filter_array;
//	int filter_array_size = CNN_total_layer_num-1;
//	cudaMalloc(&d_filter_array, filter_array_size*sizeof(float *));
//	h_filter_array = (float**)malloc(filter_array_size * sizeof(float*));
//	filter_util(network_config, NeuronList_temp, SIZE, h_filter_array, d_filter_array, 0);
//
//	/*
//	img_util(mnist_img, "tensorflow_small.png", 1);
//	img_util(mnist_img, "test_output_-1.png", 0);
//
//	float *output = new float[size_per_img*training_set_number];
//
//	int image_bytes = input_image_channel * input_image_l * input_image_w * sizeof(float);
//
//	float* convolution_device_input{nullptr};
//	cudaMalloc(&convolution_device_input, image_bytes);
//	cudaMemcpy(convolution_device_input, mnist_img, image_bytes, cudaMemcpyHostToDevice);
//
//	int filter_in_channel = input_image_channel;
//	int filter_out_channel = input_image_channel;
//	int filter_height = 3;
//	int filter_width = 3;
//	const float kernel_template[3][3] = {
//	{1, 1, 1},
//	{1, -8, 1},
//	{1, 1, 1}
//	};
//	float h_kernel[filter_in_channel][filter_out_channel][filter_height][filter_width];
//	for (int kernel = 0; kernel < filter_in_channel; ++kernel) {
//		for (int channel = 0; channel < filter_out_channel; ++channel) {
//		  for (int row = 0; row < filter_height; ++row) {
//			for (int column = 0; column < filter_width; ++column) {
//			  h_kernel[kernel][channel][row][column] = kernel_template[row][column];
//			}
//		  }
//		}
//	}
//	float* filter{nullptr};
//	cudaMalloc(&filter, sizeof(h_kernel));
//	cudaMemcpy(filter, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);
//
//
//	convolution_kernel(convolution_device_input, filter, output);
//	img_util(output, "test_output_1.png", 0);
//
//	cudaFree(filter);
//	cudaFree(convolution_device_input);
//
//	*/
//
////===========END of CNN special setting-up phase============
//
//	cudaSetDevice(0);
//	//set parameters
//
//	int training_time_each_img = input_int;
//	int calculated_total_time = training_time_each_img*50000;
//	#undef MAX_TIME
//	#define MAX_TIME calculated_total_time
//	printf("==Training Total Iter: %d==", MAX_TIME);
//	int total_neuron_num = 0;
//	int total_spiking_num = 0;
//	for(int i=0;i<CNN_total_layer_num;i++){
//		total_neuron_num += network_config->layer[i].neuron_num;
//		if(i!=0)
//		total_spiking_num += network_config->layer[i].neuron_num;
//	}
//	total_neuron_num += 100;
//	//total_neuron_num = 20000;
//	cout<<endl<<"total neuron num: "<<total_neuron_num<<endl;
//	cout<<"total spiking neuron num: "<<total_spiking_num<<endl;
//	#undef SIZE
//	#define SIZE total_neuron_num
//	#undef SPIKING_NEURON_NUM
//	#define SPIKING_NEURON_NUM total_spiking_num
//
//
//	float max_frequency = 22; //in Hz default 22
//	float min_frequency = 1;
//	int training_set_number = 55000;
//	int input_neuron_num = input_image_w*input_image_l*input_image_channel;
//	int spiking_neuron_num = SPIKING_NEURON_NUM;
//	int output_layer_neuron_num = OUTPUT_LAYER_NEURON_NUM;
//	int tenpercent_iter = MAX_TIME/10;
//	int connection_size = MAX_CONNECTION;
//	int syn_timer_max = 25;
//	int input_signal_width = 25;	//default 25
//	int inhibition_time = input_int_2;	//default 10
//
//	float target_frequency_param = 0.5;
//	float target_frequency = 100;
//	float *mnist_img = new float[input_neuron_num*training_set_number];
//	for(int i=0;i<input_neuron_num*training_set_number;i++) mnist_img[i] = 0;
//	string image_file = "train-images-idx3-ubyte";//"fashion-train-images-idx3-ubyte";//"train_dataset_noisy";//"train_dataset_noisy"; //"train-images-idx3-ubyte";
//	read_filter_data(image_file, mnist_img, training_set_number, input_neuron_num);
//	int *mnist_label = new int[training_set_number];
//	string image_label_file = "train-labels-idx1-ubyte";
//	MNIST_read_label(image_label_file, mnist_label, training_set_number);
//	//special_function: learn one category
//	int learn_one_digit = 0;
//	int *num_one_digit_img = new int[1];
//	if(learn_one_digit){
//		MNIST_labeling("abc", 60000, mnist_img, mnist_label, mnist_img, num_one_digit_img, spiking_neuron_num, 1, 5);
//		printf("Learning only one digit, number of img: %d\n", num_one_digit_img);
//	}
//
//	//int synapse_size = SIZE*SIZE;
//	//cout<<SIZE<<endl;
//    Neuron *NeuronList = new Neuron[SIZE];
//	//unsigned char *synapse_timer = new unsigned char[synapse_size];  //this is the array that stores timer used in STPD. e.g Neuron x --->  Neuron y Spike! In the array index [(x-1)*SIZE+(y-1)]  => 1
//	//curandState_t *states;
//	//float *random_number_list = new float[SIZE];
//	float *log_v_host = new float[MAX_TIME];
//	float *log_spike_host = new float[total_depth_number];
//
//	float *log_total_spike_host = new float[SIZE];
//	for(int i=0; i < SIZE; i++){
//		log_total_spike_host[i] = 0;
//	}
//	int *spike_flag = new int[CNN_total_layer_num];
//	for(int i=0; i < CNN_total_layer_num; i++){
//		spike_flag[i] = 0;
//	}
//	for(int i=0; i<total_depth_number; i++) log_spike_host[i] = 0;
//	//init_log_v(log_v_host);
//	//init_data_log(log_v_host,log_spike_host,log_total_spike_host, MAX_TIME);
//	neuron_list_init(NeuronList, total_neuron_num);
//	//printf("=0=\n");
//	read_neuron_list(NeuronList, 1, "spike_cnn.txt");
//    //write_neuron_list(NeuronList, "learning_output_confirm.txt", SIZE);
//	//check_neuron(NeuronList, 800, 820);
//
//	Neuron *Neuron_list_device;
//	//Neuron *old_device_neurons;
//	//unsigned char *snapse_timer_device;
//	float *log_v;
//	float *log_spike;
//	float *log_spike_default;
//	float *log_total_spike;
//	int *spike_flag_device;
//
//
//    printf("2\n");
//	//printf("=1=\n");
//	//random number function:
//    float rand_list_size_to_total_connection_ratio = 1;
//	int rand_numb_size = SPIKING_NEURON_NUM*MAX_CONNECTION;
//	curandState_t *states;
//	cudaMalloc((void **)&states, rand_numb_size * sizeof(curandState_t));
//	int SIZE_PER_SIDE = sqrt(rand_numb_size)+1;
//    dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );
//    dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));
//	dim3 print_grid(1);
//	dim3 print_block(1);
//    printf("2.1\n");
//	rand_init<<<dimGrid,dimBlock>>>(time(0), rand_numb_size, states);
//
//	float *random_number_list = new float[rand_numb_size];
//	float *random_number_list_device;
//	SIZE_PER_SIDE = sqrt(rand_numb_size)+1;
//	dim3 dimBlock_synapse( ThreadsPerBlock, ThreadsPerBlock );
//	dim3 dimGrid_synapse( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));
//	cudaMalloc((void **)&random_number_list_device,rand_numb_size*sizeof(float));
//	cudaMemcpy(random_number_list_device,random_number_list,rand_numb_size*sizeof(float),cudaMemcpyHostToDevice);
//
//    random<<<dimGrid_synapse,dimBlock_synapse>>>(random_number_list_device, rand_numb_size, states);
//    printf("2.11\n");
//    //Setting up input instance matrix:
//    float **d_input_instance;
//    float **d_convolution_result;
//    float **h_input_instance;
//    float **h_convolution_result;
//    float *probe = new float[1000];
//	int instance_array_size = CNN_total_layer_num;
//	cudaMalloc(&d_input_instance, instance_array_size*sizeof(float *));
//	int convolution_result_size = CNN_total_layer_num - 1;
//	cudaMalloc(&d_convolution_result, convolution_result_size*sizeof(float *));
//    h_input_instance = (float**)malloc(instance_array_size * sizeof(float*));
//    h_convolution_result = (float**)malloc(convolution_result_size * sizeof(float*));
//    CNN_util(network_config, d_input_instance, d_convolution_result, h_input_instance, h_convolution_result, 0);
//
////	float **add = &h_convolution_result[0];
////	printf("Address On GPU: %p\n", add);
//
//    //Setting up others
//    cudaMalloc((void **)&Neuron_list_device, SIZE*sizeof(Neuron));
//    //cudaMalloc((void **)&old_device_neurons, SIZE*sizeof(Neuron));
//    printf("2.2\n");
//    //cudaMalloc((void **)&states, SIZE * sizeof(curandState_t));
//    cudaMalloc((void **)&log_v, MAX_TIME * sizeof(float));
//    cudaMalloc((void **)&log_spike, total_depth_number * sizeof(float));
//    cudaMalloc((void **)&log_spike_default, total_depth_number * sizeof(float));
//    //cudaMalloc((void **)&log_total_spike, SIZE * sizeof(float));
//    gpuErrchk( cudaMalloc((void **)&log_total_spike, SIZE * sizeof(float)) );
//    cudaMalloc((void **)&spike_flag_device, instance_array_size*sizeof(int));
//    //rand_init<<<dimGrid,dimBlock>>>(time(0), states);
//
//    cudaMemcpy(Neuron_list_device,NeuronList,SIZE*sizeof(Neuron),cudaMemcpyHostToDevice);
//    //cudaMemcpy(old_device_neurons,NeuronList,SIZE*sizeof(Neuron),cudaMemcpyHostToDevice);
//    //cudaMemcpy(random_number_list_device, random_number_list, SIZE*sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(log_v,log_v_host,MAX_TIME*sizeof(float),cudaMemcpyHostToDevice);
//    cudaMemcpy(log_spike,log_spike_host,total_depth_number*sizeof(float),cudaMemcpyHostToDevice);
//    cudaMemcpy(log_spike_default,log_spike_host,total_depth_number*sizeof(float),cudaMemcpyHostToDevice);
//    gpuErrchk( cudaMemcpy(log_total_spike,log_total_spike_host,SIZE*sizeof(float),cudaMemcpyHostToDevice) );
//    cudaMemcpy(spike_flag_device,spike_flag,instance_array_size*sizeof(int),cudaMemcpyHostToDevice);
//    printf("3\n");
//    //cout<<"network size: "<<SIZE<<endl;
//    int network_size = SIZE;
//
//    int max_time = MAX_TIME;
//
//
//    //cudaMemcpy(Neuron_list_device,old_device_neurons,sizeof(Neuron)*SIZE,cudaMemcpyDeviceToDevice);
//    //first change raw img data into frequency
//    int mnist_start_index = 0;
//    int mnist_end_index = input_neuron_num;
//    //change pixel signal to frequency
//
//    MNIST_drive(NeuronList, mnist_img, network_size, training_set_number, mnist_start_index, mnist_end_index, max_frequency, min_frequency, 1);
//
//
//    cudaDeviceSynchronize();
//
//    //data_check(Neuron_list_device,log_total_spike,SIZE,1);
//    float *one_mnist_img = new float[input_neuron_num];
//
//    clock_t iter_start, iter_log;
//    iter_start = clock();
//    int log_interval = MAX_TIME/10;
//    //read_filter_GPU_one_layer<<<1, 1>>>(d_network_config, h_filter_array[0], 1);
//    //read_filter_GPU<<<1, 1>>>(d_network_config, d_filter_array);
//
//    int reiter_run = 1;
//
//    int time = 0;
//    int training_img_index = 0;
//    while (time<max_time){
//    	//cout<<endl<<" It: "<<time<<endl;
//        //random<<<dimGrid,dimBlock>>>(random_number_list_device, states);
//    	//first create an array of 1 MNIST image
////    	if(STOCHASTIC_STDP || STOCHASTIC_ROUNDING){
////            random<<<dimGrid_synapse,dimBlock_synapse>>>(random_number_list_device, rand_numb_size, states);
////    	}
//    	if(time%log_interval == 0){
//    		cudaMemcpy(NeuronList,Neuron_list_device,SIZE*sizeof(Neuron),cudaMemcpyDeviceToHost);
//    		string interval_file_name = "device2_output_at_iter_" + to_string(time) + ".txt";
//    		//write_neuron_list(NeuronList, interval_file_name, network_size);
//    		//printf("%");
//    	}
//
//    	if(time%tenpercent_iter == 0){
//    		iter_log = clock();
//    		cout<<to_string(10*(time/tenpercent_iter))<<"% done, time used is: " << (iter_log - iter_start)/1000 << " (ms)" << endl;
//    	}
//    	//fault below here:
//
//    	if(time%training_time_each_img==0){//at the beginning of each img's training, load into
//    		//cout<<"Image Load Iter: "<<time<<endl;
//    		for(int i=0;i<input_neuron_num;i++){
//    			one_mnist_img[i] = mnist_img[training_img_index*input_neuron_num+i];
//    		}
////    	    for (int y=0; y<28; ++y) {
////    	    	    for (int x=0; x<28; ++x) {
////    	    	      std::cout << ((one_mnist_img[y*28+x] <= 1.1)? ' ' : '*');
////    	    	      //std::cout << int(one_mnist_img[y*28+x]) << ' ';
////    	    	    }
////    	    	    std::cout << std::endl;
////    	    }
//    		MNIST_drive(Neuron_list_device, one_mnist_img, network_size, training_set_number, mnist_start_index, mnist_end_index, max_frequency, min_frequency, 0);
//    		//MNIST_drive(old_device_neurons, one_mnist_img, network_size,training_set_number, mnist_start_index, mnist_end_index, max_frequency, min_frequency, 0);
//    		training_img_index ++;
//    		//confirm the data in signal neuron
//    		//cudaMemcpy(NeuronList,Neuron_list_device,SIZE*sizeof(Neuron),cudaMemcpyDeviceToHost);
//    		//data_check(NeuronList,log_total_spike,SIZE, mnist_start_index, mnist_end_index, 3);danshe.2010@gamil.com
//    		//printf("\n\n\n************************\n\n\n\n");
//    	}
//    	//cout<<"One IMG loaded"<<endl;
//    	//enter spiking neuron simulation:
//
//
//    	for(int layer_iter=0;layer_iter<CNN_total_layer_num;layer_iter++){
//    		int convolution_result_index = layer_iter - 1;
//    		if (layer_iter==0) {//fault at convolution kernel and spiking cnn
//    			convolution_result_index = 0;
//    	    	//CNN_struct *settings; int layer_index; float **d_input_2d; float **filter_2d; float **output_2d;
//    	    	//convolution_kernel(settings, layer_index, d_input_2d, filter_2d, output_2d);
//    	    	//problem is in spiking_cnn_main
//    			spiking_cnn_main(Neuron_list_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, layer_iter, network_size, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, input_float, time);
//    			//spiking_cnn_main(Neuron_list_device, old_device_neurons, d_network_config, random_number_list_device, d_convolution_result[convolution_result_index], d_input_instance[layer_iter], layer_iter, network_size, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, time);
//    			//spiking_cnn_main(Neuron_list_device, old_device_neurons, d_network_config, random_number_list_device, d_convolution_result[0], d_input_instance[0], layer_iter, network_size, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, time);
//    			convolution_kernel(network_config, layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
//    		}else{
//    			//printf("In layer: %d\n", layer_iter);
//    			spiking_cnn_main(Neuron_list_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, layer_iter, network_size, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, input_float, time);
//    			//network_config->layer[layer_iter].depth;
//    			if (layer_iter!=(CNN_total_layer_num-1)) convolution_kernel(network_config, layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
//				synapse_drive_cnn_v2(Neuron_list_device, network_config, d_network_config, d_filter_array, layer_iter, network_size, syn_timer_max, connection_size, random_number_list_device, states, -1.0, -1.0);//STDP
//    		}
//
//    	}
//    	//=================TRY WITH LAYER wise inhibition=====================
//    	cudaMemcpy(spike_flag, spike_flag_device, CNN_total_layer_num*sizeof(int),cudaMemcpyDeviceToHost);
//    	for(int layer_iter=0;layer_iter<CNN_total_layer_num;layer_iter++){
//    		//if (layer_iter==0) printf("Layer[%d] SpikeFlag: %d\n", layer_iter, spike_flag[layer_iter]);
//            if(spike_flag[layer_iter]>0){//use lateral inhibition
//            	spiking_learning_drive(Neuron_list_device, network_size, inhibition_time, log_total_spike, target_frequency, time, log_spike, layer_iter, network_config, 4);
//            }
//    		spike_flag[layer_iter] = 0;
//    	}
//    	cudaMemcpy(spike_flag_device,spike_flag,CNN_total_layer_num*sizeof(int),cudaMemcpyHostToDevice);
//	//=================TRY WITH NO LAYERAL INHIBITION, MAY BE WRONG=====================
//    	//printf("network_size: %d", network_size);
//    	//spiking_learning_drive(Neuron_list_device, network_size, inhibition_time, log_total_spike, target_frequency, time, log_spike, 3); //lateral inhibition
//	//==================================================================================
//		if(HOMEOSTASIS_ENABLE){
//			if(time%HOMEOSTASIS_UPDATE_FREQUENCY == 0 && time != 0){
//				//spiking_learning_drive(Neuron_list_device, network_size, inhibition_time, log_total_spike, target_frequency, time, log_spike, 0, 1);
//			}
//		}
//        cudaDeviceSynchronize();
//	cudaMemcpy(log_spike,log_spike_default,total_depth_number*sizeof(float),cudaMemcpyDeviceToDevice);	//set the log_spike to default value
//
//        //if any neuron spikes, run inhibition
//	//cudaMemcpy(spike_flag, spike_flag_device, CNN_total_layer_num*sizeof(int),cudaMemcpyDeviceToHost);
//	//printf("AtTime:%d_spike_flag_is:%d\n",time,spike_flag[0]);
//        //if(spike_flag[0]>0){//use lateral inhibition
//        	//spiking_learning_drive(Neuron_list_device, network_size, inhibition_time, log_total_spike, target_frequency, time, log_spike, 0);
//        	//spike_flag[0] = 0;
//    	    	//cudaMemcpy(spike_flag_device,spike_flag,sizeof(int),cudaMemcpyHostToDevice);
//        //}
//
//        //cudaMemcpy(old_device_neurons,Neuron_list_device,sizeof(Neuron)*SIZE,cudaMemcpyDeviceToDevice);
//        //cudaDeviceSynchronize();
//    	time ++;
//    }
//    //spiking_learning_drive(Neuron_list_device, network_size, inhibition_time, 2);
//	//cudaDeviceSynchronize();
//
//	filter_util(network_config, Neuron_list_device, network_size, h_filter_array, d_filter_array, 2);
//    cudaMemcpy(NeuronList,Neuron_list_device,SIZE*sizeof(Neuron),cudaMemcpyDeviceToHost);
//    cudaMemcpy(log_v_host,log_v,MAX_TIME*sizeof(float),cudaMemcpyDeviceToHost);
//    cudaMemcpy(log_spike_host,log_spike,total_depth_number*sizeof(float),cudaMemcpyDeviceToHost);
//    gpuErrchk( cudaMemcpy(log_total_spike_host,log_total_spike,SIZE*sizeof(float),cudaMemcpyDeviceToHost) );
//
//
//    //print out the synapse conductance data
//    //data_check(NeuronList,log_total_spike,SIZE, mnist_start_index, mnist_end_index, 2);
//
//    ofstream myfile ((index_prefix+"device2_spike_of_neuron_out.csv"));
//    if (myfile.is_open()){
//    	//myfile << "This is a new test\n";
//    	for(int i=0; i < SIZE; i++){
//    		//printf("_%f_", log_v_host[i]);
//    		myfile << log_total_spike_host[i] << ", ";
//    	}
//    	myfile.close();
//    }
//
//    ofstream myfile_p ((index_prefix+"probe.csv"));
//    if (myfile_p.is_open()){
//    	//myfile << "This is a new test\n";
//    	for(int i=0; i < 1000; i++){
//    		//printf("_%f_", log_v_host[i]);
//    		myfile_p << probe[i] << ", ";
//    	}
//    	myfile_p.close();
//    }
//
////
////    ofstream myfile_0 ((index_prefix+"device2_out_v.csv"));
////    if (myfile_0.is_open()){
////    	//myfile << "This is a new test\n";
////    	for(int i=0; i < MAX_TIME; i++){
////    		//printf("_%f_", log_v_host[i]);
////    		myfile_0 << log_v_host[i] << ", ";
////    	}
////    	myfile.close();
////    }
////
////    ofstream myfile_2 ((index_prefix+"device2_spike_of_one.csv"));
////    if (myfile_2.is_open()){
////    	//myfile << "This is a new test\n";
////    	for(int i=0; i < MAX_TIME; i++){
////    		//printf("_%f_", log_v_host[i]);
////    		myfile_2 << log_spike_host[i] << ", ";
////    	}
////    	myfile_2.close();
////    }
//
//    data_check(NeuronList,log_total_spike,SIZE, mnist_start_index, mnist_end_index, 2, "");
//
//    cudaMemcpy(h_filter_array, d_filter_array, filter_array_size* sizeof(float*), cudaMemcpyDeviceToHost);
//	filter_util(network_config, NeuronList, network_size, h_filter_array, d_filter_array, 1);	//write filter to file
//    write_neuron_list(NeuronList, (index_prefix+"device2_output_network.txt"), network_size);
//
//    //===clean up===
//    //delete[] random_number_list;
//    delete[] log_v_host;
//	delete[] NeuronList;
//	delete[] log_spike_host;
//	delete[] log_total_spike_host;
//	delete[] mnist_img;
//	delete[] NeuronList_temp;
//	delete[] one_mnist_img;
//	delete[] probe;
//	delete[] random_number_list;
//	delete[] mnist_label;
//	delete[] spike_flag;
//	delete[] num_one_digit_img;
//	//cudaFree(states);
//	cudaFree(log_v);
//	cudaFree(log_spike);
//	cudaFree(log_total_spike);
//	cudaFree(Neuron_list_device);
//	//cudaFree(old_device_neurons);
//	cudaFree(random_number_list_device);
//	cudaFree(d_network_config);
//	cudaFree(states);
//	cudaFree(spike_flag_device);
//	cudaFree(log_spike_default);
//
//}
//
//
//void space_transfer(string index_prefix, float input_float, float input_float_2, int input_int, int input_int_2, string input_img){
//	/*
//	int training_set_number = 1;
//	int size_per_img = input_image_w * input_image_l*input_image_channel;
//	float *mnist_img = new float[size_per_img*training_set_number];
//	string image_file = "train-images-idx3-ubyte"; //"train-images-idx3-ubyte";
//	MNIST_read_image(image_file, mnist_img, training_set_number);
//	int *mnist_label = new int[training_set_number];
//	string image_label_file = "train-labels-idx1-ubyte";
//	MNIST_read_label(image_label_file, mnist_label, training_set_number);
//
//	float *filter;
//	float *output = new float[size_per_img*training_set_number];
//
//	convolution_kernel(mnist_img, filter, output);
//	img_util(output, "test_output.jpg", 0);
//	*/
//	int resume_learning = 0;
//	CNN_struct *network_config = new CNN_struct;
//	network_config_generator(3, network_config);
//	Neuron *NeuronList_temp = new Neuron[1];
//	CNN_struct *d_network_config;
//	cudaMalloc((void **)&d_network_config,sizeof(CNN_struct));
//	cudaMemcpy(d_network_config,network_config,sizeof(CNN_struct),cudaMemcpyHostToDevice);
//	int total_depth_number = 0;
//	for(int i=0;i<CNN_total_layer_num; i++){
//		total_depth_number = total_depth_number + network_config->layer[i].depth;
//		cout<<"depth number: "<<network_config->layer[i].depth<<endl;
//	}
//
//	cout<<endl<<"Total depth number: "<<total_depth_number<<endl;
//	float **h_filter_array;
//	float **d_filter_array;
//	int filter_array_size = CNN_total_layer_num-1;
//	cudaMalloc(&d_filter_array, filter_array_size*sizeof(float *));
//	h_filter_array = (float**)malloc(filter_array_size * sizeof(float*));
//	filter_util(network_config, NeuronList_temp, 0, h_filter_array, d_filter_array, 0);
//
//	/*
//	img_util(mnist_img, "tensorflow_small.png", 1);
//	img_util(mnist_img, "test_output_-1.png", 0);
//
//	float *output = new float[size_per_img*training_set_number];
//
//	int image_bytes = input_image_channel * input_image_l * input_image_w * sizeof(float);
//
//	float* convolution_device_input{nullptr};
//	cudaMalloc(&convolution_device_input, image_bytes);
//	cudaMemcpy(convolution_device_input, mnist_img, image_bytes, cudaMemcpyHostToDevice);
//
//	int filter_in_channel = input_image_channel;
//	int filter_out_channel = input_image_channel;
//	int filter_height = 3;
//	int filter_width = 3;
//	const float kernel_template[3][3] = {
//	{1, 1, 1},
//	{1, -8, 1},
//	{1, 1, 1}
//	};
//	float h_kernel[filter_in_channel][filter_out_channel][filter_height][filter_width];
//	for (int kernel = 0; kernel < filter_in_channel; ++kernel) {
//		for (int channel = 0; channel < filter_out_channel; ++channel) {
//		  for (int row = 0; row < filter_height; ++row) {
//			for (int column = 0; column < filter_width; ++column) {
//			  h_kernel[kernel][channel][row][column] = kernel_template[row][column];
//			}
//		  }
//		}
//	}
//	float* filter{nullptr};
//	cudaMalloc(&filter, sizeof(h_kernel));
//	cudaMemcpy(filter, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);
//
//
//	convolution_kernel(convolution_device_input, filter, output);
//	img_util(output, "test_output_1.png", 0);
//
//	cudaFree(filter);
//	cudaFree(convolution_device_input);
//
//	*/
//
////===========END of CNN special setting-up phase============
//
//
//	//set parameters
//
//	int training_time_each_img = input_int;
//	int calculated_total_time = training_time_each_img*1;
//	#undef MAX_TIME
//	#define MAX_TIME calculated_total_time
//	printf("==Training Total Iter: %d==", MAX_TIME);
//	int total_neuron_num = 0;
//	int total_spiking_num = 0;
//	for(int i=0;i<CNN_total_layer_num;i++){
//		total_neuron_num += network_config->layer[i].neuron_num;
//		if(i!=0)
//		total_spiking_num += network_config->layer[i].neuron_num;
//	}
//	total_neuron_num += 100;
//	//total_neuron_num = 20000;
//	cout<<endl<<"total neuron num: "<<total_neuron_num<<endl;
//	cout<<"total spiking neuron num: "<<total_spiking_num<<endl;
//	#undef SIZE
//	#define SIZE total_neuron_num
//	#undef SPIKING_NEURON_NUM
//	#define SPIKING_NEURON_NUM total_spiking_num
//
//
//	float max_frequency = 22; //in Hz default 22
//	float min_frequency = 1;
//	int training_set_number = 50000;
//	int input_neuron_num = input_image_w*input_image_l*input_image_channel;
//	int input_image_signal_channel_size  = input_image_w*input_image_l;
//	int spiking_neuron_num = SPIKING_NEURON_NUM;
//	int output_layer_neuron_num = OUTPUT_LAYER_NEURON_NUM;
//	int tenpercent_iter = MAX_TIME/10;
//	int connection_size = MAX_CONNECTION;
//	int syn_timer_max = 25;
//	int input_signal_width = 15;	//default 25
//	int inhibition_time = 3;	//default 10
//
//	float target_frequency_param = 0.5;
//	float target_frequency = 100;
//	float *mnist_img = new float[input_neuron_num*training_set_number];
//	for(int i=0;i<input_neuron_num*training_set_number;i++) mnist_img[i] = 0;
//	string image_file = "train_dataset_noisy_cifar";//"fashion-train-images-idx3-ubyte";//"train_dataset_noisy";//"train_dataset_noisy"; //"train-images-idx3-ubyte";
//	if(input_image_channel==1){
//		CIFAR_read_image(mnist_img, input_neuron_num, 0, 1);
//	}else{
//		CIFAR_read_image(mnist_img, input_neuron_num, 0, 0);
//	}
//	//CIFAR_read_image_one_channel(mnist_img, input_image_signal_channel_size, input_int_2, 0);
//	//MNIST_read_image(image_file, mnist_img, training_set_number);
//	int *mnist_label = new int[training_set_number];
//	string image_label_file = "train-labels-idx1-ubyte";
//	CIFAR_read_label(mnist_label, 0);
//	//MNIST_read_label(image_label_file, mnist_label, training_set_number);
//	//special_function: learn one category
//	int learn_one_digit = 0;
//	int *num_one_digit_img = new int[1];
//	if(learn_one_digit){
//		MNIST_labeling("abc", 60000, mnist_img, mnist_label, mnist_img, num_one_digit_img, spiking_neuron_num, 1, 5);
//		printf("Learning only one digit, number of img: %d\n", num_one_digit_img);
//	}
//
//	//int synapse_size = SIZE*SIZE;
//	//cout<<SIZE<<endl;
//    Neuron *NeuronList = new Neuron[SIZE];
//	//unsigned char *synapse_timer = new unsigned char[synapse_size];  //this is the array that stores timer used in STPD. e.g Neuron x --->  Neuron y Spike! In the array index [(x-1)*SIZE+(y-1)]  => 1
//	//curandState_t *states;
//	//float *random_number_list = new float[SIZE];
//	float *log_v_host = new float[MAX_TIME];
//	float *log_spike_host = new float[total_depth_number];
//
//	float *log_total_spike_host = new float[SIZE];
//	for(int i=0; i < SIZE; i++){
//		log_total_spike_host[i] = 0;
//	}
//	int *spike_flag = new int[CNN_total_layer_num];
//	for(int i=0; i < CNN_total_layer_num; i++){
//		spike_flag[i] = 0;
//	}
//	for(int i=0; i<total_depth_number; i++) log_spike_host[i] = 0;
//	//init_log_v(log_v_host);
//	//init_data_log(log_v_host,log_spike_host,log_total_spike_host, MAX_TIME);
//	neuron_list_init(NeuronList, total_neuron_num);
//	//printf("=0=\n");
//	//
//	if(resume_learning){
//		printf("RESUME LEARNING\n");
//		read_neuron_list(NeuronList, 1, "device2_output_network.txt");
//	}else{
//		read_neuron_list(NeuronList, 1, "spike_cnn.txt");
//	}
//    //write_neuron_list(NeuronList, "learning_output_confirm.txt", SIZE);
//	//check_neuron(NeuronList, 800, 820);
//
//	Neuron *Neuron_list_device;
//	//Neuron *old_device_neurons;
//	//unsigned char *snapse_timer_device;
//	float *log_v;
//	float *log_spike;
//	float *log_spike_default;
//	float *log_total_spike;
//	int *spike_flag_device;
//
//
////    printf("2\n");
//	//printf("=1=\n");
//	//random number function:
//    float rand_list_size_to_total_connection_ratio = 1;
//	int rand_numb_size = SPIKING_NEURON_NUM*MAX_CONNECTION;
//	curandState_t *states;
//	cudaMalloc((void **)&states, rand_numb_size * sizeof(curandState_t));
//	int SIZE_PER_SIDE = sqrt(rand_numb_size)+1;
//    dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );
//    dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));
//	dim3 print_grid(1);
//	dim3 print_block(1);
////    printf("2.1\n");
//	rand_init<<<dimGrid,dimBlock>>>(time(0), rand_numb_size, states);
//
//	float *random_number_list = new float[rand_numb_size];
//	float *random_number_list_device;
//	SIZE_PER_SIDE = sqrt(rand_numb_size)+1;
//	dim3 dimBlock_synapse( ThreadsPerBlock, ThreadsPerBlock );
//	dim3 dimGrid_synapse( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));
//	cudaMalloc((void **)&random_number_list_device,rand_numb_size*sizeof(float));
//	cudaMemcpy(random_number_list_device,random_number_list,rand_numb_size*sizeof(float),cudaMemcpyHostToDevice);
//
//    random<<<dimGrid_synapse,dimBlock_synapse>>>(random_number_list_device, rand_numb_size, states);
////    printf("2.11\n");
//    //Setting up input instance matrix:
//    float **d_input_instance;
//    float **d_convolution_result;
//    float **h_input_instance;
//    float **h_convolution_result;
//    float *probe = new float[1000];
//	int instance_array_size = CNN_total_layer_num;
//	cudaMalloc(&d_input_instance, instance_array_size*sizeof(float *));
//	int convolution_result_size = CNN_total_layer_num - 1;
//	cudaMalloc(&d_convolution_result, convolution_result_size*sizeof(float *));
//    h_input_instance = (float**)malloc(instance_array_size * sizeof(float*));
//    h_convolution_result = (float**)malloc(convolution_result_size * sizeof(float*));
//    CNN_util(network_config, d_input_instance, d_convolution_result, h_input_instance, h_convolution_result, 0);
//
////	float **add = &h_convolution_result[0];
////	printf("Address On GPU: %p\n", add);
//
//    //Setting up others
//    cudaMalloc((void **)&Neuron_list_device, SIZE*sizeof(Neuron));
//    //cudaMalloc((void **)&old_device_neurons, SIZE*sizeof(Neuron));
////    printf("2.2\n");
//    //cudaMalloc((void **)&states, SIZE * sizeof(curandState_t));
//    cudaMalloc((void **)&log_v, MAX_TIME * sizeof(float));
//    cudaMalloc((void **)&log_spike, total_depth_number * sizeof(float));
//    cudaMalloc((void **)&log_spike_default, total_depth_number * sizeof(float));
//    //cudaMalloc((void **)&log_total_spike, SIZE * sizeof(float));
//    gpuErrchk( cudaMalloc((void **)&log_total_spike, SIZE * sizeof(float)) );
//    cudaMalloc((void **)&spike_flag_device, instance_array_size*sizeof(int));
//    //rand_init<<<dimGrid,dimBlock>>>(time(0), states);
//
//    cudaMemcpy(Neuron_list_device,NeuronList,SIZE*sizeof(Neuron),cudaMemcpyHostToDevice);
//    //cudaMemcpy(old_device_neurons,NeuronList,SIZE*sizeof(Neuron),cudaMemcpyHostToDevice);
//    //cudaMemcpy(random_number_list_device, random_number_list, SIZE*sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(log_v,log_v_host,MAX_TIME*sizeof(float),cudaMemcpyHostToDevice);
//    cudaMemcpy(log_spike,log_spike_host,total_depth_number*sizeof(float),cudaMemcpyHostToDevice);
//    cudaMemcpy(log_spike_default,log_spike_host,total_depth_number*sizeof(float),cudaMemcpyHostToDevice);
//    gpuErrchk( cudaMemcpy(log_total_spike,log_total_spike_host,SIZE*sizeof(float),cudaMemcpyHostToDevice) );
//    cudaMemcpy(spike_flag_device,spike_flag,instance_array_size*sizeof(int),cudaMemcpyHostToDevice);
//    printf("3\n");
//    //cout<<"network size: "<<SIZE<<endl;
//    int network_size = SIZE;
//
//    int max_time = MAX_TIME;
//
//
//    //cudaMemcpy(Neuron_list_device,old_device_neurons,sizeof(Neuron)*SIZE,cudaMemcpyDeviceToDevice);
//    //first change raw img data into frequency
//    int mnist_start_index = 0;
//    int mnist_end_index = input_neuron_num;
//    //change pixel signal to frequency
//
//    MNIST_drive(NeuronList, mnist_img, network_size, training_set_number, mnist_start_index, mnist_end_index, max_frequency, min_frequency, 1);
//
//
//    cudaDeviceSynchronize();
//
//    //data_check(Neuron_list_device,log_total_spike,SIZE,1);
//    float *one_mnist_img = new float[input_neuron_num];
//
//    clock_t iter_start, iter_log;
//    iter_start = clock();
//    int log_interval = MAX_TIME/10;
//    //read_filter_GPU_one_layer<<<1, 1>>>(d_network_config, h_filter_array[0], 1);
//    //read_filter_GPU<<<1, 1>>>(d_network_config, d_filter_array);
//
//    int reiter_run = 1;
//
//    int time = 0;
//    int training_img_index = 0;
//    while (time<max_time){
//    	//cout<<endl<<" It: "<<time<<endl;
//        //random<<<dimGrid,dimBlock>>>(random_number_list_device, states);
//    	//first create an array of 1 MNIST image
////    	if(STOCHASTIC_STDP || STOCHASTIC_ROUNDING){
////            random<<<dimGrid_synapse,dimBlock_synapse>>>(random_number_list_device, rand_numb_size, states);
////    	}
//    	if(time%log_interval == 0){
//    		cudaMemcpy(NeuronList,Neuron_list_device,SIZE*sizeof(Neuron),cudaMemcpyDeviceToHost);
//    		string interval_file_name = "device2_output_at_iter_" + to_string(time) + ".txt";
//    		write_neuron_list(NeuronList, interval_file_name, network_size);
//
//    		//printf("%");
//    	}
//
//    	if(time%tenpercent_iter == 0){
//    		iter_log = clock();
//    		cout<<to_string(10*(time/tenpercent_iter))<<"% done, time used is: " << (iter_log - iter_start)/1000 << " (ms)" << endl;
//    	}
//    	//fault below here:
//
//    	if(time%training_time_each_img==0){//at the beginning of each img's training, load into
//    		//cout<<"Image Load Iter: "<<time<<endl;
//    		for(int i=0;i<input_neuron_num;i++){
//    			one_mnist_img[i] = mnist_img[training_img_index*input_neuron_num+i];
//    		}
////    	    for (int y=0; y<28; ++y) {
////    	    	    for (int x=0; x<28; ++x) {
////    	    	      std::cout << ((one_mnist_img[y*28+x] <= 1.1)? ' ' : '*');
////    	    	      //std::cout << int(one_mnist_img[y*28+x]) << ' ';
////    	    	    }
////    	    	    std::cout << std::endl;
////    	    }
//    		MNIST_drive(Neuron_list_device, one_mnist_img, network_size, training_set_number, mnist_start_index, mnist_end_index, max_frequency, min_frequency, 0);
//    		//MNIST_drive(old_device_neurons, one_mnist_img, network_size,training_set_number, mnist_start_index, mnist_end_index, max_frequency, min_frequency, 0);
//    		training_img_index ++;
//    		if(training_img_index>=49999) training_img_index = 0;
//    		//confirm the data in signal neuron
//    		//cudaMemcpy(NeuronList,Neuron_list_device,SIZE*sizeof(Neuron),cudaMemcpyDeviceToHost);
//    		//data_check(NeuronList,log_total_spike,SIZE, mnist_start_index, mnist_end_index, 3);
//    		//printf("\n\n\n************************\n\n\n\n");
//    	}
//    	//cout<<"One IMG loaded"<<endl;
//    	//enter spiking neuron simulation:
//
//
//    	for(int layer_iter=0;layer_iter<CNN_total_layer_num;layer_iter++){
//    		int convolution_result_index = layer_iter - 1;
//    		if (layer_iter==0) {//fault at convolution kernel and spiking cnn
//    			convolution_result_index = 0;
//    	    	//CNN_struct *settings; int layer_index; float **d_input_2d; float **filter_2d; float **output_2d;
//    	    	//convolution_kernel(settings, layer_index, d_input_2d, filter_2d, output_2d);
//    	    	//problem is in spiking_cnn_main
//    			spiking_cnn_main(Neuron_list_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, layer_iter, network_size, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, input_float, time);
//    			//spiking_cnn_main(Neuron_list_device, old_device_neurons, d_network_config, random_number_list_device, d_convolution_result[convolution_result_index], d_input_instance[layer_iter], layer_iter, network_size, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, time);
//    			//spiking_cnn_main(Neuron_list_device, old_device_neurons, d_network_config, random_number_list_device, d_convolution_result[0], d_input_instance[0], layer_iter, network_size, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, time);
//    			convolution_kernel(network_config, layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
//    		}else{
//    			//printf("In layer: %d\n", layer_iter);
//    			spiking_cnn_main(Neuron_list_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, layer_iter, network_size, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, input_float, time);
//    			//network_config->layer[layer_iter].depth;
//    			if (layer_iter!=(CNN_total_layer_num-1)) convolution_kernel(network_config, layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
//				synapse_drive_cnn_v2(Neuron_list_device, network_config, d_network_config, d_filter_array, layer_iter, network_size, syn_timer_max, connection_size, random_number_list_device, states, -1.0, -1.0);//STDP
//    		}
//
//    	}
//    	//=================TRY WITH LAYER wise inhibition=====================
//
//    	cudaMemcpy(spike_flag, spike_flag_device, CNN_total_layer_num*sizeof(int),cudaMemcpyDeviceToHost);
//    	for(int layer_iter=0;layer_iter<CNN_total_layer_num;layer_iter++){
//    		//if (layer_iter==0) printf("Layer[%d] SpikeFlag: %d\n", layer_iter, spike_flag[layer_iter]);
//            if(spike_flag[layer_iter]>0){//use lateral inhibition
//            	//spiking_learning_drive(Neuron_list_device, network_size, inhibition_time, log_total_spike, target_frequency, time, log_spike, layer_iter, network_config, 4);
//            }
//    		spike_flag[layer_iter] = 0;
//    	}
//    	cudaMemcpy(spike_flag_device,spike_flag,CNN_total_layer_num*sizeof(int),cudaMemcpyHostToDevice);
//    	cudaMemcpy(log_spike,log_spike_default,total_depth_number*sizeof(float),cudaMemcpyDeviceToDevice);	//set the log_spike to default value
//
//	//=================TRY WITH NO LAYERAL INHIBITION, MAY BE WRONG=====================
//    	//printf("network_size: %d", network_size);
//    	//spiking_learning_drive(Neuron_list_device, network_size, inhibition_time, log_total_spike, target_frequency, time, log_spike, 3); //lateral inhibition
//	//==================================================================================
//		if(HOMEOSTASIS_ENABLE){
//			if(time%HOMEOSTASIS_UPDATE_FREQUENCY == 0 && time != 0){
//				//spiking_learning_drive(Neuron_list_device, network_size, inhibition_time, log_total_spike, target_frequency, time, log_spike, 0, 1);
//			}
//		}
//        cudaDeviceSynchronize();
//
//
//        //if any neuron spikes, run inhibition
//	//cudaMemcpy(spike_flag, spike_flag_device, CNN_total_layer_num*sizeof(int),cudaMemcpyDeviceToHost);
//	//printf("AtTime:%d_spike_flag_is:%d\n",time,spike_flag[0]);
//        //if(spike_flag[0]>0){//use lateral inhibition
//        	//spiking_learning_drive(Neuron_list_device, network_size, inhibition_time, log_total_spike, target_frequency, time, log_spike, 0);
//        	//spike_flag[0] = 0;
//    	    	//cudaMemcpy(spike_flag_device,spike_flag,sizeof(int),cudaMemcpyHostToDevice);
//        //}
//
//        //cudaMemcpy(old_device_neurons,Neuron_list_device,sizeof(Neuron)*SIZE,cudaMemcpyDeviceToDevice);
//        //cudaDeviceSynchronize();
//    	time ++;
//    }
//    //spiking_learning_drive(Neuron_list_device, network_size, inhibition_time, 2);
//	//cudaDeviceSynchronize();
//
//	filter_util(network_config, Neuron_list_device, network_size, h_filter_array, d_filter_array, 2);
//    cudaMemcpy(NeuronList,Neuron_list_device,SIZE*sizeof(Neuron),cudaMemcpyDeviceToHost);
//    cudaMemcpy(log_v_host,log_v,MAX_TIME*sizeof(float),cudaMemcpyDeviceToHost);
//    cudaMemcpy(log_spike_host,log_spike,total_depth_number*sizeof(float),cudaMemcpyDeviceToHost);
//    gpuErrchk( cudaMemcpy(log_total_spike_host,log_total_spike,SIZE*sizeof(float),cudaMemcpyDeviceToHost) );
//
//
//    //print out the synapse conductance data
//    //data_check(NeuronList,log_total_spike,SIZE, mnist_start_index, mnist_end_index, 2);
//
//    ofstream myfile ((index_prefix+"device2_spike_of_neuron_out.csv"));
//    if (myfile.is_open()){
//    	//myfile << "This is a new test\n";
//    	for(int i=0; i < SIZE; i++){
//    		//printf("_%f_", log_v_host[i]);
//    		myfile << log_total_spike_host[i] << ", ";
//    	}
//    	myfile.close();
//    }
//
//    ofstream myfile_p ((index_prefix+"probe.csv"));
//    if (myfile_p.is_open()){
//    	//myfile << "This is a new test\n";
//    	for(int i=0; i < 1000; i++){
//    		//printf("_%f_", log_v_host[i]);
//    		myfile_p << probe[i] << ", ";
//    	}
//    	myfile_p.close();
//    }
//
////
////    ofstream myfile_0 ((index_prefix+"device2_out_v.csv"));
////    if (myfile_0.is_open()){
////    	//myfile << "This is a new test\n";
////    	for(int i=0; i < MAX_TIME; i++){
////    		//printf("_%f_", log_v_host[i]);
////    		myfile_0 << log_v_host[i] << ", ";
////    	}
////    	myfile.close();
////    }
////
////    ofstream myfile_2 ((index_prefix+"device2_spike_of_one.csv"));
////    if (myfile_2.is_open()){
////    	//myfile << "This is a new test\n";
////    	for(int i=0; i < MAX_TIME; i++){
////    		//printf("_%f_", log_v_host[i]);
////    		myfile_2 << log_spike_host[i] << ", ";
////    	}
////    	myfile_2.close();
////    }
//
//
//
//    cudaMemcpy(h_filter_array, d_filter_array, filter_array_size* sizeof(float*), cudaMemcpyDeviceToHost);
//	filter_util(network_config, NeuronList, network_size, h_filter_array, d_filter_array, 1);	//write filter to file
//    write_neuron_list(NeuronList, (index_prefix+"device2_output_network.txt"), network_size);
//    data_check(NeuronList,log_total_spike,SIZE, mnist_start_index, mnist_end_index, 2, "");
//    //===clean up===
//    //delete[] random_number_list;
//    delete[] log_v_host;
//	delete[] NeuronList;
//	delete[] log_spike_host;
//	delete[] log_total_spike_host;
//	delete[] mnist_img;
//	delete[] NeuronList_temp;
//	delete[] one_mnist_img;
//	delete[] probe;
//	delete[] random_number_list;
//	delete[] mnist_label;
//	delete[] spike_flag;
//	delete[] num_one_digit_img;
//	//cudaFree(states);
//	cudaFree(log_v);
//	cudaFree(log_spike);
//	cudaFree(log_total_spike);
//	cudaFree(Neuron_list_device);
//	//cudaFree(old_device_neurons);
//	cudaFree(random_number_list_device);
//	cudaFree(d_network_config);
//	cudaFree(states);
//	cudaFree(spike_flag_device);
//	cudaFree(log_spike_default);
//
//}



void run_cnn(string index_prefix, float input_float, float input_float_2, int input_int, int input_int_2, string input_img){
	/*
	int training_set_number = 1;
	int size_per_img = input_image_w * input_image_l*input_image_channel;
	float *mnist_img = new float[size_per_img*training_set_number];
	string image_file = "train-images-idx3-ubyte"; //"train-images-idx3-ubyte";
	MNIST_read_image(image_file, mnist_img, training_set_number);
	int *mnist_label = new int[training_set_number];
	string image_label_file = "train-labels-idx1-ubyte";
	MNIST_read_label(image_label_file, mnist_label, training_set_number);

	float *filter;
	float *output = new float[size_per_img*training_set_number];

	convolution_kernel(mnist_img, filter, output);
	img_util(output, "test_output.jpg", 0);
	*/
	int resume_learning = 0;
	CNN_struct *network_config = new CNN_struct;
	network_config_generator(3, network_config);
	Neuron *NeuronList_temp = new Neuron[1];
	CNN_struct *d_network_config;
	cudaMalloc((void **)&d_network_config,sizeof(CNN_struct));
	cudaMemcpy(d_network_config,network_config,sizeof(CNN_struct),cudaMemcpyHostToDevice);
	int total_depth_number = 0;
	for(int i=0;i<CNN_total_layer_num; i++){
		total_depth_number = total_depth_number + network_config->layer[i].depth;
		cout<<"depth number: "<<network_config->layer[i].depth<<endl;
	}

	cout<<endl<<"Total depth number: "<<total_depth_number<<endl;
	float **h_filter_array;
	float **d_filter_array;
	int filter_array_size = CNN_total_layer_num-1;
	cudaMalloc(&d_filter_array, filter_array_size*sizeof(float *));
	h_filter_array = (float**)malloc(filter_array_size * sizeof(float*));
	filter_util(network_config, NeuronList_temp, 0,0,  h_filter_array, d_filter_array, 0);

	/*
	img_util(mnist_img, "tensorflow_small.png", 1);
	img_util(mnist_img, "test_output_-1.png", 0);

	float *output = new float[size_per_img*training_set_number];

	int image_bytes = input_image_channel * input_image_l * input_image_w * sizeof(float);

	float* convolution_device_input{nullptr};
	cudaMalloc(&convolution_device_input, image_bytes);
	cudaMemcpy(convolution_device_input, mnist_img, image_bytes, cudaMemcpyHostToDevice);

	int filter_in_channel = input_image_channel;
	int filter_out_channel = input_image_channel;
	int filter_height = 3;
	int filter_width = 3;
	const float kernel_template[3][3] = {
	{1, 1, 1},
	{1, -8, 1},
	{1, 1, 1}
	};
	float h_kernel[filter_in_channel][filter_out_channel][filter_height][filter_width];
	for (int kernel = 0; kernel < filter_in_channel; ++kernel) {
		for (int channel = 0; channel < filter_out_channel; ++channel) {
		  for (int row = 0; row < filter_height; ++row) {
			for (int column = 0; column < filter_width; ++column) {
			  h_kernel[kernel][channel][row][column] = kernel_template[row][column];
			}
		  }
		}
	}
	float* filter{nullptr};
	cudaMalloc(&filter, sizeof(h_kernel));
	cudaMemcpy(filter, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);


	convolution_kernel(convolution_device_input, filter, output);
	img_util(output, "test_output_1.png", 0);

	cudaFree(filter);
	cudaFree(convolution_device_input);

	*/

//===========END of CNN special setting-up phase============

	Convolution_setting_struct *convolution_settings = new Convolution_setting_struct[CNN_total_layer_num];

	//set parameters

	int training_time_each_img = input_int;
	int calculated_total_time = training_time_each_img*80000;
	#undef MAX_TIME
	#define MAX_TIME calculated_total_time
	printf("==Training Total Iter: %d==", MAX_TIME);
	int total_neuron_num = 0;
	int total_spiking_num = 0;
	for(int i=0;i<CNN_total_layer_num;i++){
		total_neuron_num += network_config->layer[i].neuron_num;
		if(i!=0)
		total_spiking_num += network_config->layer[i].neuron_num;
	}
	total_spiking_num += 10;
	total_neuron_num += 10;
	//total_neuron_num = 20000;
	cout<<endl<<"total neuron num: "<<total_neuron_num<<endl;
	cout<<"total spiking neuron num: "<<total_spiking_num<<endl;
	#undef SIZE
	#define SIZE total_neuron_num
	#undef SPIKING_NEURON_NUM
	#define SPIKING_NEURON_NUM total_spiking_num


	float max_frequency = 22; //in Hz default 22
	float min_frequency = 1;
	int training_set_number = 5000;
	int input_neuron_num = input_image_w*input_image_l*input_image_channel;
	int input_image_signal_channel_size  = input_image_w*input_image_l;
	int spiking_neuron_num = SPIKING_NEURON_NUM;
	int output_layer_neuron_num = OUTPUT_LAYER_NEURON_NUM;
	int tenpercent_iter = MAX_TIME/10;
	int connection_size = MAX_CONNECTION;
	int syn_timer_max = 25;
	int input_signal_width = 15;	//default 25
	int inhibition_time = 10;	//default 10

	float target_frequency_param = 0.5;
	float target_frequency = 100;
	float *mnist_img = new float[input_neuron_num*training_set_number];
	for(int i=0;i<input_neuron_num*training_set_number;i++) mnist_img[i] = 0;
	string image_file = "train-images-idx3-ubyte";//"train_dataset_noisy_cifar";//"fashion-train-images-idx3-ubyte";//"train_dataset_noisy";//"train_dataset_noisy"; //"train-images-idx3-ubyte";
	cout<<endl<<"Image loading"<<endl;
	if(input_image_channel==1){
		//CIFAR_read_image(mnist_img, input_neuron_num, 0, 1);
		//GTVIR_read_image(mnist_img, input_neuron_num, training_set_number);
		MNIST_read_image(image_file, mnist_img, training_set_number);

	}else{
		//CIFAR_read_image(mnist_img, input_neuron_num, 0, 0);
		KAIST_PED_read_image("", mnist_img , training_set_number);

	}
	cout<<endl<<"Image loading done"<<endl;
	//CIFAR_read_image_one_channel(mnist_img, input_image_signal_channel_size, input_int_2, 0);
	//MNIST_read_image(image_file, mnist_img, training_set_number);
	int *mnist_label = new int[training_set_number];
	string image_label_file = "train-labels-idx1-ubyte";
	//CIFAR_read_label(mnist_label, 0);
	//MNIST_read_label(image_label_file, mnist_label, training_set_number);
	//special_function: learn one category
	printf("=0=\n");
	int learn_one_digit = 0;
	int *num_one_digit_img = new int[1];
	if(learn_one_digit){
		MNIST_labeling("abc", 60000, mnist_img, mnist_label, mnist_img, num_one_digit_img, spiking_neuron_num, 1, 5);
		printf("Learning only one digit, number of img: %d\n", num_one_digit_img);
	}

	//int synapse_size = SIZE*SIZE;
	//cout<<SIZE<<endl;
    Neuron *NeuronList = new Neuron[spiking_neuron_num];
    Input_neuron *Input_neuronlist = new Input_neuron[input_neuron_num];
	//unsigned char *synapse_timer = new unsigned char[synapse_size];  //this is the array that stores timer used in STPD. e.g Neuron x --->  Neuron y Spike! In the array index [(x-1)*SIZE+(y-1)]  => 1
	//curandState_t *states;
	//float *random_number_list = new float[SIZE];
	float *log_v_host = new float[MAX_TIME];
	float *log_spike_host = new float[total_depth_number];

	float *log_total_spike_host = new float[SIZE];
	for(int i=0; i < SIZE; i++){
		log_total_spike_host[i] = 0;
	}
	int *spike_flag = new int[CNN_total_layer_num];
	for(int i=0; i < CNN_total_layer_num; i++){
		spike_flag[i] = 0;
	}
	for(int i=0; i<total_depth_number; i++) log_spike_host[i] = 0;
	//init_log_v(log_v_host);
	//init_data_log(log_v_host,log_spike_host,log_total_spike_host, MAX_TIME);
	neuron_list_init(NeuronList, spiking_neuron_num);
	input_neuron_list_init(Input_neuronlist, input_neuron_num);
	printf("=1=\n");
	//
	if(resume_learning){
		printf("RESUME LEARNING\n");
		read_neuron_list(NeuronList, 1, "device2_output_network.txt");
	}else{
		read_neuron_list(NeuronList, 1, "spike_cnn.txt");
	}
    //write_neuron_list(NeuronList, "learning_output_confirm.txt", SIZE);
	//check_neuron(NeuronList, 800, 820);


	//Neuron *old_device_neurons;
	//unsigned char *snapse_timer_device;
	float *log_v;
	float *log_spike;
	float *log_spike_default;
	float *log_total_spike;
	int *spike_flag_device;


    printf("2\n");
	//random number function:

    float rand_list_size_to_total_connection_ratio = 1;
	int rand_numb_size = SPIKING_NEURON_NUM*MAX_CONNECTION;

	int SIZE_PER_SIDE = sqrt(rand_numb_size)+1;
    dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );
    dim3 dimGrid( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));
	dim3 print_grid(1);
	dim3 print_block(1);
	dim3 dimBlock_unit( 1, 1 );
	dim3 dimGrid_unit(1, 1);
    printf("2.1\n");

	curandState_t *states;
	cudaMalloc((void **)&states, rand_numb_size * sizeof(curandState_t));
//	if (STOCHASTIC_STDP) rand_init<<<dimGrid,dimBlock>>>(time(0), rand_numb_size, states);
//	float *random_number_list = new float[rand_numb_size];
//	float *random_number_list_device;
//	SIZE_PER_SIDE = sqrt(rand_numb_size)+1;
//	dim3 dimBlock_synapse( ThreadsPerBlock, ThreadsPerBlock );
//	dim3 dimGrid_synapse( (SIZE_PER_SIDE/dimBlock.x+1), (SIZE_PER_SIDE/dimBlock.y+1));

//	cudaMalloc((void **)&random_number_list_device,rand_numb_size*sizeof(float));
//	cudaMemcpy(random_number_list_device,random_number_list,rand_numb_size*sizeof(float),cudaMemcpyHostToDevice);
//	if (STOCHASTIC_STDP) random<<<dimGrid_synapse,dimBlock_synapse>>>(random_number_list_device, rand_numb_size, states);

    curandGenerator_t gen_uniform;
    float *random_number_list_device;
    cudaMalloc((void **)&random_number_list_device,rand_numb_size*sizeof(float));
    curandCreateGenerator(&gen_uniform, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen_uniform, time(0));
    curandGenerateUniform(gen_uniform, random_number_list_device, rand_numb_size);


    curandGenerator_t gen_normal;
    float *random_number_normal_device;
    float normal_mean = 0;
    float normal_sd = 5.0;
    cudaMalloc((void **)&random_number_normal_device,rand_numb_size*sizeof(float));
    curandCreateGenerator(&gen_normal, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen_normal, time(0));
    curandGenerateNormal(gen_normal, random_number_normal_device, rand_numb_size, normal_mean, normal_sd);


//    printf("2.11\n");
    //Setting up input instance matrix:
    float **d_input_instance;
    float **d_convolution_result;
    float **h_input_instance;
    float **h_convolution_result;
    float *probe = new float[1000];
	int instance_array_size = CNN_total_layer_num;
	cudaMalloc(&d_input_instance, instance_array_size*sizeof(float *));
	int convolution_result_size = CNN_total_layer_num - 1;
	cudaMalloc(&d_convolution_result, convolution_result_size*sizeof(float *));
    h_input_instance = (float**)malloc(instance_array_size * sizeof(float*));
    h_convolution_result = (float**)malloc(convolution_result_size * sizeof(float*));
    CNN_util(network_config, d_input_instance, d_convolution_result, h_input_instance, h_convolution_result, 0);
    printf("2.2\n");
//	float **add = &h_convolution_result[0];
//	printf("Address On GPU: %p\n", add);

    //========Setting up device neuron list============

	Neuron *Neuron_list_device;
    Input_neuron *Input_neuronlist_device;
    cudaMalloc((void **)&Neuron_list_device, spiking_neuron_num*sizeof(Neuron));
    cudaMalloc((void **)&Input_neuronlist_device, input_neuron_num*sizeof(Input_neuron));
    //cudaMalloc((void **)&old_device_neurons, SIZE*sizeof(Neuron));

    //cudaMalloc((void **)&states, SIZE * sizeof(curandState_t));
    cudaMalloc((void **)&log_v, MAX_TIME * sizeof(float));
    cudaMalloc((void **)&log_spike, total_depth_number * sizeof(float));
    cudaMalloc((void **)&log_spike_default, total_depth_number * sizeof(float));
    //cudaMalloc((void **)&log_total_spike, SIZE * sizeof(float));
    gpuErrchk( cudaMalloc((void **)&log_total_spike, SIZE * sizeof(float)) );
    cudaMalloc((void **)&spike_flag_device, instance_array_size*sizeof(int));
    //rand_init<<<dimGrid,dimBlock>>>(time(0), states);
    printf("2.3\n");
    cudaMemcpy(Neuron_list_device,NeuronList,spiking_neuron_num*sizeof(Neuron),cudaMemcpyHostToDevice);
    cudaMemcpy(Input_neuronlist_device,Input_neuronlist,input_neuron_num*sizeof(Input_neuron),cudaMemcpyHostToDevice);
    //cudaMemcpy(old_device_neurons,NeuronList,SIZE*sizeof(Neuron),cudaMemcpyHostToDevice);
    //cudaMemcpy(random_number_list_device, random_number_list, SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(log_v,log_v_host,MAX_TIME*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(log_spike,log_spike_host,total_depth_number*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(log_spike_default,log_spike_host,total_depth_number*sizeof(float),cudaMemcpyHostToDevice);
    gpuErrchk( cudaMemcpy(log_total_spike,log_total_spike_host,SIZE*sizeof(float),cudaMemcpyHostToDevice) );
    cudaMemcpy(spike_flag_device,spike_flag,instance_array_size*sizeof(int),cudaMemcpyHostToDevice);
    printf("3\n");
    //cout<<"network size: "<<SIZE<<endl;
    int network_size = SIZE;

    int max_time = MAX_TIME;


    //cudaMemcpy(Neuron_list_device,old_device_neurons,sizeof(Neuron)*SIZE,cudaMemcpyDeviceToDevice);
    //first change raw img data into frequency
    int mnist_start_index = 0;
    int mnist_end_index = input_neuron_num;
    //change pixel signal to frequency

    MNIST_drive(NeuronList, Input_neuronlist, mnist_img, network_size, training_set_number, mnist_start_index, mnist_end_index, max_frequency, min_frequency, 1);
    std::srand ( unsigned ( std::time(0) ) );
    std::vector<int> myvector;
    for (int i=0; i<training_set_number; ++i) myvector.push_back(i); // 1 2 3 4 5 6 7 8 9

    if(shuffle_image){
    	  std::random_shuffle ( myvector.begin(), myvector.end() );
    }

    cudaDeviceSynchronize();

    //data_check(Neuron_list_device,log_total_spike,SIZE,1);
    float *one_mnist_img = new float[input_neuron_num];

    clock_t iter_start, iter_log;
    iter_start = clock();
    int log_interval = MAX_TIME/10;
    //read_filter_GPU_one_layer<<<1, 1>>>(d_network_config, h_filter_array[0], 1);
    //read_filter_GPU<<<1, 1>>>(d_network_config, d_filter_array);

    int reiter_run = 1;

    int time = 0;
    int training_img_index = 0;

    //============now load all convolution settings===========
	for(int layer_iter=0;layer_iter<CNN_total_layer_num;layer_iter++){
		if (layer_iter==0) {
			convolution_kernel_setup(convolution_settings, network_config, layer_iter);
		}else{
			if (layer_iter!=(CNN_total_layer_num-1)) convolution_kernel_setup(convolution_settings, network_config, layer_iter);
		}
	}

	while (time<max_time){
    	//cout<<endl<<" It: "<<time<<endl;
        //random<<<dimGrid,dimBlock>>>(random_number_list_device, states);
    	//first create an array of 1 MNIST image
//    	if(STOCHASTIC_STDP || STOCHASTIC_ROUNDING){
//            random<<<dimGrid_synapse,dimBlock_synapse>>>(random_number_list_device, rand_numb_size, states);
//    	}

    	if(DEVICE_VARIATION){
            curandGenerateNormal(gen_normal, random_number_normal_device, rand_numb_size, normal_mean, normal_sd);
    	}
    	if(STOCHASTIC_STDP || STOCHASTIC_ROUNDING){
    	    curandGenerateUniform(gen_uniform, random_number_list_device, rand_numb_size);
    	}

    	if(time%log_interval == 0){
    		//cudaMemcpy(NeuronList,Neuron_list_device,SIZE*sizeof(Neuron),cudaMemcpyDeviceToHost);
    		printf("NN data copy complete\n");
    		string interval_file_name = "device2_output_at_iter_" + to_string(time) + ".txt";
    	    //data_check(NeuronList,log_total_spike,SIZE, mnist_start_index, mnist_end_index, 2, (to_string(time)+"_"));
    		//if (time>0) write_neuron_list(NeuronList, interval_file_name, network_size);
    	}

    	if(time%tenpercent_iter == 0){
    		iter_log = clock();
    		cout<<to_string(10*(time/tenpercent_iter))<<"% done, time used is: " << (iter_log - iter_start)/1000 << " (ms)" << endl;
    	}
    	//fault below here:

    	if(time%training_time_each_img==0){//at the beginning of each img's training, load into
    		//cout<<"Image Load Iter: "<<time<<endl;
			int locate_index = myvector[training_img_index];
			//cout<<"loading index: "<<locate_index<<endl;
    		for(int i=0;i<input_neuron_num;i++){
    			one_mnist_img[i] = mnist_img[locate_index*input_neuron_num+i];
    		}
//    	    for (int y=0; y<28; ++y) {
//    	    	    for (int x=0; x<28; ++x) {
//    	    	      std::cout << ((one_mnist_img[y*28+x] <= 1.1)? ' ' : '*');
//    	    	      //std::cout << int(one_mnist_img[y*28+x]) << ' ';
//    	    	    }
//    	    	    std::cout << std::endl;
//    	    }
    		MNIST_drive(Neuron_list_device, Input_neuronlist_device, one_mnist_img, input_neuron_num, training_set_number, mnist_start_index, mnist_end_index, max_frequency, min_frequency, 0);
    		//MNIST_drive(old_device_neurons, one_mnist_img, network_size,training_set_number, mnist_start_index, mnist_end_index, max_frequency, min_frequency, 0);
    		training_img_index ++;
    		if(training_img_index>=training_set_number-1){
    	    	std::random_shuffle ( myvector.begin(), myvector.end() );
    			training_img_index = 0;
    		}
    		//confirm the data in signal neuron
    		//cudaMemcpy(NeuronList,Neuron_list_device,SIZE*sizeof(Neuron),cudaMemcpyDeviceToHost);
    		//data_check(NeuronList,log_total_spike,SIZE, mnist_start_index, mnist_end_index, 3);
    		//printf("\n\n\n************************\n\n\n\n");
    	}
    	//cout<<"One IMG loaded"<<endl;
    	//enter spiking neuron simulation:


    	for(int layer_iter=0;layer_iter<CNN_total_layer_num;layer_iter++){
    		int convolution_result_index = layer_iter - 1;
    		if (layer_iter==0) {//fault at convolution kernel and spiking cnn
    			convolution_result_index = 0;
    	    	//CNN_struct *settings; int layer_index; float **d_input_2d; float **filter_2d; float **output_2d;
    	    	//convolution_kernel(settings, layer_index, d_input_2d, filter_2d, output_2d);
    	    	//problem is in spiking_cnn_main
    			spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, input_float, time);
    			//spiking_cnn_main(Neuron_list_device, old_device_neurons, d_network_config, random_number_list_device, d_convolution_result[convolution_result_index], d_input_instance[layer_iter], layer_iter, network_size, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, time);
    			//spiking_cnn_main(Neuron_list_device, old_device_neurons, d_network_config, random_number_list_device, d_convolution_result[0], d_input_instance[0], layer_iter, network_size, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, time);
    			convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
    		}else{
    			//printf("In layer: %d\n", layer_iter);
    			spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, input_float, time);
    			//network_config->layer[layer_iter].depth;
    			if (layer_iter!=(CNN_total_layer_num-1)) convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
				synapse_drive_cnn_v2(Neuron_list_device, Input_neuronlist_device, network_config, d_network_config, d_filter_array, layer_iter, spiking_neuron_num, input_neuron_num, syn_timer_max, connection_size, random_number_list_device, random_number_normal_device, states, -1.0, -1.0);//STDP
    		}

    	}
    	//=================TRY WITH LAYER wise inhibition=====================
    	//cudaDeviceSynchronize();
    	lateral_inhibition_mother_thread<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, inhibition_time, d_network_config, spike_flag_device);
//    	cudaMemcpy(spike_flag, spike_flag_device, CNN_total_layer_num*sizeof(int),cudaMemcpyDeviceToHost);
//    	for(int layer_iter=0;layer_iter<CNN_total_layer_num;layer_iter++){
//    		//if (layer_iter==0) printf("Layer[%d] SpikeFlag: %d\n", layer_iter, spike_flag[layer_iter]);
//            if(spike_flag[layer_iter]>0){//use lateral inhibition
//            	spiking_learning_drive(Neuron_list_device, network_size, inhibition_time, log_total_spike, target_frequency, time, log_spike, layer_iter, network_config, 4);
//            }
//    		spike_flag[layer_iter] = 0;
//    	}
//    	cudaMemcpy(spike_flag_device,spike_flag,CNN_total_layer_num*sizeof(int),cudaMemcpyHostToDevice);
//    	cudaMemcpy(log_spike,log_spike_default,total_depth_number*sizeof(float),cudaMemcpyDeviceToDevice);	//set the log_spike to default value

	//=================TRY WITH NO LAYERAL INHIBITION, MAY BE WRONG=====================
    	//printf("network_size: %d", network_size);
    	//spiking_learning_drive(Neuron_list_device, network_size, inhibition_time, log_total_spike, target_frequency, time, log_spike, 3); //lateral inhibition
	//==================================================================================
		if(HOMEOSTASIS_ENABLE){
			if(time%HOMEOSTASIS_UPDATE_FREQUENCY == 0 && time != 0){
				//spiking_learning_drive(Neuron_list_device, network_size, inhibition_time, log_total_spike, target_frequency, time, log_spike, 0, 1);
			}
		}
        //cudaDeviceSynchronize();


        //if any neuron spikes, run inhibition
	//cudaMemcpy(spike_flag, spike_flag_device, CNN_total_layer_num*sizeof(int),cudaMemcpyDeviceToHost);
	//printf("AtTime:%d_spike_flag_is:%d\n",time,spike_flag[0]);
        //if(spike_flag[0]>0){//use lateral inhibition
        	//spiking_learning_drive(Neuron_list_device, network_size, inhibition_time, log_total_spike, target_frequency, time, log_spike, 0);
        	//spike_flag[0] = 0;
    	    	//cudaMemcpy(spike_flag_device,spike_flag,sizeof(int),cudaMemcpyHostToDevice);
        //}

        //cudaMemcpy(old_device_neurons,Neuron_list_device,sizeof(Neuron)*SIZE,cudaMemcpyDeviceToDevice);
        //cudaDeviceSynchronize();
    	time ++;
    }
    //spiking_learning_drive(Neuron_list_device, network_size, inhibition_time, 2);
	//cudaDeviceSynchronize();

	filter_util(network_config, Neuron_list_device, spiking_neuron_num, input_neuron_num, h_filter_array, d_filter_array, 2);
    cudaMemcpy(NeuronList,Neuron_list_device,spiking_neuron_num*sizeof(Neuron),cudaMemcpyDeviceToHost);
    cudaMemcpy(log_v_host,log_v,MAX_TIME*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(log_spike_host,log_spike,total_depth_number*sizeof(float),cudaMemcpyDeviceToHost);
    gpuErrchk( cudaMemcpy(log_total_spike_host,log_total_spike,SIZE*sizeof(float),cudaMemcpyDeviceToHost) );


    //print out the synapse conductance data
    //data_check(NeuronList,log_total_spike,SIZE, mnist_start_index, mnist_end_index, 2);

    ofstream myfile ((index_prefix+"device2_spike_of_neuron_out.csv"));
    if (myfile.is_open()){
    	//myfile << "This is a new test\n";
    	for(int i=0; i < SIZE; i++){
    		//printf("_%f_", log_v_host[i]);
    		myfile << log_total_spike_host[i] << ", ";
    	}
    	myfile.close();
    }

    ofstream myfile_p ((index_prefix+"probe.csv"));
    if (myfile_p.is_open()){
    	//myfile << "This is a new test\n";
    	for(int i=0; i < 1000; i++){
    		//printf("_%f_", log_v_host[i]);
    		myfile_p << probe[i] << ", ";
    	}
    	myfile_p.close();
    }

//
//    ofstream myfile_0 ((index_prefix+"device2_out_v.csv"));
//    if (myfile_0.is_open()){
//    	//myfile << "This is a new test\n";
//    	for(int i=0; i < MAX_TIME; i++){
//    		//printf("_%f_", log_v_host[i]);
//    		myfile_0 << log_v_host[i] << ", ";
//    	}
//    	myfile.close();
//    }
//
//    ofstream myfile_2 ((index_prefix+"device2_spike_of_one.csv"));
//    if (myfile_2.is_open()){
//    	//myfile << "This is a new test\n";
//    	for(int i=0; i < MAX_TIME; i++){
//    		//printf("_%f_", log_v_host[i]);
//    		myfile_2 << log_spike_host[i] << ", ";
//    	}
//    	myfile_2.close();
//    }



    cudaMemcpy(h_filter_array, d_filter_array, filter_array_size* sizeof(float*), cudaMemcpyDeviceToHost);
	filter_util(network_config, NeuronList, network_size, input_neuron_num, h_filter_array, d_filter_array, 1);	//write filter to file
    write_neuron_list(NeuronList, (index_prefix+"device2_output_network.txt"), spiking_neuron_num);
    data_check(NeuronList,log_total_spike,SIZE, mnist_start_index, mnist_end_index, 2, "");
    //===clean up===
    //delete[] random_number_list;
    delete[] log_v_host;
	delete[] NeuronList;
	delete[] log_spike_host;
	delete[] log_total_spike_host;
	delete[] mnist_img;
	delete[] NeuronList_temp;
	delete[] one_mnist_img;
	delete[] probe;
//	delete[] random_number_list;
	delete[] mnist_label;
	delete[] spike_flag;
	delete[] num_one_digit_img;
	//cudaFree(states);
	cudaFree(log_v);
	cudaFree(log_spike);
	cudaFree(log_total_spike);
	cudaFree(Neuron_list_device);
	//cudaFree(old_device_neurons);
	cudaFree(random_number_list_device);
	cudaFree(d_network_config);
	cudaFree(states);
	cudaFree(spike_flag_device);
	cudaFree(log_spike_default);

}

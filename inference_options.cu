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





void run_cnn_multilayer_inference(string index_prefix, float input_float, float input_float_2, int input_int, int input_int_2, string input_img){
	cout << "Running CNN Multilayer Inference" << endl << endl;
	cout<<"Functions: \n"<<"0. One image inference\n"<<"1. load a folder\n"<<"2. One folder separate run";
	cout << endl;
	cout<<"Function Select: ";
	int mode_select;
	cin >> mode_select;

	switch (mode_select){
		case 0:
		{
			cout<<"One image inference selected"<<endl;
			cout<<endl;
		}
		break;
		case 1:
		{
			printf("Case 1 selected");
			cout<<"How many iterations for each image:";
			cin>>input_int;
		}
		break;
		case 2:
		{
			printf("Case 2 selected");
			cout<<"How many iterations for each image:";
			cin>>input_int;
		}
		break;
	}



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
	filter_util(network_config, NeuronList_temp, 0,0,  h_filter_array, d_filter_array, index_prefix, 0);

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
	int per_run_img_num=6;
	int training_set_number = 23;
	if(mode_select==1){
		cout<<"how many images to read: ";
		cin>>training_set_number;
	}
	else if(mode_select==0){
		training_set_number=1;
	}else if(mode_select==2){
		cout<<"how many images in total: ";
		cin>>training_set_number;
		cout<<"how many images in per run: ";
		cin>>per_run_img_num;
	}

	int training_time_each_img = input_int;
	int calculated_total_time = training_time_each_img*training_set_number;
	#undef MAX_TIME
	#define MAX_TIME calculated_total_time
	printf("==Training Total Iter: %d==", MAX_TIME);
	int total_neuron_num = 0;
	int total_spiking_num = 0;
	for(int i=0;i<CNN_total_layer_num;i++){
		total_neuron_num += network_config->layer[i].neuron_num;
		if(i!=0) total_spiking_num += network_config->layer[i].neuron_num;
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


	float max_frequency = 120; //in Hz default 22
	float min_frequency = 30;

	int input_neuron_num = input_image_w*input_image_l*input_image_channel;
	int input_image_signal_channel_size  = input_image_w*input_image_l;
	int spiking_neuron_num = SPIKING_NEURON_NUM;
	int output_layer_neuron_num = OUTPUT_LAYER_NEURON_NUM;
	int tenpercent_iter = MAX_TIME/10;
	int connection_size = MAX_CONNECTION;
	int syn_timer_max = 25;
	int input_signal_width = 15;	//default 25
	int inhibition_time = 20;	//default 10

	float target_frequency_param = 0.5;
	float target_frequency = 100;
	float *mnist_img = new float[input_neuron_num*training_set_number];
	for(int i=0;i<input_neuron_num*training_set_number;i++) mnist_img[i] = 0;
	string image_file = "train-images-idx3-ubyte";//"train_dataset_noisy_cifar";//"fashion-train-images-idx3-ubyte";//"train_dataset_noisy";//"train_dataset_noisy"; //"train-images-idx3-ubyte";

	if(mode_select==0){
		cout<<"image directory: ";
		cin>>image_file;
	}else if(mode_select==1){
		cout<<"image folder directory: ";
		cin>>image_file;
	}else if(mode_select==2){
		cout<<"image folder directory: ";
		cin>>image_file;
	}
	cout<<endl<<"Image loading"<<endl;
	clock_t load_start = clock();
    std::vector<std::string> folder_list;
    int input_folder_cnt = 0;
	if(input_image_channel==1){
		//CIFAR_read_image(mnist_img, input_neuron_num, 0, 1);
		//GTVIR_read_image(mnist_img, input_neuron_num, training_set_number);
		//MNIST_read_image(image_file, mnist_img, training_set_number);
		//read_polygon("/inverse_polygon/drawings", mnist_img, training_set_number);
		if(mode_select==0)read_one_image(image_file, mnist_img, training_set_number);//"/home/xshe6/Documents/CUDA/Spike_CNN/Debug/inverse_polygon/drawings/Slide15.png"
		else if(mode_select==1)read_polygon(image_file, mnist_img, training_set_number); //"/inverse_polygon/drawings"
		else if(mode_select==2)read_polygon(image_file, mnist_img, training_set_number); //"/inverse_polygon/drawings"

	}else{
		bool learn_imageNet = false;
		if(learn_imageNet){

			ifstream file ("imageNet_folder_list.csv");
			string val;

		    while(file.good()) {
				getline(file, val, ',');
		    	folder_list.push_back(val);
		    }

			imageNET_read_image(folder_list[input_folder_cnt], mnist_img, training_set_number);
		}else{
			CIFAR_read_image(mnist_img, input_neuron_num, training_set_number, 0, 0);
			//KAIST_PED_read_image("", mnist_img , training_set_number);
		}

	}
	clock_t load_end = clock();

	cout<<endl<<"Image loading done"<<", time used is " << (load_end - load_start)/1000 << " (ms)"<<endl;
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
		//read_neuron_list(NeuronList, 1, "spike_cnn.txt");
	}else{
		read_neuron_list(NeuronList, 1, "device2_output_network.txt");
		//read_neuron_list(NeuronList, 1, "device2_output_network.txt");
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

	int SIZE_PER_SIDE_whole_network = sqrt(spiking_neuron_num)+1;
    dim3 dimBlock_whole_network( ThreadsPerBlock*2, ThreadsPerBlock );
    dim3 dimGrid_whole_network( (SIZE_PER_SIDE_whole_network/dimBlock.x+1), (SIZE_PER_SIDE_whole_network/dimBlock.y+1));
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


    printf("2.11\n");
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
    int first_layer_time = 1;
    int second_layer_time = first_layer_time+(max_time-first_layer_time)*2/3;
    if(CNN_total_layer_num==3){
        first_layer_time = 1;
    	second_layer_time = max_time;
    }else if (CNN_total_layer_num==2){
    	first_layer_time = max_time;
    	second_layer_time = max_time + 1;
    }
    if(resume_learning){
    	first_layer_time = 100;
    	second_layer_time = max_time;
    }

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
	bool enable_log_interval = false;
    //read_filter_GPU_one_layer<<<1, 1>>>(d_network_config, h_filter_array[0], 1);
    //read_filter_GPU<<<1, 1>>>(d_network_config, d_filter_array);

//    int reiter_run = 1;

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
	copy_filter_to_cuDNN(Neuron_list_device, d_network_config, d_filter_array, spiking_neuron_num);
    cudaDeviceSynchronize();
    cout<<"Layer 0 has: "<<network_config->layer[0].neuron_num<<endl;
	while (time<max_time){
		//if(time==first_layer_time)MNIST_drive(NeuronList, Input_neuronlist, mnist_img, network_size, training_set_number, mnist_start_index, mnist_end_index, max_frequency*2, min_frequency, 1);
    	if(DEVICE_VARIATION){
            curandGenerateNormal(gen_normal, random_number_normal_device, rand_numb_size, normal_mean, normal_sd);
    	}
    	if(STOCHASTIC_STDP || STOCHASTIC_ROUNDING){
    	    curandGenerateUniform(gen_uniform, random_number_list_device, rand_numb_size);
    	}

    	if(time%log_interval == 0 && enable_log_interval){
    	    cudaMemcpy(NeuronList,Neuron_list_device,spiking_neuron_num*sizeof(Neuron),cudaMemcpyDeviceToHost);
    		printf("NN data copy complete\n");
    		string interval_file_name = "device2_output_at_iter_" + to_string(time) + ".txt";
    		//string interval_weight_file_name = to_string(time);
    	    //data_check(NeuronList,log_total_spike,SIZE, mnist_start_index, mnist_end_index, 2, (to_string(time)+"_"));
    		if (time>0) {
    			write_neuron_list(NeuronList, interval_file_name, spiking_neuron_num);
    		    cudaMemcpy(h_filter_array, d_filter_array, filter_array_size* sizeof(float*), cudaMemcpyDeviceToHost);
    			filter_util(network_config, NeuronList, network_size, input_neuron_num, h_filter_array, d_filter_array, to_string(time), 1);	//write filter to file
    		}
    	}

    	if(time==first_layer_time){

        	float start_depth = network_config->layer[1].first_depth_id - 0.1;
        	float end_depth = network_config->layer[1].last_depth_id + 0.1;
    		cout<<"Changing threshold, start: "<< start_depth<<" end: "<<end_depth<<endl;
    		change_threshold<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, -65.2);

        	start_depth = network_config->layer[2].first_depth_id - 0.1;
        	end_depth = network_config->layer[2].last_depth_id + 0.1;
    		//cout<<"Changing threshold, start: "<< start_depth<<" end: "<<end_depth<<endl;
    		//change_threshold<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, -65.0);
    		cudaDeviceSynchronize();
    	}else if(time==second_layer_time){
        	float start_depth = network_config->layer[1].first_depth_id - 0.1;
        	float end_depth = network_config->layer[1].last_depth_id + 0.1;
    		cout<<"Changing threshold, start: "<< start_depth<<" end: "<<end_depth<<endl;
    		change_threshold<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, -73.2);
        	start_depth = network_config->layer[2].first_depth_id - 0.1;
        	end_depth = network_config->layer[2].last_depth_id + 0.1;
    		cout<<"Changing threshold, start: "<< start_depth<<" end: "<<end_depth<<endl;
    		change_threshold<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, -73.2);
        	start_depth = network_config->layer[3].first_depth_id - 0.1;
        	end_depth = network_config->layer[3].last_depth_id + 0.1;
    		cout<<"Changing threshold, start: "<< start_depth<<" end: "<<end_depth<<endl;
    		change_threshold<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, -71.0);
    		training_time_each_img = training_time_each_img*1.5;
    		cudaDeviceSynchronize();
    	}

    	if(time%tenpercent_iter == 0){
    		iter_log = clock();
    		cout<<to_string(10*(time/tenpercent_iter))<<"% done, time used is: " << (iter_log - iter_start)/1000 << " (ms)" << endl;
    	}
    	//fault below here:

    	if(time%training_time_each_img==0){//at the beginning of each img's training, load into
    		//cout<<"Image Load Iter: "<<time<<endl;
    		//cout<<";;;"<<training_img_index<<"\n";
    		if(mode_select==2 && time!=0)
    		{
				if(training_img_index%per_run_img_num==0){//print current spike numbers and reset neurons
				    gpuErrchk( cudaMemcpy(log_total_spike_host,log_total_spike,SIZE*sizeof(float),cudaMemcpyDeviceToHost) );
				    ofstream myfile;
				    myfile.open((index_prefix+"inf_multirun_output_spike.csv"), std::ios_base::app);
				    if (myfile.is_open()){
				    	//myfile << "This is a new test\n";
				    	cout<<"Checking number of neuron spike:\n";
				    	for(int i=0; i < spiking_neuron_num; i++){
				    		//printf("_%f_", log_v_host[i]);
				    		if(CNN_total_layer_num>2){
								if(i>=network_config->layer[CNN_total_layer_num-2].neuron_num){
									myfile << log_total_spike_host[i] << ", ";
									cout<<log_total_spike_host[i]<<" ";
								}
				    		}
				    		else{
								myfile << log_total_spike_host[i] << ", ";
								cout<<log_total_spike_host[i]<<" ";
				    		}
				    	}
				    	cout<<endl;
				    	myfile<<endl;
				    	myfile.close();
				    }
					for(int i=0; i < SIZE; i++){
						log_total_spike_host[i] = 0;
					}
				    gpuErrchk( cudaMemcpy(log_total_spike,log_total_spike_host,SIZE*sizeof(float),cudaMemcpyHostToDevice) );
				    //print spike numbers done
				    cudaMemcpy(Neuron_list_device,NeuronList,spiking_neuron_num*sizeof(Neuron),cudaMemcpyHostToDevice); //reset the neurons
				    if(CNN_total_layer_num>2){

						float start_depth = network_config->layer[1].first_depth_id - 0.1;
						float end_depth = network_config->layer[1].last_depth_id + 0.1;
						cout<<"Changing threshold, start: "<< start_depth<<" end: "<<end_depth<<endl;
						change_threshold<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, -65.2);

						start_depth = network_config->layer[2].first_depth_id - 0.1;
						end_depth = network_config->layer[2].last_depth_id + 0.1;
						cout<<"Changing threshold, start: "<< start_depth<<" end: "<<end_depth<<endl;
						//change_threshold<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, -65.0);
						//cudaDeviceSynchronize();
				    }else{
						float start_depth = network_config->layer[1].first_depth_id - 0.1;
						float end_depth = network_config->layer[1].last_depth_id + 0.1;
						//cout<<"Changing threshold, start: "<< start_depth<<" end: "<<end_depth<<endl;
						//change_threshold<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, -64);
				    }

				}
    		}
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
    		if(training_img_index>training_set_number-1) training_img_index = 0;
    		if(training_img_index>training_set_number-1 && shuffle_image){
    	    	std::random_shuffle ( myvector.begin(), myvector.end() );
    			training_img_index = 0;
    		}


    	}
    	//cout<<"One IMG loaded"<<endl;
    	//enter spiking neuron simulation:
    	int one_layer_neuron_num = 0;
    	if(time<first_layer_time){
			for(int layer_iter=0;layer_iter<CNN_total_layer_num;layer_iter++){
				one_layer_neuron_num = network_config->layer[layer_iter].neuron_num;
				int convolution_result_index = layer_iter - 1;
				if (layer_iter==0) {//fault at convolution kernel and spiking cnn
					convolution_result_index = 0;
					spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, input_float, 0, false);
					convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
				}else if(layer_iter==1){
					spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, input_float, 0, true);
					if (layer_iter!=(CNN_total_layer_num-1)) convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, \
							h_convolution_result, probe);
					//synapse_drive_cnn_v2(Neuron_list_device, Input_neuronlist_device, network_config, d_network_config, d_filter_array, layer_iter, \
							spiking_neuron_num, input_neuron_num, syn_timer_max, connection_size, random_number_list_device, random_number_normal_device, states, -1.0, -1.0);//STDP
				}
			}
			//=================TRY WITH LAYER wise inhibition=====================
	    	if(depth_wise_inhibition) {
	    		lateral_inhibition_depth_wise_mother_thread<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, input_image_channel, inhibition_time, d_network_config, log_spike, total_depth_number);
	    	}else if(through_depth_inhibition){

	    	}else if(apply_local_inhibition){

	    	}
	    	else{
	    		lateral_inhibition_mother_thread<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, 1, inhibition_time, d_network_config, spike_flag_device);
	    	}
			if(HOMEOSTASIS_ENABLE){
				if(time%HOMEOSTASIS_UPDATE_FREQUENCY == 0 && time != 0){
					//spiking_learning_drive(Neuron_list_device, network_size, inhibition_time, log_total_spike, target_frequency, time, log_spike, 0, 1);
				}
			}
    	}else if(time<second_layer_time){
			for(int layer_iter=0;layer_iter<CNN_total_layer_num;layer_iter++){
				one_layer_neuron_num = network_config->layer[layer_iter].neuron_num;
				int convolution_result_index = layer_iter - 1;
				if (layer_iter==0) {//fault at convolution kernel and spiking cnn
					convolution_result_index = 0;
					spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, input_float, 0, false);
					convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
				}else if(layer_iter==1 || layer_iter==3){
					spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, input_float, 0, false);
					if (layer_iter!=(CNN_total_layer_num-1)) convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
				}else{
					spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, 0.1*input_float, 0, true);
					if (layer_iter!=(CNN_total_layer_num-1)) convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, \
							h_convolution_result, probe);
					//synapse_drive_cnn_v2(Neuron_list_device, Input_neuronlist_device, network_config, d_network_config, d_filter_array, layer_iter, spiking_neuron_num, input_neuron_num, syn_timer_max, connection_size, random_number_list_device, random_number_normal_device, states, -1.0, -1.0);//STDP
				}

			}
			//=================TRY WITH LAYER wise inhibition=====================
	    	if(depth_wise_inhibition) {
	    		lateral_inhibition_depth_wise_mother_thread<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, input_image_channel+network_config->layer[1].depth, inhibition_time, d_network_config, log_spike, total_depth_number);
	    	}else if(through_depth_inhibition){

	    	}else if(apply_local_inhibition&& CNN_total_layer_num!=3){

	    	}else if(forced_lateral_inhibition_at_last_layer && CNN_total_layer_num==3){//if this is the last layer, use lateral_inhibition
	    		lateral_inhibition_mother_thread<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, 2, inhibition_time, d_network_config, spike_flag_device);
	    	}else{
	    		lateral_inhibition_mother_thread<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, 2, inhibition_time, d_network_config, spike_flag_device);
	    	}
			if(HOMEOSTASIS_ENABLE){
				if(time%HOMEOSTASIS_UPDATE_FREQUENCY == 0 && time != 0){
					//spiking_learning_drive(Neuron_list_device, network_size, inhibition_time, log_total_spike, target_frequency, time, log_spike, 0, 1);
				}
			}

    	}else{
			for(int layer_iter=0;layer_iter<CNN_total_layer_num;layer_iter++){
				one_layer_neuron_num = network_config->layer[layer_iter].neuron_num;
				int convolution_result_index = layer_iter - 1;
				if (layer_iter==0) {//fault at convolution kernel and spiking cnn
					convolution_result_index = 0;
					spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, input_float, 0, false);
					convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
				}else if(layer_iter==1 || layer_iter==2){
					spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, input_float, 0, false);
					if (layer_iter!=(CNN_total_layer_num-1)) convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
				}else{
					spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, input_float, 0, true);
					if (layer_iter!=(CNN_total_layer_num-1)) convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
					//synapse_drive_cnn_v2(Neuron_list_device, Input_neuronlist_device, network_config, d_network_config, d_filter_array, layer_iter, spiking_neuron_num, input_neuron_num, syn_timer_max, connection_size, random_number_list_device, random_number_normal_device, states, -1.0, -1.0);//STDP
				}

			}
			//=================TRY WITH LAYER wise inhibition=====================
	    	if(depth_wise_inhibition) {
	    		lateral_inhibition_depth_wise_mother_thread<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, input_image_channel+network_config->layer[1].depth+network_config->layer[2].depth, inhibition_time, d_network_config, log_spike, total_depth_number);
	    	}else if(apply_local_inhibition){

	    	}else{
	    		lateral_inhibition_mother_thread<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, 1, inhibition_time, d_network_config, spike_flag_device);
	    	}
			if(HOMEOSTASIS_ENABLE){
				if(time%HOMEOSTASIS_UPDATE_FREQUENCY == 0 && time != 0){
					//spiking_learning_drive(Neuron_list_device, network_size, inhibition_time, log_total_spike, target_frequency, time, log_spike, 0, 1);
				}
			}

    	}
    	time ++;
    }
    //spiking_learning_drive(Neuron_list_device, network_size, inhibition_time, 2);
	//cudaDeviceSynchronize();



	filter_util(network_config, Neuron_list_device, spiking_neuron_num, input_neuron_num, h_filter_array, d_filter_array, index_prefix, 2);
    cudaMemcpy(NeuronList,Neuron_list_device,spiking_neuron_num*sizeof(Neuron),cudaMemcpyDeviceToHost);

    cudaMemcpy(log_v_host,log_v,MAX_TIME*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(log_spike_host,log_spike,total_depth_number*sizeof(float),cudaMemcpyDeviceToHost);
    gpuErrchk( cudaMemcpy(log_total_spike_host,log_total_spike,SIZE*sizeof(float),cudaMemcpyDeviceToHost) );


    //print out the synapse conductance data
    //data_check(NeuronList,log_total_spike,SIZE, mnist_start_index, mnist_end_index, 2);

    ofstream myfile ((index_prefix+"inf_out_device2_spike_of_neuron_out.csv"));
    if (myfile.is_open()){
    	//myfile << "This is a new test\n";
    	cout<<"Checking number of neuron spike:\n";
    	for(int i=0; i < spiking_neuron_num; i++){
    		//printf("_%f_", log_v_host[i]);
    		myfile << log_total_spike_host[i] << ", ";
    		if(i>=network_config->layer[1].neuron_num) cout<<log_total_spike_host[i]<<" ";

    	}
    	myfile.close();
    }

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
	//filter_util(network_config, NeuronList, network_size, input_neuron_num, h_filter_array, d_filter_array, index_prefix, 1);	//write filter to file
    write_neuron_list(NeuronList, ("inf_out_device2_output_network.txt"), spiking_neuron_num);
    //data_check(NeuronList,log_total_spike,SIZE, mnist_start_index, mnist_end_index, 2, "");
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

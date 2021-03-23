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

void run_event_based_learning(string index_prefix, float input_float, float input_float_2, int input_int, int input_int_2, string input_img){

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


//===========END of CNN special setting-up phase============

	Convolution_setting_struct *convolution_settings = new Convolution_setting_struct[CNN_total_layer_num];

	//set parameters
	int img_load_max = 100000000;
	int time_per_event = input_int;
	long calculated_total_time = time_per_event*12000*10000;
    if(resume_learning) calculated_total_time = calculated_total_time/3;

	#undef MAX_TIME
	#define MAX_TIME calculated_total_time
	int tenpercent_iter = MAX_TIME/10;
	printf("==Training Total Iter: %d==", MAX_TIME);
	int total_neuron_num = 0;
	int total_spiking_num = 0;
	for(int i=0;i<CNN_total_layer_num;i++){
		total_neuron_num += network_config->layer[i].neuron_num;
		if(i!=0) total_spiking_num += network_config->layer[i].neuron_num;
	}
	total_spiking_num += 5;
	total_neuron_num += 5;
	//total_neuron_num = 20000;
	cout<<endl<<"total neuron num: "<<total_neuron_num<<endl;
	cout<<"total spiking neuron num: "<<total_spiking_num<<endl;
	#undef SIZE
	#define SIZE total_neuron_num
	#undef SPIKING_NEURON_NUM
	#define SPIKING_NEURON_NUM total_spiking_num


//	float max_frequency = 50; //in Hz default 22
//	float min_frequency = 10;
	int training_set_number = 32000;
	bool batch_load = false;
	int batched_load_remain = 0;
	int batch_load_grand_total = 0;
	int img_load_offset = 0;


	if (training_set_number>img_load_max){ //manually set the maximum number of images to be loaded once is 60000
		cout<<"Using batch loading"<<endl;
		batch_load_grand_total = training_set_number;
		batch_load = true;
		batched_load_remain = training_set_number - img_load_max;
		training_set_number = img_load_max;
	}
	int input_neuron_num = input_image_w*input_image_l*input_image_channel;
	int input_image_signal_channel_size  = input_image_w*input_image_l;
	int spiking_neuron_num = SPIKING_NEURON_NUM;
	int output_layer_neuron_num = OUTPUT_LAYER_NEURON_NUM;

	int connection_size = MAX_CONNECTION;
	int syn_timer_max = 25;
	int input_signal_width = 10;	//default 25
	int inhibition_time = 10;	//default 10

	float target_frequency_param = 0.5;
	float target_frequency = 100;

	Event_Camera_Input *events_host = new Event_Camera_Input[img_load_max];
	Event_Camera_Input *events_GPU;
	cudaMalloc((void **)&events_GPU,img_load_max*sizeof(Event_Camera_Input));

	int current_input_file_id = 1;
	int input_file_id_max = 10;
	string image_file = "";
	if (current_input_file_id<10) {
		image_file = "/DVS128_event_based/user0" + to_string(current_input_file_id) + "_event_based.csv";
	}
	else{
		image_file = "/DVS128_event_based/user" + to_string(current_input_file_id) + "_event_based.csv";
	}

	cout<<endl<<"Image loading"<<endl;
	clock_t load_start = clock();
    std::vector<std::string> folder_list;
    int input_folder_cnt = 0;
    int current_total_read_event = 0;
	if(input_image_channel==1 || input_image_channel==2){


		current_total_read_event = IBM_DVS128_event_based(image_file, events_host, img_load_max, img_load_max);
	    cudaMemcpy(events_GPU,events_host,img_load_max*sizeof(Event_Camera_Input),cudaMemcpyHostToDevice);

	}else{
		printf("Input channel error.");
		return;

	}
	clock_t load_end = clock();
	cout<<endl<<"Image loading done"<<", time used is " << (load_end - load_start)/1000 << " (ms)"<<endl;
	//CIFAR_read_image_one_channel(events_host, input_image_signal_channel_size, input_int_2, 0);
	//MNIST_read_image(image_file, events_host, training_set_number);
	int *mnist_label = new int[training_set_number];
	string image_label_file = "train-labels-idx1-ubyte";
	//CIFAR_read_label(mnist_label, 0);
	//MNIST_read_label(image_label_file, mnist_label, training_set_number);
	//special_function: learn one category
	printf("=0=\n");
	int learn_one_digit = 0;
	int *num_one_digit_img = new int[1];


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
		printf("------RESUME LEARNING-------\n");
		read_neuron_list(NeuronList, 1, "spike_cnn.txt");
		//read_neuron_list(NeuronList, 1, "device2_output_network.txt");
		int start_layer = 3;//the layer that starts to learn
		read_neuron_list_special(NeuronList, (start_layer-1), network_config, "device2_output_network_org.txt"); //duplicate the previous layer
		bool do_reset_weight = false;
		if(do_reset_weight){

			float start_depth = network_config->layer[start_layer].first_depth_id - 0.1;
			float end_depth = network_config->layer[start_layer].last_depth_id + 0.1;

			reset_weight(NeuronList, start_depth, end_depth, 1, spiking_neuron_num);

		}
		//read_neuron_list(NeuronList, 1, "spike_cnn.txt");
	}else{
		read_neuron_list(NeuronList, 1, "spike_cnn.txt");
		//read_neuron_list(NeuronList, 1, "device2_output_network.txt");

	}
	//printf("read out one neuron depth: %f", NeuronList[116000].param[7]);
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
    if(STOCHASTIC_STDP || STOCHASTIC_ROUNDING || DEVICE_VARIATION){
    	cudaMalloc((void **)&states, rand_numb_size * sizeof(curandState_t));
		cudaMalloc((void **)&random_number_list_device,rand_numb_size*sizeof(float));
		curandCreateGenerator(&gen_uniform, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(gen_uniform, time(0));
		curandGenerateUniform(gen_uniform, random_number_list_device, rand_numb_size);
    }


    curandGenerator_t gen_normal;
    float *random_number_normal_device;
    float normal_mean = 0;
    float normal_sd = 5.0;
    if(STOCHASTIC_STDP || STOCHASTIC_ROUNDING || DEVICE_VARIATION){
		cudaMalloc((void **)&random_number_normal_device,rand_numb_size*sizeof(float));
		curandCreateGenerator(&gen_normal, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(gen_normal, time(0));
		curandGenerateNormal(gen_normal, random_number_normal_device, rand_numb_size, normal_mean, normal_sd);
    }

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
    printf("3.0\n");
    //cout<<"network size: "<<SIZE<<endl;
    int network_size = SIZE;

    int max_time = MAX_TIME;
    int first_layer_time = max_time*1/6;
    int second_layer_time = max_time*1/3;
    int third_layer_time = max_time*2/3;
    if(CNN_total_layer_num==3){
    	first_layer_time = max_time*1/3;
    	second_layer_time = max_time + 1;
    	third_layer_time = max_time + 1;
    }
    else if (CNN_total_layer_num==2){
    	first_layer_time = max_time;
    	second_layer_time = max_time + 1;
    	third_layer_time = max_time + 1;
    }else if (CNN_total_layer_num==4){
    	first_layer_time = max_time*1/5;
    	second_layer_time = max_time*3/5;
    	third_layer_time = max_time + 1;
    }else if (CNN_total_layer_num==5){
    	first_layer_time = max_time*1/7;
    	second_layer_time = max_time*3/7;
    	third_layer_time = max_time*5/7;
    }
    if(resume_learning){
    	first_layer_time = 1;
        if(CNN_total_layer_num==3) second_layer_time = max_time;
        if(CNN_total_layer_num==4) second_layer_time = 2;
        if(CNN_total_layer_num==4){
        	second_layer_time = 2;
        	third_layer_time = 3;
        }
    }

    //cudaMemcpy(Neuron_list_device,old_device_neurons,sizeof(Neuron)*SIZE,cudaMemcpyDeviceToDevice);
    //first change raw img data into frequency
    int mnist_start_index = 0;
    int mnist_end_index = input_neuron_num;
    //change pixel signal to frequency


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
	bool last_layer_teach = false;

	int total_class = 10;
	int num_per_class = 500;
	int frame_per_seq = 50;

	int rotation_num = 2;
	int translation_num = 1;
	int frame_per_class = frame_per_seq*rotation_num*num_per_class;
	int frame_per_rotation = frame_per_seq;
	int frame_per_translation = frame_per_seq*rotation_num*num_per_class*total_class;

    std::vector<int> seq_vector_head;
    std::vector<int> seq_vector;
    for (int i=0; i<training_set_number/frame_per_seq; ++i) seq_vector_head.push_back(i); // 1 2 3 4 5 6 7 8 9
	std::random_shuffle ( seq_vector_head.begin(), seq_vector_head.end() );

	for (int i=0 ; i<training_set_number/frame_per_seq; ++i){
		int begin_index = seq_vector_head[i];
		for (int j=0; j<frame_per_seq; ++j) seq_vector.push_back(10*begin_index+j);

	}

    //read_filter_GPU_one_layer<<<1, 1>>>(d_network_config, h_filter_array[0], 1);
    //read_filter_GPU<<<1, 1>>>(d_network_config, d_filter_array);

//    int reiter_run = 1;

    int time = 0;
    int training_img_index = 0;

    //============now load all convolution settings===========
	for(int layer_iter=0;layer_iter<CNN_total_layer_num;layer_iter++){
		if (layer_iter==0) {
			cout<<endl<<"Setting up the first layer"<<endl;
			convolution_kernel_setup(convolution_settings, network_config, layer_iter);
		}else{

			if (layer_iter!=(CNN_total_layer_num-1))
			{
				convolution_kernel_setup(convolution_settings, network_config, layer_iter);
				cout<<"Setting up the"<<" layer "<< layer_iter <<endl;
			}
		}
	}
    if(resume_learning){
		copy_filter_to_cuDNN(Neuron_list_device, d_network_config, d_filter_array, spiking_neuron_num);
		cudaDeviceSynchronize();
    }
	float start_depth = network_config->layer[1].first_depth_id - 0.1;
	float end_depth = network_config->layer[1].last_depth_id + 0.1;
	cout<<"Changing threshold, start: "<< start_depth<<" end: "<<end_depth<<endl;
	change_threshold<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, -62.2);
	//return;
	int event_count = 0;
	while (time<max_time){

		//if(time==first_layer_time)MNIST_drive(NeuronList, Input_neuronlist, events_host, network_size, training_set_number, mnist_start_index, mnist_end_index, max_frequency*2, min_frequency, 1);
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

    	if(time%time_per_event){
    		event_count++;
    		while(events_host[event_count].valid==False && event_count<current_total_read_event){
    			event_count++;
    		}

    		if (event_count>=current_total_read_event){
    			cout<<endl<<"Image loading"<<endl;
    			current_input_file_id ++;

    			if(current_input_file_id>input_file_id_max) current_input_file_id = 1;

    			if (current_input_file_id<10) {
    				image_file = "/hdd2/extra_home/xshe6/Event_camera/event_based/user0" + to_string(current_input_file_id) + "_event_based.csv";
    			}
    			else{
    				image_file = "/hdd2/extra_home/xshe6/Event_camera/event_based/user" + to_string(current_input_file_id) + "_event_based.csv";
    			}

    		    current_total_read_event = 0;
				current_total_read_event = IBM_DVS128_event_based(image_file, events_host, img_load_max, img_load_max);
				cout<<"Total loaded:"<< current_total_read_event<<endl;
				cudaMemcpy(events_GPU,events_host,img_load_max*sizeof(Event_Camera_Input),cudaMemcpyHostToDevice);
    			event_count=0;
    		}
    	}

    	if(time==first_layer_time){


    	    gpuErrchk( cudaMemcpy(log_total_spike_host,log_total_spike,SIZE*sizeof(float),cudaMemcpyDeviceToHost) );
    	    ofstream myfile ((index_prefix+"first_stage_device2_spike_of_neuron_out.csv"));
    	    if (myfile.is_open()){
    	    	//myfile << "This is a new test\n";
    	    	for(int i=0; i < SIZE; i++){
    	    		//printf("_%f_", log_v_host[i]);
    	    		myfile << log_total_spike_host[i] << ", ";
    	    	}
    	    	myfile.close();
    	    }

        	float start_depth = network_config->layer[1].first_depth_id - 0.1;
        	float end_depth = network_config->layer[1].last_depth_id + 0.1;
    		//cout<<"Changing threshold, start: "<< start_depth<<" end: "<<end_depth<<endl;
			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, 5, -5.07);
			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, 4, 0.453);
			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, 0, -0.02);
    		change_threshold<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, -63.2);
//    		cout<<"Changing param of long-term neuron, start: "<< start_depth+32<<" end: "<<end_depth<<endl;
//			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth+32, end_depth, 5, -1.6);
//			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth+32, end_depth, 4, 0.16);
//			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth+32, end_depth, 0, -0.001);
    		//change_threshold<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth+32, end_depth, -56.2);

//        	start_depth = network_config->layer[2].first_depth_id - 0.1;
//        	end_depth = network_config->layer[2].last_depth_id + 0.1;
//    		cout<<"Changing threshold, start: "<< start_depth<<" end: "<<end_depth<<endl;
//    		change_threshold<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, -61.0);


    		//training_time_each_img = training_time_each_img*1.3;
    		cudaDeviceSynchronize();

    	}else if(time==second_layer_time){

    	    gpuErrchk( cudaMemcpy(log_total_spike_host,log_total_spike,SIZE*sizeof(float),cudaMemcpyDeviceToHost) );
    	    ofstream myfile ((index_prefix+"second_stage_device2_spike_of_neuron_out.csv"));
    	    if (myfile.is_open()){
    	    	//myfile << "This is a new test\n";
    	    	for(int i=0; i < SIZE; i++){
    	    		//printf("_%f_", log_v_host[i]);
    	    		myfile << log_total_spike_host[i] << ", ";
    	    	}
    	    	myfile.close();
    	    }

        	float start_depth = network_config->layer[1].first_depth_id - 0.1;
        	float end_depth = network_config->layer[1].last_depth_id + 0.1;
    		//cout<<"Changing threshold, start: "<< start_depth<<" end: "<<end_depth<<endl;
    		change_threshold<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, -64);


        	start_depth = network_config->layer[2].first_depth_id - 0.1;
        	end_depth = network_config->layer[2].last_depth_id + 0.1;
    		//cout<<"Changing threshold, start: "<< start_depth<<" end: "<<end_depth<<endl;
			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, 5, -5.07);
			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, 4, 0.453);
			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, 0, -0.02);
    		change_threshold<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, -63);
//    		cout<<"Changing param, start: "<< start_depth+32<<" end: "<<end_depth<<endl;
//			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth+64, end_depth, 5, -1.6);
//			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth+64, end_depth, 4, 0.16);
//			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth+64, end_depth, 0, -0.001);

//        	start_depth = network_config->layer[3].first_depth_id - 0.1;
//        	end_depth = network_config->layer[3].last_depth_id + 0.1;
//    		cout<<"Changing threshold, start: "<< start_depth<<" end: "<<end_depth<<endl;
//    		change_threshold<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, -60.0);

    	}else if(time==third_layer_time){

    	    gpuErrchk( cudaMemcpy(log_total_spike_host,log_total_spike,SIZE*sizeof(float),cudaMemcpyDeviceToHost) );
    	    ofstream myfile ((index_prefix+"third_stage_device2_spike_of_neuron_out.csv"));
    	    if (myfile.is_open()){
    	    	//myfile << "This is a new test\n";
    	    	for(int i=0; i < SIZE; i++){
    	    		//printf("_%f_", log_v_host[i]);
    	    		myfile << log_total_spike_host[i] << ", ";
    	    	}
    	    	myfile.close();
    	    }

        	float start_depth = network_config->layer[1].first_depth_id - 0.1;
        	float end_depth = network_config->layer[1].last_depth_id + 0.1;
    		//cout<<"Changing threshold, start: "<< start_depth<<" end: "<<end_depth<<endl;
    		//change_threshold<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, -68.2);

//
        	start_depth = network_config->layer[3].first_depth_id - 0.1;
        	end_depth = network_config->layer[3].last_depth_id + 0.1;
    		cout<<"Changing threshold, start: "<< start_depth<<" end: "<<end_depth<<endl;
//			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, 5, -5.07);
//			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, 4, 0.453);
//			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, 0, -0.02);
//    		change_threshold<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, -63);
//    		cout<<"Changing param, start: "<< start_depth+32<<" end: "<<end_depth<<endl;
//			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth+128, end_depth, 5, -1.6);
//			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth+128, end_depth, 4, 0.16);
//			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth+128, end_depth, 0, -0.001);

        	start_depth = network_config->layer[4].first_depth_id - 0.1;
        	end_depth = network_config->layer[4].last_depth_id + 0.1;
//    		cout<<"Changing threshold, start: "<< start_depth<<" end: "<<end_depth<<endl;
    		change_threshold<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, -60.0);

    		cout<<"Parameter Changing complete.\n";
    		cudaDeviceSynchronize();
    	}

    	if(time%tenpercent_iter == 0){
    		iter_log = clock();
    		cout<<to_string(10*(time/tenpercent_iter))<<"% done, time used is: " << (iter_log - iter_start)/1000 << " (ms)" << endl;
    	}

//    	if(time%training_time_each_sequence==0){//at the beginning of each img's training, load into
//			int locate_index = myvector[training_img_index];
//    		events_host = new Event_Camera_Input[img_load_max];
//    		IBM_DVS128_event_based(image_file, events_host, img_load_max, img_load_max);
//    	}

    	//cout<<"One IMG loaded"<<endl;
    	//enter spiking neuron simulation:
    	int one_layer_neuron_num = 0;
    	if(time<first_layer_time){
			for(int layer_iter=0;layer_iter<CNN_total_layer_num;layer_iter++){
				one_layer_neuron_num = network_config->layer[layer_iter].neuron_num;
				int convolution_result_index = layer_iter - 1;
				if (layer_iter==0) {//fault at convolution kernel and spiking cnn
					convolution_result_index = 0;
					spiking_cnn_main_event_based(Neuron_list_device, Input_neuronlist_device, events_GPU, event_count, network_config, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, input_float, time, false);
					convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
				}else if(layer_iter==1){
					spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, 0.7*input_float, time, true);
					synapse_drive_cnn_v2(Neuron_list_device, Input_neuronlist_device, network_config, d_network_config, d_filter_array, layer_iter, \
							spiking_neuron_num, input_neuron_num, syn_timer_max, connection_size, random_number_list_device, random_number_normal_device, states, -1.0, -1.0, log_total_spike);//STDP
				}
			}
			//=================TRY WITH LAYER wise inhibition=====================
	    	if(depth_wise_inhibition) {

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
					spiking_cnn_main_event_based(Neuron_list_device, Input_neuronlist_device, events_GPU, event_count, network_config, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, input_float, time, false);
					convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
				}else if(layer_iter==1){
					spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, 0.5*input_float, time, false);
					if (layer_iter!=(CNN_total_layer_num-1)) convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
				}else if(layer_iter==2){
					spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, 2*input_float, time, true);
					synapse_drive_cnn_v2(Neuron_list_device, Input_neuronlist_device, network_config, d_network_config, d_filter_array, layer_iter, spiking_neuron_num, input_neuron_num, \
							syn_timer_max, connection_size, random_number_list_device, random_number_normal_device, states, -1.0, -1.0, log_total_spike);//STDP
				}

			}
			//=================TRY WITH LAYER wise inhibition=====================
	    	if(depth_wise_inhibition) {

	    	}else if(forced_lateral_inhibition_at_last_layer && CNN_total_layer_num==3){//if this is the last layer, use lateral_inhibition
	    		lateral_inhibition_mother_thread<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, 2, inhibition_time, d_network_config, spike_flag_device);
	    	}else if(through_depth_inhibition){

	    	}else if(apply_local_inhibition && CNN_total_layer_num!=3){

	    	}
	    	else{
	    		lateral_inhibition_mother_thread<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, 2, inhibition_time, d_network_config, spike_flag_device);
	    	}
			if(HOMEOSTASIS_ENABLE){
				if(time%HOMEOSTASIS_UPDATE_FREQUENCY == 0 && time != 0){
					//spiking_learning_drive(Neuron_list_device, network_size, inhibition_time, log_total_spike, target_frequency, time, log_spike, 0, 1);
				}
			}

    	}else if(time<third_layer_time){
			for(int layer_iter=0;layer_iter<CNN_total_layer_num;layer_iter++){
				one_layer_neuron_num = network_config->layer[layer_iter].neuron_num;
				int convolution_result_index = layer_iter - 1;
				if (layer_iter==0) {//fault at convolution kernel and spiking cnn
					convolution_result_index = 0;
					spiking_cnn_main_event_based(Neuron_list_device, Input_neuronlist_device, events_GPU, event_count, network_config, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, input_float, time, false);
					convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
				}else if(layer_iter==1){
					spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, 0.5*input_float, time, false);
					if (layer_iter!=(CNN_total_layer_num-1)) convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
				}else if(layer_iter==2){
					spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, 0.5*input_float, time, false);
					if (layer_iter!=(CNN_total_layer_num-1)) convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, \
							h_convolution_result, probe);
				}else if(layer_iter==3){
					spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, 2*input_float, time, true);
					synapse_drive_cnn_v2(Neuron_list_device, Input_neuronlist_device, network_config, d_network_config, d_filter_array, layer_iter, spiking_neuron_num, input_neuron_num, \
							syn_timer_max, connection_size, random_number_list_device, random_number_normal_device, states, -1.0, -1.0, log_total_spike);//STDP
				}

			}
			//=================TRY WITH LAYER wise inhibition=====================
	    	if(depth_wise_inhibition) {

	    	}else if(forced_lateral_inhibition_at_last_layer && CNN_total_layer_num==3){//if this is the last layer, use lateral_inhibition
	    		lateral_inhibition_mother_thread<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, 2, inhibition_time, d_network_config, spike_flag_device);
	    	}else if(through_depth_inhibition){

	    	}else if(apply_local_inhibition && CNN_total_layer_num!=3){

	    	}
	    	else{
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
					spiking_cnn_main_event_based(Neuron_list_device, Input_neuronlist_device, events_GPU, event_count, network_config, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, input_float, time, false);
					convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
				}else if(layer_iter==1 ){
					spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, 3*input_float, time, false);
					if (layer_iter!=(CNN_total_layer_num-1)) convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
				}else if(layer_iter==2){
					spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, 0.5*input_float, time, false);
					if (layer_iter!=(CNN_total_layer_num-1)) convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
				}else if(layer_iter==3){
					bool last_layer_inhib = !last_layer_teach;
					spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, 4*input_float, time, false);
					if (layer_iter!=(CNN_total_layer_num-1)) convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
					//synapse_drive_cnn_v2(Neuron_list_device, Input_neuronlist_device, network_config, d_network_config, d_filter_array, layer_iter, spiking_neuron_num, input_neuron_num, syn_timer_max, connection_size, random_number_list_device, random_number_normal_device, states, -1.0, -1.0, log_total_spike);//STDP
				}else if(layer_iter==4){
					bool last_layer_inhib = !last_layer_teach;
					spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, 15*input_float, time, true);
					synapse_drive_cnn_v2(Neuron_list_device, Input_neuronlist_device, network_config, d_network_config, d_filter_array, layer_iter, spiking_neuron_num, input_neuron_num, syn_timer_max, connection_size, random_number_list_device, random_number_normal_device, states, -1.0, -1.0, log_total_spike);//STDP
				}

			}
			//=================TRY WITH LAYER wise inhibition=====================
	    	if(depth_wise_inhibition) {

	    	}else if(forced_lateral_inhibition_at_last_layer && CNN_total_layer_num==4){//if this is the last layer, use lateral_inhibition
	    		lateral_inhibition_mother_thread<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, 2, inhibition_time, d_network_config, spike_flag_device);
	    	}else if(through_depth_inhibition){

	    	}else if(apply_local_inhibition && CNN_total_layer_num!=3){

	    	}
	    	else{
	    		lateral_inhibition_mother_thread<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, 2, inhibition_time, d_network_config, spike_flag_device);
	    	}
			if(HOMEOSTASIS_ENABLE){
				if(time%HOMEOSTASIS_UPDATE_FREQUENCY == 0 && time != 0){
					//spiking_learning_drive(Neuron_list_device, network_size, inhibition_time, log_total_spike, target_frequency, time, log_spike, 0, 1);
				}
			}

    	}
//    	printf("T: %d_",time);
//    	if(time==100) 		break;
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
	filter_util(network_config, NeuronList, network_size, input_neuron_num, h_filter_array, d_filter_array, index_prefix, 1);	//write filter to file
    write_neuron_list(NeuronList, (index_prefix+"device2_output_network.txt"), spiking_neuron_num);
    data_check(NeuronList,log_total_spike,SIZE, mnist_start_index, mnist_end_index, 2, "");
    //===clean up===
    //delete[] random_number_list;
    delete[] log_v_host;
	delete[] NeuronList;
	delete[] log_spike_host;
	delete[] log_total_spike_host;
	delete[] events_host;
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



void run_event_based_learning_hsnn(string index_prefix, float input_float, float input_float_2, int input_int, int input_int_2, string input_img, int resume_learning, int start_layer){
	int depth_list[3];
	if (start_layer==1){
		depth_list[0]=16; depth_list[1]=32; depth_list[2]=32;
	}
	else if (start_layer==2){
		depth_list[0]=32; depth_list[1]=32; depth_list[2]=32;
	}
	else if (start_layer==3){
		depth_list[0]=32; depth_list[1]=64; depth_list[2]=32;
	}
	//cout<<depth_list[0]<<" "<<depth_list[1]<<" "<<depth_list[2]<<endl;
	CNN_struct *network_config = new CNN_struct;
	hsnn_config_generator(depth_list, network_config);
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


//===========END of CNN special setting-up phase============

	Convolution_setting_struct *convolution_settings = new Convolution_setting_struct[CNN_total_layer_num];

	//set parameters
	int img_load_max = 100000000;
	int time_per_event = input_int;
	long calculated_total_time = time_per_event*100*10000;

	#undef MAX_TIME
	#define MAX_TIME calculated_total_time
	int tenpercent_iter = MAX_TIME/10;
	printf("==Training Total Iter: %d==", MAX_TIME);
	int total_neuron_num = 0;
	int total_spiking_num = 0;
	for(int i=0;i<CNN_total_layer_num;i++){
		total_neuron_num += network_config->layer[i].neuron_num;
		if(i!=0) total_spiking_num += network_config->layer[i].neuron_num;
	}
	total_spiking_num += 5;
	total_neuron_num += 5;
	//total_neuron_num = 20000;
	cout<<endl<<"total neuron num: "<<total_neuron_num<<endl;
	cout<<"total spiking neuron num: "<<total_spiking_num<<endl;
	#undef SIZE
	#define SIZE total_neuron_num
	#undef SPIKING_NEURON_NUM
	#define SPIKING_NEURON_NUM total_spiking_num


//	float max_frequency = 50; //in Hz default 22
//	float min_frequency = 10;
	int training_set_number = 32000;
	bool batch_load = false;
	int batched_load_remain = 0;
	int batch_load_grand_total = 0;
	int img_load_offset = 0;


	if (training_set_number>img_load_max){ //manually set the maximum number of images to be loaded once is 60000
		cout<<"Using batch loading"<<endl;
		batch_load_grand_total = training_set_number;
		batch_load = true;
		batched_load_remain = training_set_number - img_load_max;
		training_set_number = img_load_max;
	}
	int input_neuron_num = input_image_w*input_image_l*input_image_channel;
	int input_image_signal_channel_size  = input_image_w*input_image_l;
	int spiking_neuron_num = SPIKING_NEURON_NUM;
	int output_layer_neuron_num = OUTPUT_LAYER_NEURON_NUM;

	int connection_size = MAX_CONNECTION;
	int syn_timer_max = 25;
	int input_signal_width = 10;	//default 25
	int inhibition_time = 10;	//default 10

	float target_frequency_param = 0.5;
	float target_frequency = 100;

	Event_Camera_Input *events_host = new Event_Camera_Input[img_load_max];
	Event_Camera_Input *events_GPU;
	cudaMalloc((void **)&events_GPU,img_load_max*sizeof(Event_Camera_Input));

	int current_input_file_id = 1;
	int input_file_id_max = 10;
	string image_file = "";
	if (current_input_file_id<10) {
		image_file = "/home/data/DVS128_event_based/user0" + to_string(current_input_file_id) + "_event_based.csv";
	}
	else{
		image_file = "/home/data/DVS128_event_based/user" + to_string(current_input_file_id) + "_event_based.csv";
	}
	//string image_file = "/hdd2/extra_home/xshe6/Event_camera/event_based/user01_event_based.csv";//"dvs_gesture_event_based_test.csv";

	cout<<endl<<"Image loading"<<endl;
	clock_t load_start = clock();
    std::vector<std::string> folder_list;
    int input_folder_cnt = 0;
    int current_total_read_event = 0;
	if(input_image_channel==1 || input_image_channel==2){


		current_total_read_event = IBM_DVS128_event_based(image_file, events_host, img_load_max, img_load_max);
	    cudaMemcpy(events_GPU,events_host,img_load_max*sizeof(Event_Camera_Input),cudaMemcpyHostToDevice);

	}else{
		printf("Input channel error.");
		return;

	}
	clock_t load_end = clock();
	cout<<endl<<"Image loading done"<<", time used is " << (load_end - load_start)/1000 << " (ms)"<<endl;
	//CIFAR_read_image_one_channel(events_host, input_image_signal_channel_size, input_int_2, 0);
	//MNIST_read_image(image_file, events_host, training_set_number);
	int *mnist_label = new int[training_set_number];
	string image_label_file = "train-labels-idx1-ubyte";
	//CIFAR_read_label(mnist_label, 0);
	//MNIST_read_label(image_label_file, mnist_label, training_set_number);
	//special_function: learn one category
	printf("=0=\n");
	int learn_one_digit = 0;
	int *num_one_digit_img = new int[1];


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
		printf("------RESUME LEARNING-------\n");
		read_neuron_list(NeuronList, 1, "spike_cnn.txt");
		//read_neuron_list(NeuronList, 1, "device2_output_network.txt");
		//int start_layer = 3;//the layer that starts to learn
		read_neuron_list_special(NeuronList, (start_layer-1), network_config, to_string(start_layer-1)+"device2_output_network.txt"); //duplicate the previous layer
		bool do_reset_weight = true;
		if(do_reset_weight){

			float start_depth = network_config->layer[start_layer].first_depth_id - 0.1;
			float end_depth = network_config->layer[start_layer].last_depth_id + 0.1;

			reset_weight(NeuronList, start_depth, end_depth, 1, spiking_neuron_num);

		}
		//read_neuron_list(NeuronList, 1, "spike_cnn.txt");
	}else{
		read_neuron_list(NeuronList, 1, "spike_cnn.txt");
		//read_neuron_list(NeuronList, 1, "device2_output_network.txt");

	}
	//printf("read out one neuron depth: %f", NeuronList[116000].param[7]);
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
    if(STOCHASTIC_STDP || STOCHASTIC_ROUNDING || DEVICE_VARIATION){
    	cudaMalloc((void **)&states, rand_numb_size * sizeof(curandState_t));
		cudaMalloc((void **)&random_number_list_device,rand_numb_size*sizeof(float));
		curandCreateGenerator(&gen_uniform, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(gen_uniform, time(0));
		curandGenerateUniform(gen_uniform, random_number_list_device, rand_numb_size);
    }


    curandGenerator_t gen_normal;
    float *random_number_normal_device;
    float normal_mean = 0;
    float normal_sd = 5.0;
    if(STOCHASTIC_STDP || STOCHASTIC_ROUNDING || DEVICE_VARIATION){
		cudaMalloc((void **)&random_number_normal_device,rand_numb_size*sizeof(float));
		curandCreateGenerator(&gen_normal, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(gen_normal, time(0));
		curandGenerateNormal(gen_normal, random_number_normal_device, rand_numb_size, normal_mean, normal_sd);
    }

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
    printf("3.0\n");
    //cout<<"network size: "<<SIZE<<endl;
    int network_size = SIZE;

    int max_time = MAX_TIME;
    int first_layer_time = max_time;
	int second_layer_time = max_time+1;
	int third_layer_time = max_time+1;
    if(resume_learning){
        if(CNN_total_layer_num==3) second_layer_time = max_time;
        if(CNN_total_layer_num==4){
        	if (start_layer==2){
				first_layer_time = 1;
				second_layer_time = max_time+1;
				third_layer_time = max_time+1;
        	}
        	else if (start_layer==3){
				first_layer_time = 1;
				second_layer_time = 2;
				third_layer_time = max_time+1;
        	}
        }
    }

    //cudaMemcpy(Neuron_list_device,old_device_neurons,sizeof(Neuron)*SIZE,cudaMemcpyDeviceToDevice);
    //first change raw img data into frequency
    int mnist_start_index = 0;
    int mnist_end_index = input_neuron_num;
    //change pixel signal to frequency


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
	bool last_layer_teach = false;

	int total_class = 10;
	int num_per_class = 500;
	int frame_per_seq = 50;

	int rotation_num = 2;
	int translation_num = 1;
	int frame_per_class = frame_per_seq*rotation_num*num_per_class;
	int frame_per_rotation = frame_per_seq;
	int frame_per_translation = frame_per_seq*rotation_num*num_per_class*total_class;

    std::vector<int> seq_vector_head;
    std::vector<int> seq_vector;
    for (int i=0; i<training_set_number/frame_per_seq; ++i) seq_vector_head.push_back(i); // 1 2 3 4 5 6 7 8 9
	std::random_shuffle ( seq_vector_head.begin(), seq_vector_head.end() );

	for (int i=0 ; i<training_set_number/frame_per_seq; ++i){
		int begin_index = seq_vector_head[i];
		for (int j=0; j<frame_per_seq; ++j) seq_vector.push_back(10*begin_index+j);

	}

    //read_filter_GPU_one_layer<<<1, 1>>>(d_network_config, h_filter_array[0], 1);
    //read_filter_GPU<<<1, 1>>>(d_network_config, d_filter_array);

//    int reiter_run = 1;

    int time = 0;
    int training_img_index = 0;

    //============now load all convolution settings===========
	for(int layer_iter=0;layer_iter<CNN_total_layer_num;layer_iter++){
		if (layer_iter==0) {
			cout<<endl<<"Setting up the first layer"<<endl;
			convolution_kernel_setup(convolution_settings, network_config, layer_iter);
		}else{

			if (layer_iter!=(CNN_total_layer_num-1))
			{
				convolution_kernel_setup(convolution_settings, network_config, layer_iter);
				cout<<"Setting up the"<<" layer "<< layer_iter <<endl;
			}
		}
	}
    if(resume_learning){
		copy_filter_to_cuDNN(Neuron_list_device, d_network_config, d_filter_array, spiking_neuron_num);
		cudaDeviceSynchronize();
    }
	float start_depth = network_config->layer[1].first_depth_id - 0.1;
	float end_depth = network_config->layer[1].last_depth_id + 0.1;
	cout<<"Changing threshold, start: "<< start_depth<<" end: "<<end_depth<<endl;
	change_threshold<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, -62.2);
	//return;
	int event_count = 0;
	while (time<max_time){

		//if(time==first_layer_time)MNIST_drive(NeuronList, Input_neuronlist, events_host, network_size, training_set_number, mnist_start_index, mnist_end_index, max_frequency*2, min_frequency, 1);
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

    	if(time%time_per_event){
    		event_count++;
    		while(events_host[event_count].valid==False && event_count<current_total_read_event){
    			event_count++;
    		}

    		if (event_count>=current_total_read_event){
    			cout<<endl<<"Image loading"<<endl;
    			current_input_file_id ++;

    			if(current_input_file_id>input_file_id_max) current_input_file_id = 1;

    			if (current_input_file_id<10) {
    				image_file = "/hdd2/extra_home/xshe6/Event_camera/event_based/user0" + to_string(current_input_file_id) + "_event_based.csv";
    			}
    			else{
    				image_file = "/hdd2/extra_home/xshe6/Event_camera/event_based/user" + to_string(current_input_file_id) + "_event_based.csv";
    			}

    		    current_total_read_event = 0;
				current_total_read_event = IBM_DVS128_event_based(image_file, events_host, img_load_max, img_load_max);
				cout<<"Total loaded:"<< current_total_read_event<<endl;
				cudaMemcpy(events_GPU,events_host,img_load_max*sizeof(Event_Camera_Input),cudaMemcpyHostToDevice);
    			event_count=0;
    		}
    	}

    	if(time==first_layer_time){


    	    gpuErrchk( cudaMemcpy(log_total_spike_host,log_total_spike,SIZE*sizeof(float),cudaMemcpyDeviceToHost) );
    	    ofstream myfile ((index_prefix+"first_stage_device2_spike_of_neuron_out.csv"));
    	    if (myfile.is_open()){
    	    	//myfile << "This is a new test\n";
    	    	for(int i=0; i < SIZE; i++){
    	    		//printf("_%f_", log_v_host[i]);
    	    		myfile << log_total_spike_host[i] << ", ";
    	    	}
    	    	myfile.close();
    	    }

        	float start_depth = network_config->layer[1].first_depth_id - 0.1;
        	float end_depth = network_config->layer[1].last_depth_id + 0.1;
    		//cout<<"Changing threshold, start: "<< start_depth<<" end: "<<end_depth<<endl;
			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, 5, -2.07);
			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, 4, 0.453);
			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, 0, 0.02);
    		change_threshold<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, -65.2);
    		cout<<"Changing param of long-term neuron, start: "<< start_depth+16<<" end: "<<end_depth<<endl;
			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth+depth_list[0]/2, end_depth, 5, -1.6);
			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth+depth_list[0]/2, end_depth, 4, 0.4);
			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth+depth_list[0]/2, end_depth, 0, 0.001);
    		change_threshold<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth+depth_list[0]/2, end_depth, -64.2);


    		//training_time_each_img = training_time_each_img*1.3;
    		cudaDeviceSynchronize();

    	}else if(time==second_layer_time){

    	    gpuErrchk( cudaMemcpy(log_total_spike_host,log_total_spike,SIZE*sizeof(float),cudaMemcpyDeviceToHost) );
    	    ofstream myfile ((index_prefix+"second_stage_device2_spike_of_neuron_out.csv"));
    	    if (myfile.is_open()){
    	    	//myfile << "This is a new test\n";
    	    	for(int i=0; i < SIZE; i++){
    	    		//printf("_%f_", log_v_host[i]);
    	    		myfile << log_total_spike_host[i] << ", ";
    	    	}
    	    	myfile.close();
    	    }

        	float start_depth = network_config->layer[1].first_depth_id - 0.1;
        	float end_depth = network_config->layer[1].last_depth_id + 0.1;
    		//cout<<"Changing threshold, start: "<< start_depth<<" end: "<<end_depth<<endl;
    		//change_threshold<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, -64);


        	start_depth = network_config->layer[2].first_depth_id - 0.1;
        	end_depth = network_config->layer[2].last_depth_id + 0.1;
    		//cout<<"Changing threshold, start: "<< start_depth<<" end: "<<end_depth<<endl;
			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, 5, -2.07);
			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, 4, 0.453);
			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, 0, 0.02);
    		change_threshold<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, -65.2);
    		cout<<"Changing param of long-term neuron, start: "<< start_depth+16<<" end: "<<end_depth<<endl;
			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth+depth_list[1]/2, end_depth, 5, -1.6);
			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth+depth_list[1]/2, end_depth, 4, 0.4);
			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth+depth_list[1]/2, end_depth, 0, 0.001);
    		change_threshold<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth+depth_list[1]/2, end_depth, -64.2);
//        	start_depth = network_config->layer[3].first_depth_id - 0.1;
//        	end_depth = network_config->layer[3].last_depth_id + 0.1;
//    		cout<<"Changing threshold, start: "<< start_depth<<" end: "<<end_depth<<endl;
//    		change_threshold<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, -60.0);

    	}else if(time==third_layer_time){

    	    gpuErrchk( cudaMemcpy(log_total_spike_host,log_total_spike,SIZE*sizeof(float),cudaMemcpyDeviceToHost) );
    	    ofstream myfile ((index_prefix+"third_stage_device2_spike_of_neuron_out.csv"));
    	    if (myfile.is_open()){
    	    	//myfile << "This is a new test\n";
    	    	for(int i=0; i < SIZE; i++){
    	    		//printf("_%f_", log_v_host[i]);
    	    		myfile << log_total_spike_host[i] << ", ";
    	    	}
    	    	myfile.close();
    	    }

        	float start_depth = network_config->layer[1].first_depth_id - 0.1;
        	float end_depth = network_config->layer[1].last_depth_id + 0.1;
    		//cout<<"Changing threshold, start: "<< start_depth<<" end: "<<end_depth<<endl;
    		//change_threshold<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, -68.2);

//
        	start_depth = network_config->layer[3].first_depth_id - 0.1;
        	end_depth = network_config->layer[3].last_depth_id + 0.1;
    		cout<<"Changing threshold, start: "<< start_depth<<" end: "<<end_depth<<endl;
//			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, 5, -5.07);
//			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, 4, 0.453);
//			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, 0, -0.02);
//    		change_threshold<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, -63);
//    		cout<<"Changing param, start: "<< start_depth+32<<" end: "<<end_depth<<endl;
//			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth+128, end_depth, 5, -1.6);
//			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth+128, end_depth, 4, 0.16);
//			update_param<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth+128, end_depth, 0, -0.001);

        	start_depth = network_config->layer[4].first_depth_id - 0.1;
        	end_depth = network_config->layer[4].last_depth_id + 0.1;
//    		cout<<"Changing threshold, start: "<< start_depth<<" end: "<<end_depth<<endl;
    		change_threshold<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, start_depth, end_depth, -60.0);

    		cout<<"Parameter Changing complete.\n";
    		cudaDeviceSynchronize();
    	}

    	if(time%tenpercent_iter == 0){
    		iter_log = clock();
    		cout<<to_string(10*(time/tenpercent_iter))<<"% done, time used is: " << (iter_log - iter_start)/1000 << " (ms)" << endl;
    	}

//    	if(time%training_time_each_sequence==0){//at the beginning of each img's training, load into
//			int locate_index = myvector[training_img_index];
//    		events_host = new Event_Camera_Input[img_load_max];
//    		IBM_DVS128_event_based(image_file, events_host, img_load_max, img_load_max);
//    	}

    	//cout<<"One IMG loaded"<<endl;
    	//enter spiking neuron simulation:
    	int one_layer_neuron_num = 0;
    	if(time<first_layer_time){
			for(int layer_iter=0;layer_iter<CNN_total_layer_num;layer_iter++){
				one_layer_neuron_num = network_config->layer[layer_iter].neuron_num;
				int convolution_result_index = layer_iter - 1;
				if (layer_iter==0) {//fault at convolution kernel and spiking cnn
					convolution_result_index = 0;
					spiking_cnn_main_event_based(Neuron_list_device, Input_neuronlist_device, events_GPU, event_count, network_config, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, input_float, time, false);
					convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
				}else if(layer_iter==1){
					spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, 0.5*input_float, time, true);
					synapse_drive_cnn_v2(Neuron_list_device, Input_neuronlist_device, network_config, d_network_config, d_filter_array, layer_iter, \
							spiking_neuron_num, input_neuron_num, syn_timer_max, connection_size, random_number_list_device, random_number_normal_device, states, -1.0, -1.0, log_total_spike);//STDP
				}
			}

	    	if(depth_wise_inhibition) {
			//implemented in spiking_cnn_main_event_based
	    	}else if(through_depth_inhibition){
			//implemented in spiking_cnn_main_event_based
	    	}else if(apply_local_inhibition){
			//implemented in spiking_cnn_main_event_based
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
					spiking_cnn_main_event_based(Neuron_list_device, Input_neuronlist_device, events_GPU, event_count, network_config, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, input_float, time, false);
					convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
				}else if(layer_iter==1){
					spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, 0.5*input_float, time, false);
					if (layer_iter!=(CNN_total_layer_num-1)) convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
				}else if(layer_iter==2){
					spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, input_float, time, true);
					synapse_drive_cnn_v2(Neuron_list_device, Input_neuronlist_device, network_config, d_network_config, d_filter_array, layer_iter, spiking_neuron_num, input_neuron_num, \
							syn_timer_max, connection_size, random_number_list_device, random_number_normal_device, states, -1.0, -1.0, log_total_spike);//STDP
				}

			}

	    	if(depth_wise_inhibition) {

	    	}else if(forced_lateral_inhibition_at_last_layer && CNN_total_layer_num==3){//if this is the last layer, use lateral_inhibition
	    		lateral_inhibition_mother_thread<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, 2, inhibition_time, d_network_config, spike_flag_device);
	    	}else if(through_depth_inhibition){

	    	}else if(apply_local_inhibition && CNN_total_layer_num!=3){

	    	}
	    	else{
	    		lateral_inhibition_mother_thread<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, 2, inhibition_time, d_network_config, spike_flag_device);
	    	}
			if(HOMEOSTASIS_ENABLE){
				if(time%HOMEOSTASIS_UPDATE_FREQUENCY == 0 && time != 0){
					//spiking_learning_drive(Neuron_list_device, network_size, inhibition_time, log_total_spike, target_frequency, time, log_spike, 0, 1);
				}
			}

    	}else if(time<third_layer_time){
			for(int layer_iter=0;layer_iter<CNN_total_layer_num;layer_iter++){
				one_layer_neuron_num = network_config->layer[layer_iter].neuron_num;
				int convolution_result_index = layer_iter - 1;
				if (layer_iter==0) {//fault at convolution kernel and spiking cnn
					convolution_result_index = 0;
					spiking_cnn_main_event_based(Neuron_list_device, Input_neuronlist_device, events_GPU, event_count, network_config, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, input_float, time, false);
					convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
				}else if(layer_iter==1){
					spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, 0.5*input_float, time, false);
					if (layer_iter!=(CNN_total_layer_num-1)) convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
				}else if(layer_iter==2){
					spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, 0.5*input_float, time, false);
					if (layer_iter!=(CNN_total_layer_num-1)) convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, \
							h_convolution_result, probe);
				}else if(layer_iter==3){
					spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, 2*input_float, time, true);
					synapse_drive_cnn_v2(Neuron_list_device, Input_neuronlist_device, network_config, d_network_config, d_filter_array, layer_iter, spiking_neuron_num, input_neuron_num, \
							syn_timer_max, connection_size, random_number_list_device, random_number_normal_device, states, -1.0, -1.0, log_total_spike);//STDP
				}

			}
			//=================TRY WITH LAYER wise inhibition=====================
	    	if(depth_wise_inhibition) {

	    	}else if(forced_lateral_inhibition_at_last_layer && CNN_total_layer_num==3){//if this is the last layer, use lateral_inhibition
	    		lateral_inhibition_mother_thread<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, 2, inhibition_time, d_network_config, spike_flag_device);
	    	}else if(through_depth_inhibition){

	    	}else if(apply_local_inhibition && CNN_total_layer_num!=3){

	    	}
	    	else{
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
					spiking_cnn_main_event_based(Neuron_list_device, Input_neuronlist_device, events_GPU, event_count, network_config, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, input_float, time, false);
					convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
				}else if(layer_iter==1 ){
					spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, 3*input_float, time, false);
					if (layer_iter!=(CNN_total_layer_num-1)) convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
				}else if(layer_iter==2){
					spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, 0.5*input_float, time, false);
					if (layer_iter!=(CNN_total_layer_num-1)) convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
				}else if(layer_iter==3){
					bool last_layer_inhib = !last_layer_teach;
					spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, 4*input_float, time, false);
					if (layer_iter!=(CNN_total_layer_num-1)) convolution_kernel(convolution_settings[layer_iter], layer_iter, h_input_instance, h_filter_array, h_convolution_result, probe);
					//synapse_drive_cnn_v2(Neuron_list_device, Input_neuronlist_device, network_config, d_network_config, d_filter_array, layer_iter, spiking_neuron_num, input_neuron_num, syn_timer_max, connection_size, random_number_list_device, random_number_normal_device, states, -1.0, -1.0, log_total_spike);//STDP
				}else if(layer_iter==4){
					bool last_layer_inhib = !last_layer_teach;
					spiking_cnn_main(Neuron_list_device, Input_neuronlist_device, d_network_config, random_number_list_device, d_convolution_result, d_input_instance, \
							layer_iter, network_size, input_neuron_num, log_v, log_spike, log_total_spike, spike_flag_device, input_signal_width, 15*input_float, time, true);
					synapse_drive_cnn_v2(Neuron_list_device, Input_neuronlist_device, network_config, d_network_config, d_filter_array, layer_iter, spiking_neuron_num, input_neuron_num, syn_timer_max, connection_size, random_number_list_device, random_number_normal_device, states, -1.0, -1.0, log_total_spike);//STDP
				}

			}
			//=================TRY WITH LAYER wise inhibition=====================
	    	if(depth_wise_inhibition) {

	    	}else if(forced_lateral_inhibition_at_last_layer && CNN_total_layer_num==4){//if this is the last layer, use lateral_inhibition
	    		lateral_inhibition_mother_thread<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, 2, inhibition_time, d_network_config, spike_flag_device);
	    	}else if(through_depth_inhibition){

	    	}else if(apply_local_inhibition && CNN_total_layer_num!=3){

	    	}
	    	else{
	    		lateral_inhibition_mother_thread<<<dimBlock_unit, dimGrid_unit>>>(Neuron_list_device, spiking_neuron_num, 2, inhibition_time, d_network_config, spike_flag_device);
	    	}
			if(HOMEOSTASIS_ENABLE){
				if(time%HOMEOSTASIS_UPDATE_FREQUENCY == 0 && time != 0){
					//spiking_learning_drive(Neuron_list_device, network_size, inhibition_time, log_total_spike, target_frequency, time, log_spike, 0, 1);
				}
			}

    	}
//    	printf("T: %d_",time);
//    	if(time==100) 		break;
    	time ++;
    	//break;
    }
    //spiking_learning_drive(Neuron_list_device, network_size, inhibition_time, 2);
	//cudaDeviceSynchronize();

	filter_util(network_config, Neuron_list_device, network_size, input_neuron_num, h_filter_array, d_filter_array, index_prefix, 2);
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
	filter_util(network_config, NeuronList, network_size, input_neuron_num, h_filter_array, d_filter_array, index_prefix, 1);	//write filter to file
    write_neuron_list(NeuronList, (index_prefix+"device2_output_network.txt"), spiking_neuron_num);
    data_check(NeuronList,log_total_spike,SIZE, mnist_start_index, mnist_end_index, 2, "");
    //===clean up===
    //delete[] random_number_list;
    delete[] log_v_host;
	delete[] NeuronList;
	delete[] log_spike_host;
	delete[] log_total_spike_host;
	delete[] events_host;
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


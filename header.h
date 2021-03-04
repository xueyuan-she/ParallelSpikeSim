/*
 * header.h
 *
 *  Created on: Nov 29, 2017
 *      Author: DanShe
 */

#ifndef HEADER_H_
#define HEADER_H_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#include <vector>
#include <string>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <curand.h>
#include <curand_kernel.h>
#include "CImg.h"
#include <cudnn.h>


#define img_width 64	//for ROI
#define img_len 64
#define input_image_w 128	//for learning           for sc2 learning make 4x4;  for dvs is 128
#define input_image_l 128
#define input_image_channel 2  //for learning           for sc2 learning make 13
#define non_random_weight_init 0
#define shuffle_image 0
#define MAX_CONNECTION 3300
#define MAX_LOCAL_INHIBITION 20
#define CNN_total_layer_num 4//this includes input layer
#define MAX_depth_in_one_layer 256  //this includes input layer
//MODE Select
#define depth_wise_inhibition 0
#define through_depth_inhibition true
#define apply_local_inhibition false
#define forced_lateral_inhibition_at_last_layer false

#define LOW_BIT_TRAINING 0
#define STOCHASTIC_STDP 0
#define EXPONENTIAL_STDP 1
#define STOCHASTIC_ROUNDING 0
#define HOMEOSTASIS_ENABLE 0
#define HOMEOSTASIS_BASE_RATE 0.004
#define LEARNER_HOMEOSTASIS_ENABLE 0
#define SPIKE_FREQ_SAMPLING_INTV 10000
#define LAYERWISE_SHARED_WEIGHT 1
#define FREQUENCY_DEPENDED_STDP 0
#define ThreadsPerBlock 8
#define SPIKING_NEURON_NUM 1000
#define OUTPUT_LAYER_NEURON_NUM 1000
#define MID_LAYER_STDP_DURATION 25
#define HOMEOSTASIS_UPDATE_FREQUENCY 50000
#define LOW_BIT_NUM 8
#define LOW_BIT_MEM_POT 0
#define DEVICE_VARIATION 0
#define TWO_POWER_2 4
#define TWO_POWER_4 16
#define TWO_POWER_8 256
#define TWO_POWER_16 65536
#define TWO_POWER_32 4294967296

using namespace std;


typedef struct {
	signed int index;	//start with 1
	signed int type;	//0: IZH, 1: Stochastic, 2: LIF, 3: HH
	float spike_frequency;
	float spike_cnt;
	float param[8]; //Izh has 5 parameters, a b c d threshold; LIF has, 1 is threshold, 2 is reset

	float state[8]; //Izh has 3 states, 0 is membrane potential, 1 is V Flag

	//change connection size, remember to change MACRO in main.cu
	unsigned int connected_in[MAX_CONNECTION];	//ROI: 3
	float connected_weight[MAX_CONNECTION];
	signed char synapse_timer[MAX_CONNECTION];	//used in STPD learning


	unsigned int local_inhibition[MAX_LOCAL_INHIBITION];
} Neuron;

typedef struct {
	signed int index;	//start with 1
	signed int type;	//0: IZH, 1: Stochastic, 2: LIF, 3: HH
	float spike_frequency;
	float spike_cnt;
	float param[8]; //Izh has 5 parameters, a b c d threshold; LIF has

	float state[8]; //Izh has 3 states, U V Flag

	unsigned int connected_in[1];	//ROI: 3
	float connected_weight[1];
	signed char synapse_timer[1];	//used in STPD learning

} Input_neuron;

typedef struct {
	 int pad_height;
	 int pad_width;
	 int vertical_stride;
	 int horizontal_stride;
	 int dilation_height;
	 int dilation_width;
	 int filter_width;
	 int filter_length;
	 int filter_depth;
} convolution_param;

typedef struct {
	int id;
	int total_neuron_num;
	int width;
	int length;
	int first_neuron;
	int last_neuron;

	float param[8];
	float state[8];
} depth_struct;

typedef struct {
	int layer_id;	//start with zero
	int first_depth_id;
	int last_depth_id;
	int width;
	int length;
	int depth;		//total depth in this layer
	int neuron_num;
	int input_layer;
	float *filter;
	depth_struct depth_list[MAX_depth_in_one_layer];

	convolution_param conv_setting;
} CNN_layer;

typedef struct {
	CNN_layer layer[CNN_total_layer_num];
} CNN_struct;

typedef struct {
	cudnnHandle_t cudnn;
	cudnnTensorDescriptor_t input_descriptor;
	cudnnFilterDescriptor_t kernel_descriptor;
	cudnnConvolutionDescriptor_t convolution_descriptor;
	cudnnConvolutionFwdAlgo_t convolution_algorithm;
	cudnnTensorDescriptor_t output_descriptor;
	size_t workspace_bytes{0};
	void* d_workspace{nullptr};
} Convolution_setting_struct;

typedef struct {
	int loc_x;
	int loc_y;
	unsigned long time;
	int sign;
	bool valid;
} Event_Camera_Input;

//void kernel_neuron(Neuron *NeuronList, Neuron *old_device_neurons, float *random_number, int network_size);
//void kernel_Stochastic(Neuron *NeuronList, float *random_number, int network_size);
int read_neuron_list(Neuron *NeuronList, int neuron_model, string file_name);
//void neuron_test(Neuron *NeuronList, Neuron *old_device_neurons, float *random_number, int network_size, float *log_v, float *log_spike, float *log_total_spike, int time_stamp);
//void test_2(Neuron *NeuronList, Neuron *old_device_neurons, float *random_number, int network_size, float *log_v, int time_stamp);
void ROI_drive(Neuron *NeuronList, float *image_signal, int network_size, int start_index, int end_index, int function_select);
void spiking_learning_drive(Neuron *NeuronList, int network_size, int inhibit_time, float *log_total_spike, float target_frequency, int time, float *log_spike, int current_layer, CNN_struct *CNN_setttings, int function_select);
void spiking_learning_drive(Neuron *NeuronList, int network_size, int inhibit_time, float *log_total_spike, float target_frequency, int time, float *log_spike, int current_layer, int function_select);
//void synapse_drive_v1(Neuron *NeuronList, int network_size, int syn_timer_max, int connection_size, float *random_number, float StochSTDP_param_1, float StochSTDP_param_2);
void MNIST_drive(Neuron *NeuronList, Input_neuron *Input_neuronlist, float *image, int network_size, int training_set_number, int start, int end, float max_frequency, float min_frequency, int function_select);
void MNIST_drive(Neuron *NeuronList, Input_neuron *Input_neuronlist, float *MNIST_stimulus_freq, int network_size, int training_set_number, int start, int end, float max_frequency, float min_frequency, int function_select, int target);
void spiking_learning_main(Neuron *NeuronList, Neuron *old_device_neurons, float *random_number, int network_size, float *log_v, float *log_spike, float *log_total_spike, int *spike_flag, int signal_width, int time_stamp);
int write_neuron_list(Neuron *NeuronList, string file_name, int network_size);
void data_check(Neuron *NeuronList, float *log_total_spike, int network_size, int mnist_start_index, int mnist_end_index, int function_select, string plot_prefix);
void MNIST_labeling(string input_file_starter, int size, float *input_array_1, int *input_array_2, float *output_array_1, int *output_array_2, int main_neuron_num, int function_select, int function_select_2);
void MNIST_labeling_2(Neuron *NeuronList, float *img_raw, float *output_v, int output_neuron_size);
int convolution_kernel(Convolution_setting_struct convolution_settings, int layer_index, float **d_input, float **filter, float **output, float *probe);
void img_util(float *img_data, string file_name, int function_select);
int network_config_generator(int function_select, CNN_struct *settings);
int hsnn_config_generator(int* depth_list, CNN_struct *settings);
//void synapse_drive_cnn(Neuron *NeuronList, CNN_struct *host_CNN_settings, CNN_struct *CNN_settings, float **filter, int current_layer, int network_size, int syn_timer_max, int connection_size, float *random_number, float StochSTDP_param_1, float StochSTDP_param_2);
void synapse_drive_cnn_v2(Neuron *NeuronList, Input_neuron *Input_neuronlist, CNN_struct *host_CNN_settings, CNN_struct *CNN_settings, float **filter, int current_layer, int network_size, int input_neuron_size, int syn_timer_max, int connection_size, float *random_number, float *random_number_normal_device, curandState_t *state, float StochSTDP_param_1, float StochSTDP_param_2, float *log_total_spike);
int filter_util(CNN_struct *settings, Neuron *NeuronList, int network_size, int input_neuron_size, float **host_filter_array, float **device_filter_array, string plot_prefix, int function_select);
int CNN_util(CNN_struct *settings, float **d_instance_matrix_array, float **d_convolution_result_array, float **h_instance_matrix_array, float **h_convolution_result_array, int function_select);
void spiking_cnn_main(Neuron *NeuronList, Input_neuron *Input_neuronlist, CNN_struct *CNN_setttings, float *random_number, float **input, float **instance_matrix, int current_layer, int network_size, int input_size, float *log_v, float *log_spike, float *log_total_spike, int *spike_flag, int signal_width, float input_float, int time_stamp, bool enable_inhibition);
void spiking_cnn_main_event_based (Neuron *NeuronList, Input_neuron *Input_neuronlist, Event_Camera_Input *events, int event_cnt, CNN_struct *host_CNN_setttings, CNN_struct *CNN_setttings, float *random_number, float **input, float **instance_matrix, int current_layer, int network_size, int input_size, float *log_v, float *log_spike, float *log_total_spike, int *spike_flag, int signal_width, float input_float, int time_stamp, bool enable_inhibition);
int convolution_kernel_setup(Convolution_setting_struct *convolution_settings, CNN_struct *settings, int layer_index);
void spiking_cnn_main(Neuron *NeuronList, Input_neuron *Input_neuronlist, CNN_struct *CNN_setttings, float *random_number, float **input, float **instance_matrix, int current_layer, int network_size, int input_size, float *log_v, float *log_spike, float *log_total_spike, int *spike_flag, int signal_width, float input_float, int time_stamp, int optional_inp, bool teaching_mode);
int normalize_weight(Neuron *NeuronList, float start_depth, float end_depth, int norm_method, int network_size);
int reset_weight(Neuron *NeuronList, float start_depth, float end_depth, int reset_method, int network_size);
int read_neuron_list_special(Neuron *NeuronList, int duplicate_layer, CNN_struct *settings, string file_name);
/*
void test_function(int a);
//vector<Neuron> read_script(string file_path);
Neuron* Object_initializer(float* parameters);
vector<float> Generate_matrices(Neuron* Pool_neurons, int neuron_number);
float GPU_kernl_Izhikevich(Neuron* Pool_neurons, int neuron_number, float* regulator);
void GPU_kernel_Stochastic(Neuron *NeuronList, float *random_number, int number_of_threads_y, int SIZE);
//void read_neuron_list(Neuron *NeuronList, int neuron_model, string file_name);
int read_neuron_list(Neuron *NeuronList, int neuron_model, string file_name);
*/
//==========initialization functions==============
void neuron_list_init(Neuron *NeuronList, int network_size);
void neuron_list_init(Neuron *NeuronList);
void input_neuron_list_init(Input_neuron *NeuronList, int network_size);
void init_log_v (float *log_v);
void init_data_log (float *log_v_host, float *log_spike_host, float *log_total_spike_host, int inter);
void izh_parameter_init(float *izh_parameters);
//==========GPU utility functions==============
__global__ void change_threshold (Neuron *NeuronList, int network_size, float start_depth, float end_depth, float target_threshold);
__global__ void lateral_inhibition_depth_wise_mother_thread (Neuron *NeuronList, int network_size, int depth_ind_to_learn, int inhibit_time, CNN_struct *CNN_setttings, float *spike_flag, int total_depth_number);
__global__ void lateral_inhibition_mother_thread (Neuron *NeuronList, int network_size, int layer_ind_to_learn, int inhibit_time, CNN_struct *CNN_setttings, int *spike_flag);
__global__ void lateral_inhibition_child (Neuron *NeuronList, int network_size, int inhibit_time, float start_depth, float end_depth, int depth_iter);
__global__ void reset_membrane_potential (Neuron *NeuronList, int network_size, float start_depth, float end_depth);
__global__ void reset_all_state (Neuron *NeuronList, int network_size, float start_depth, float end_depth);
__global__ void update_param (Neuron *NeuronList, int network_size, float start_depth, float end_depth, int target_param, float target_value);


void copy_filter_to_cuDNN(Neuron *NeuronList, CNN_struct *CNN_settings, float **filter, int spiking_neuron_size);
//==========learning options==============
void run_cnn(string index_prefix, float input_float, float input_float_2, int input_int, int input_int_2, string input_img);
float spiking_learning_label(string network_data, string flag_file, int input_index, int num_test, int function_select, int data_set_select);
void run_cnn_multilayer(string index_prefix, float input_float, float input_float_2, int input_int, int input_int_2, string input_img);
void run_autotune(string index_prefix, float input_float, float input_float_2, int input_int, int input_int_2, string input_img);
void run_time_sequence(string index_prefix, float input_float, float input_float_2, int input_int, int input_int_2, string input_img);
void run_sc2(string index_prefix, float input_float, float input_float_2, int input_int, int input_int_2, string input_img);
void run_event_based_learning(string index_prefix, float input_float, float input_float_2, int input_int, int input_int_2, string input_img);
void run_event_based_learning_hsnn(string index_prefix, float input_float, float input_float_2, int input_int, int input_int_2, string input_img, int resume_learning, int start_layer);
//==========inference options==============
void run_cnn_multilayer_inference(string index_prefix, float input_float, float input_float_2, int input_int, int input_int_2, string input_img);
void run_autotune_inference(string index_prefix, float input_float, float input_float_2, int input_int, int input_int_2, string input_img);
void run_event_based_inference(string index_prefix, float input_float, float input_float_2, int input_int, int input_int_2, string input_img);
void run_event_based_inference_hsnn(string index_prefix, float input_float, float input_float_2, int input_int, int input_int_2, string input_img);
//==========data reader===================
void read_filter_data(string image_file, float *image, int num, int pixel_num);
void CIFAR_read_image_one_channel(float *image, int image_size, int channel, int data_set_choise);
void CIFAR_read_image(float *image, int image_size, int total_img_num, int data_set_choise, bool if_gray_scale);
void GTVIR_read_image(float *image, int image_size, int total_img_num);
void CIFAR_read_label(int *label, int data_set_choise);
void MNIST_read_image(string image_file, float *image , int num);
void MNIST_read_label(string label_file, int *label, int num);
void KAIST_PED_read_image(string image_path, float *image , int num);
void read_sine_seq(string image_file, float *image, int num);
void imageNET_read_image(string folder_to_read, float *image , int num);
void read_sc2(string image_file, float *image, int num);
void read_sc2_2(string image_file, float *image, int num);
void read_sc2_3(string image_file, float *image, int num);
void read_polygon(string folder_to_read, float *image, int num);
void read_one_image(string dir_to_read, float *image, int num);
void DVS_read_image_8bit(string image_file, float *image , int num);
void NTU_skeleton_read_image(string image_file, float *image , int num, int offset);
int IBM_DVS128_event_based(string file_name, Event_Camera_Input *events, int image_size, int total_img_num);
int IBM_DVS128_event_based_one_line(string file_name, Event_Camera_Input *events, int this_file_total_line, int target_line);
int IBM_DVS128_event_based_count_line(string file_name);
#endif /* HEADER_H_ */

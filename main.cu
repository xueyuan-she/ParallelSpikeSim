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
//#include "learning_options.cu"
//#include "mnist/mnist_reader_less.hpp"
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
using namespace std;


#define tau 10
#define exp_coeff 1.442695
#define SIZE 50000  //for ROI, use 30000
#define MAX_TIME 2500000 //in ms
#define TEST_TIME 1000

int main()
{
	//clock_t t1_reading, t2_reading;
	clock_t t_start, t_end;
	//float time;

	//cout << "================ Welcome to Xueyuan She and Yun Long's ParallelSpikeSim =================" << endl << endl;
	//cout << endl;
	cout<<"Function Select: ";
	int mode_select;
	cin >> mode_select;
	t_start = clock();

	int input_index = 0;
	switch (mode_select){

					case 0:
					{
						cudaSetDevice(0);
						run_cnn_multilayer_inference("", 0.8, -1.0, 10000, 5, "spike_cnn.txt");
					}
					break;

					case 1:
					{
						printf("Case 1 selected/n");
						spiking_learning_label("device2_output_network.txt", "device2_output_network_flaged_network_4.csv", 500, 1000, 1, 0);
					}
					break;
					//case 2: run_test(); break; ./rotating_f_mnist/test_2_4/rotating_mnist_val
					case 3: {

					}
					break;
					case 4:
					{
						cudaSetDevice(1);
						run_time_sequence("", 1, -1.0, 10, 5, "spike_cnn.txt");
					}
					break;
					case 5:
						cout<<"Running CNN Multilayer"<<endl;
						cudaSetDevice(2);
						run_cnn_multilayer("", 1, -1.0, 100, 5, "spike_cnn.txt");
					break;
					case 6:
					{
						cudaSetDevice(0);
						run_cnn("", 0.8, -1.0, 500, 5, "spike_cnn.txt");
					}
					break;
					case 7:
					{
						cudaSetDevice(1);
						cout<<"Ruuning H-SNN Learning Layer by Layer"<<endl;
						//run HSNN learning
						for (int layer_to_learn=1; layer_to_learn<CNN_total_layer_num; layer_to_learn++){
							cout<<endl<<"==========Learning Layer "<<layer_to_learn<<"=========="<<endl;
							if (layer_to_learn==1) run_event_based_learning_hsnn("1", 1, -1.0, 2, 5, "spike_cnn.txt", 0, layer_to_learn);
							else run_event_based_learning_hsnn(to_string(layer_to_learn), 1, -1.0, 2, 5, "spike_cnn.txt", 1, layer_to_learn);
						}
					}
					break;
					case 8:
					{
						cudaSetDevice(2);
						run_event_based_inference_hsnn("", 0.8, -1.0, 2, 5, "spike_cnn.txt");
					}
					break;


	}

	t_end = clock();
	cout << "Information summary: " << endl;

	//cout << "Calling GPU kernel uses: " << elapase_time[0]/1000 << " (ms)" << endl;
	//cout << "Actual GPU kernel elapse time is: " << elapase_time[1] << " (ms)" << endl << endl;
	cout << "Total simulation time is " << (t_end - t_start)/1000 << " (ms)" << endl;

	cout << endl;
	cout << "============ Simulation is done, please check your output ============" << endl << endl;
	cout << "Thanks for using my Simulator" << endl << endl;

	return 0;
}
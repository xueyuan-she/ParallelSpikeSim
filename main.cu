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

	cout << "================ Welcome to Xueyuan She and YunLong's Biophysical Neural Network Simulator =================" << endl << endl;
	cout << endl;
	cout<<"Function Select: ";
	int mode_select;
	cin >> mode_select;
	t_start = clock();


	string data_file = "50k_12_output.txt";
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
					//case 2: run_test(); break;
					case 3: {
						cudaSetDevice(1);
						int total_folder_num = 30;
						//string path_prefix[total_folder_num] = {"1_1","1_2","1_3","1_4","1_5","1_6","1_7","1_8","2_0","2_1","3_0","4_0","4_1","4_2","4_3","5_0", "5_1","6_0","6_1"};
						//string path_prefix[total_folder_num] = {"4_0","4_1","4_2","4_3"};"3_0","3_1","3_2","4_0","4_1","4_2",
						string path_prefix[total_folder_num] = {"2_0","2_1","2_2","3_0","3_1","3_2","4_0","4_1","4_2","5_0","5_1","5_2","5_3","6_0","6_1","6_2","6_3","6_4","6_5","6_6"};
						//string path_prefix[total_folder_num] = {"1_0","2_0","3_0","4_0","5_0","6_0","7_1","7_2","7_3"};//for 20 steps
						for (int i=0; i<total_folder_num; i++) {
							for (int j=0; j<1; j++) {
								string prefix = (path_prefix[i]+"-"+to_string(j));
								cout<<prefix<<endl;
								run_sc2(prefix, 1, -1.0, 300, 5, "spike_cnn.txt");
								//run_time_sequence(prefix, 1, -1.0, 100, 5, "spike_cnn.txt");
							}
						}
					}
					break;
					case 4:
					{
						cudaSetDevice(1);
						run_time_sequence("", 1, -1.0, 10, 5, "spike_cnn.txt");
					}
					break;
					case 5:
						cudaSetDevice(1);
						run_cnn_multilayer("", 0.8, -1.0, 300, 5, "spike_cnn.txt");
					break;
					case 6:
					{
						cudaSetDevice(0);
						run_cnn("", 0.8, -1.0, 500, 5, "spike_cnn.txt");
					}
					break;
					case 7:
					{
						  string image_path = "/hdd3/KAIST_PED/KAIST_PED/images/set09/V000/lwir/I00014.jpg";
						  cv::Mat image = cv::imread(image_path.c_str(), CV_LOAD_IMAGE_COLOR);

						  image.convertTo(image, CV_32FC3);
						  cv::resize(image,image, cv::Size(), 0.5, 0.5);
						  cv::imwrite("test_3.png", image);
					}
					break;
					case 8:
					{
						string ext = ".jpg";
						namespace fs = boost::filesystem;
						fs::path Path("./IR_data/IR1");
						int cnt = 0;
						fs::directory_iterator end_iter; // Default constructor for an iterator is the end iterator
						for (fs::directory_iterator iter(Path); iter != end_iter; ++iter){
							if(iter->path().extension() == ext){
								cout<<to_string(cnt)<<" ";
								cnt ++;
							}
						}
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
	cout << "Press any key to exit SNNsim." << endl;

	return 0;
}

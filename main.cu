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

					//case 0: spiking_learning_label(); break;
					case 1:
					{
						printf("Case 1 selected/n");
						spiking_learning_label("device2_output_network.txt", "device2_output_network_flaged_network_4.csv", 500, 1000, 1, 0);
					}
					break;
					//case 2: run_test(); break;
					//case 3: run_ROI(); break;
					case 4:
					{
						float *mnist_img = new float[1];
						KAIST_PED_read_image("", mnist_img, 4000);
					}
					break;
					case 5:
						run_cnn("", 400, -1.0, 300, 10, "spike_cnn.txt");
					break;
					case 6:
					{
						cudaSetDevice(0);
						run_cnn("", 1, -1.0, 400, 5, "spike_cnn.txt");
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

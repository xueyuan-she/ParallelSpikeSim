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
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
using namespace std;


//void save_image(float *img_data, string file_name){
//	int in_channel = input_image_channel;
//	int out_width = input_image_w;
//	int out_height = input_image_l;
//	int total_pixel = input_image_w*input_image_l;
//
//	//cimg_library::CImg<unsigned char> image("spiking_out_seed.jpg");
//	cimg_library::CImg<float> image(input_image_w, input_image_l, 1, 3, 0);
//	int img_i;
//	int img_j;
//	int img_k;
//	if (in_channel == 1){
//		for (img_i=0;img_i<out_width;img_i++){
//			for (img_j=0;img_j<out_height;img_j++){
//				for(img_k=0;img_k<3;img_k++){
//					float weight_raw = img_data[img_i*out_width+img_j];
//					image(img_j, img_i, 0, img_k) = 255*weight_raw;
//				}
//			}
//		}
//	}
//	else{
//		for(img_k=0;img_k<in_channel;img_k++){
//			for (img_i=0;img_i<out_width;img_i++){
//				for (img_j=0;img_j<out_height;img_j++){
//					float weight_raw = img_data[total_pixel*img_k+img_i*out_width+img_j];
//					image(img_j, img_i, img_k) = weight_raw;
//				}
//			}
//		}
//	}
//	string out_file_name = file_name;
//
//	image._data = img_data;
//	image.normalize(0,255);
//	//image.save(out_file_name.c_str());
//	image.save_png(out_file_name.c_str());
//
//}
//
//void load_image(float *img_data, string file_name){
//	int in_channel = input_image_channel;
//	int out_width = input_image_w;
//	int out_height = input_image_l;
//	int total_pixel = input_image_w*input_image_l;
//
//	cimg_library::CImg<float> image(file_name.c_str());
//	image.normalize(0,1);
//	int img_i;
//	int img_j;
//	int img_k;
//	for(img_k=0;img_k<in_channel;img_k++){
//		for (img_i=0;img_i<out_width;img_i++){
//			for (img_j=0;img_j<out_height;img_j++){
//				img_data[total_pixel*img_k+img_i*out_width+img_j] = image(img_j, img_i, img_k);
//			}
//		}
//	}
//	img_data = image._data;
//
//}



void load_image(float *img_data, string image_path) {

  cv::Mat image = cv::imread(image_path.c_str(), cv::IMREAD_COLOR);

  image.convertTo(image, CV_32FC3);

  cv::imwrite("test_3.png", image);
  cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);

  std::cerr << "Input Image: " << image.rows << " x " << image.cols << " x "
            << image.channels() << std::endl;

  int image_bytes = input_image_channel * input_image_l * input_image_w * sizeof(float);
  memcpy(img_data, image.ptr<float>(0), image_bytes);


//  int image_size = input_image_channel * input_image_l * input_image_w;
//  for (int i=0; i<image_size; i++){
//	  cout<<image.ptr<float>(0);
//  }

}



void save_image(float* buffer, string image_path) {
	int width = input_image_w;
	int height = input_image_l;

  cv::Mat output_image(height, width, CV_32FC3, buffer);

//  int image_bytes = input_image_channel * input_image_l * input_image_w * sizeof(float);
//  int image_size = input_image_channel * input_image_l * input_image_w;
//  for (int i=0; i<image_size; i++){
//	  //output_image.at<float>(i) = buffer[i];
//  }

  cv::threshold(output_image,
                output_image,
                /*threshold=*/0,
                /*maxval=*/0,
                cv::THRESH_TOZERO);
  cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
  output_image.convertTo(output_image, CV_8UC3);
  cv::imwrite(image_path, output_image);

}


void img_util(float *img_data, string file_name, int function_select){
	switch (function_select){
						case 0: save_image(img_data, file_name);
						break;
						case 1: load_image(img_data, file_name);
						break;
	}
}

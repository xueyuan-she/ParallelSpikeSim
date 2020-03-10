#include <cudnn.h>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include "header.h"
#include <opencv2/opencv.hpp>

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

cv::Mat load_image(const char* image_path) {
  cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
  image.convertTo(image, CV_32FC3);
  cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
  std::cerr << "Input Image: " << image.rows << " x " << image.cols << " x "
            << image.channels() << std::endl;
  return image;
}

void save_image(const char* output_filename,
                float* buffer,
                int height,
                int width) {
  cv::Mat output_image(height, width, CV_32FC3, buffer);
  // Make negative values zero.
  cv::threshold(output_image,
                output_image,
                /*threshold=*/0,
                /*maxval=*/0,
                cv::THRESH_TOZERO);
  cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
  output_image.convertTo(output_image, CV_8UC3);
  cv::imwrite(output_filename, output_image);
  std::cerr << "Wrote output to " << output_filename << std::endl;
}

__global__ void copy_pointer(float **source, float *target, int index){
	target = source[index];
}

__global__ void read_data(float *data){
	printf("Reading data from GPU");
	printf("%f\n", data[0]);
}

static int checkCudnnError(cudnnStatus_t code, const char* expr, const char* file, int line) {
    if (code)  {
        printf("CUDNN error at %s:%d, code=%d (%s) in '%s'\n", file, line, (int) code, cudnnGetErrorString(code), expr);
        return 1;
    }
    return 0;
}

#define checkCudnnErr(...)      do { int err = checkCudnnError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__);  } while (0)

//int convolution_kernel(CNN_struct *settings, int layer_index, float **d_input_2d, float **filter_2d, float **output_2d) {
////
//	cv::Mat image = load_image("testimg_small.png");
//
//
//	  cudnnHandle_t cudnn;
//	  cudnnCreate(&cudnn);
//
//	  cudnnTensorDescriptor_t input_descriptor;
//	  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
//	  checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
//	                                        /*format=*/CUDNN_TENSOR_NHWC,
//	                                        /*dataType=*/CUDNN_DATA_FLOAT,
//	                                        /*batch_size=*/1,
//	                                        /*channels=*/3,
//	                                        /*image_height=*/image.rows,
//	                                        /*image_width=*/image.cols));
//
//	  cudnnFilterDescriptor_t kernel_descriptor;
//	  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
//	  checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
//	                                        /*dataType=*/CUDNN_DATA_FLOAT,
//	                                        /*format=*/CUDNN_TENSOR_NCHW,
//	                                        /*out_channels=*/3,
//	                                        /*in_channels=*/3,
//	                                        /*kernel_height=*/3,
//	                                        /*kernel_width=*/3));
//
//	  cudnnConvolutionDescriptor_t convolution_descriptor;
//	  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
//	  checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
//	                                             /*pad_height=*/1,
//	                                             /*pad_width=*/1,
//	                                             /*vertical_stride=*/1,
//	                                             /*horizontal_stride=*/1,
//	                                             /*dilation_height=*/1,
//	                                             /*dilation_width=*/1,
//	                                             /*mode=*/CUDNN_CROSS_CORRELATION,
//	                                             /*computeType=*/CUDNN_DATA_FLOAT));
//
//	  int batch_size{0}, channels{0}, height{0}, width{0};
//	  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
//	                                                   input_descriptor,
//	                                                   kernel_descriptor,
//	                                                   &batch_size,
//	                                                   &channels,
//	                                                   &height,
//	                                                   &width));
//
//	  std::cerr << "Output Image: " << height << " x " << width << " x " << channels
//	            << std::endl;
//
//	  cudnnTensorDescriptor_t output_descriptor;
//	  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
//	  checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
//	                                        /*format=*/CUDNN_TENSOR_NHWC,
//	                                        /*dataType=*/CUDNN_DATA_FLOAT,
//	                                        /*batch_size=*/1,
//	                                        /*channels=*/3,
//	                                        /*image_height=*/image.rows,
//	                                        /*image_width=*/image.cols));
//
//	  cudnnConvolutionFwdAlgo_t convolution_algorithm;
//	  checkCUDNN(
//	      cudnnGetConvolutionForwardAlgorithm(cudnn,
//	                                          input_descriptor,
//	                                          kernel_descriptor,
//	                                          convolution_descriptor,
//	                                          output_descriptor,
//	                                          CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
//	                                          /*memoryLimitInBytes=*/0,
//	                                          &convolution_algorithm));
//
//	  size_t workspace_bytes{0};
//	  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
//	                                                     input_descriptor,
//	                                                     kernel_descriptor,
//	                                                     convolution_descriptor,
//	                                                     output_descriptor,
//	                                                     convolution_algorithm,
//	                                                     &workspace_bytes));
//	  std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
//	            << std::endl;
//	  assert(workspace_bytes > 0);
//
//	  void* d_workspace{nullptr};
//	  cudaMalloc(&d_workspace, workspace_bytes);
//
//	  int image_bytes = batch_size * channels * height * width * sizeof(float);
//
//	  float* d_input{nullptr};
//	  cudaMalloc(&d_input, image_bytes);
//	  cudaMemcpy(d_input, image.ptr<float>(0), image_bytes, cudaMemcpyHostToDevice);
//
//	  float* d_output{nullptr};
//	  cudaMalloc(&d_output, image_bytes);
//	  cudaMemset(d_output, 0, image_bytes);
//
//	  // clang-format off
//	  const float kernel_template[3][3] = {
//	    {1, 1, 1},
//	    {1, -8, 1},
//	    {1, 1, 1}
//	  };
//	  // clang-format on
//
//	  float h_kernel[3][3][3][3];
//	  for (int kernel = 0; kernel < 3; ++kernel) {
//	    for (int channel = 0; channel < 3; ++channel) {
//	      for (int row = 0; row < 3; ++row) {
//	        for (int column = 0; column < 3; ++column) {
//	          h_kernel[kernel][channel][row][column] = kernel_template[row][column];
//	        }
//	      }
//	    }
//	  }
//
//	  float* d_kernel{nullptr};
//	  cudaMalloc(&d_kernel, sizeof(h_kernel));
//	  cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);
//
//	  const float alpha = 1.0f, beta = 0.0f;
//
//	  checkCUDNN(cudnnConvolutionForward(cudnn,
//	                                     &alpha,
//	                                     input_descriptor,
//	                                     d_input,
//	                                     kernel_descriptor,
//	                                     d_kernel,
//	                                     convolution_descriptor,
//	                                     convolution_algorithm,
//	                                     d_workspace,
//	                                     workspace_bytes,
//	                                     &beta,
//	                                     output_descriptor,
//	                                     d_output));
//
//
//
//	  float* h_output = new float[image_bytes];
//	  cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost);
//
//	  save_image("cudnn-out.png", h_output, height, width);
//
//	  delete[] h_output;
//	  cudaFree(d_kernel);
//	  cudaFree(d_input);
//	  cudaFree(d_output);
//	  cudaFree(d_workspace);
//
//	  cudnnDestroyTensorDescriptor(input_descriptor);
//	  cudnnDestroyTensorDescriptor(output_descriptor);
//	  cudnnDestroyFilterDescriptor(kernel_descriptor);
//	  cudnnDestroyConvolutionDescriptor(convolution_descriptor);
//
//	  cudnnDestroy(cudnn);
//
//	return 1;
//}




int convolution_kernel_setup(Convolution_setting_struct *convolution_settings, CNN_struct *settings, int layer_index){
	int filter_in_channel;
	int filter_out_channel;
	int filter_height;
	int filter_width;

	int input_batch_size = 1;
	int input_channel;
	int input_height;
	int input_width;

	int output_channel;
	int output_batch_size = 1;
	int output_height;
	int output_width;



	if(layer_index==0){
		filter_in_channel = input_image_channel;
		filter_out_channel = settings->layer[layer_index+1].depth;
		filter_height = settings->layer[layer_index+1].conv_setting.filter_length;
		filter_width = settings->layer[layer_index+1].conv_setting.filter_width;

		input_batch_size = 1;
		input_channel = input_image_channel;
		input_height = input_image_l;
		input_width = input_image_w;

		output_channel = settings->layer[layer_index+1].depth;
		output_batch_size = 1;
		output_height = settings->layer[layer_index+1].depth_list[0].length;
		output_width = settings->layer[layer_index+1].depth_list[0].width;
	}else{
		filter_in_channel = settings->layer[layer_index+1].conv_setting.filter_depth;
		filter_out_channel = settings->layer[layer_index+1].depth;
		filter_height = settings->layer[layer_index+1].conv_setting.filter_length;
		filter_width = settings->layer[layer_index+1].conv_setting.filter_width;

		input_batch_size = 1;
		input_channel = settings->layer[layer_index+1].conv_setting.filter_depth;
		input_height = settings->layer[layer_index].depth_list[0].length;
		input_width = settings->layer[layer_index].depth_list[0].width;

		output_channel = settings->layer[layer_index+1].depth;
		output_batch_size = 1;
		output_height = settings->layer[layer_index+1].depth_list[0].length;
		output_width = settings->layer[layer_index+1].depth_list[0].width;

		printf("\n=====Input Channel: %d, height: %d, width: %d___output: %d, %d, %d=====\n", input_channel, input_height, input_width, output_channel, output_height, output_width);
	}





	cudnnHandle_t cudnn;
	cudnnCreate(&cudnn);



	cudnnTensorDescriptor_t input_descriptor;
	checkCudnnErr(cudnnCreateTensorDescriptor(&input_descriptor));
	checkCudnnErr(cudnnSetTensor4dDescriptor(input_descriptor,
										/*format=*/CUDNN_TENSOR_NHWC,
										/*dataType=*/CUDNN_DATA_FLOAT,
										/*batch_size=*/input_batch_size,
										/*channels=*/input_channel,
										/*image_height=*/input_height,
										/*image_width=*/input_width));




	cudnnFilterDescriptor_t kernel_descriptor;
	checkCudnnErr(cudnnCreateFilterDescriptor(&kernel_descriptor));
	checkCudnnErr(cudnnSetFilter4dDescriptor(kernel_descriptor,
										/*dataType=*/CUDNN_DATA_FLOAT,
										/*format=*/CUDNN_TENSOR_NCHW,
										/*out_channels=*/filter_out_channel,
										/*in_channels=*/filter_in_channel,
										/*kernel_height=*/filter_height,
										/*kernel_width=*/filter_width));



	cudnnConvolutionDescriptor_t convolution_descriptor;
	checkCudnnErr(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
	checkCudnnErr(cudnnSetConvolution2dDescriptor(convolution_descriptor,
											 /*pad_height=*/settings->layer[layer_index+1].conv_setting.pad_height,
											 /*pad_width=*/settings->layer[layer_index+1].conv_setting.pad_width,
											 /*vertical_stride=*/settings->layer[layer_index+1].conv_setting.vertical_stride,
											 /*horizontal_stride=*/settings->layer[layer_index+1].conv_setting.horizontal_stride,
											 /*dilation_height=*/settings->layer[layer_index+1].conv_setting.dilation_height,
											 /*dilation_width=*/settings->layer[layer_index+1].conv_setting.dilation_width,
											 /*mode=*/CUDNN_CROSS_CORRELATION,
											 /*computeType=*/CUDNN_DATA_FLOAT));




	int batch_size{0}, channels{0}, height{0}, width{0};
	checkCudnnErr(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
												   input_descriptor,
												   kernel_descriptor,
												   &batch_size,
												   &channels,
												   &height,
												   &width));


	cudnnTensorDescriptor_t output_descriptor;
	checkCudnnErr(cudnnCreateTensorDescriptor(&output_descriptor));
	checkCudnnErr(cudnnSetTensor4dDescriptor(output_descriptor,
										/*format=*/CUDNN_TENSOR_NHWC,
										/*dataType=*/CUDNN_DATA_FLOAT,
										/*batch_size=*/output_batch_size,
										/*channels=*/output_channel,
										/*image_height=*/output_height,
										/*image_width=*/output_width));


	cudnnConvolutionFwdAlgo_t convolution_algorithm;
	checkCudnnErr(
	  cudnnGetConvolutionForwardAlgorithm(cudnn,
										  input_descriptor,
										  kernel_descriptor,
										  convolution_descriptor,
										  output_descriptor,
										  CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
										  /*memoryLimitInBytes=*/0,
										  &convolution_algorithm));


	size_t workspace_bytes{0};
	checkCudnnErr(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
														input_descriptor,
														kernel_descriptor,
														convolution_descriptor,
														output_descriptor,
														convolution_algorithm,
															&workspace_bytes));
	//std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB" << std::endl;
	assert(workspace_bytes > 0);

	void* d_workspace{nullptr};
	cudaMalloc(&d_workspace, workspace_bytes);

	convolution_settings[layer_index].convolution_algorithm = convolution_algorithm;
	convolution_settings[layer_index].convolution_descriptor = convolution_descriptor;
	convolution_settings[layer_index].cudnn = cudnn;

	convolution_settings[layer_index].input_descriptor = input_descriptor;
	convolution_settings[layer_index].kernel_descriptor = kernel_descriptor;
	convolution_settings[layer_index].output_descriptor = output_descriptor;
	convolution_settings[layer_index].workspace_bytes = workspace_bytes;
	convolution_settings[layer_index].d_workspace = d_workspace;
	return 1;
}



int convolution_kernel(Convolution_setting_struct convolution_settings, int layer_index, float **d_input_2d, float **filter_2d, float **output_2d, float *probe) {
	//int filter_index = layer_index - 1;
	int convolution_result_index = layer_index;
	//if (layer_index==0) convolution_result_index = 0;

//	float *d_input;
//	float *filter;
//	float *output;
//
//    dim3 dimBlock(1, 1 );
//    dim3 dimGrid(1, 1);
//    copy_pointer<<<dimGrid, dimBlock>>>(d_input_2d, d_input, layer_index);
//    copy_pointer<<<dimGrid, dimBlock>>>(filter_2d, filter, convolution_result_index);
//    copy_pointer<<<dimGrid, dimBlock>>>(output_2d, output, convolution_result_index);
//	float **add = &output_2d[0];
//	printf("Address On GPU: %p\n", add);
	//read_data<<<1, 1>>>(output_2d[0]);


//	printf("\nperforming convolution for layer: %d\n", layer_index);
//	cout<<"input_batch_size: "<<input_batch_size<<endl;
//	cout<<"input_channel: "<<input_channel<<endl;
//	cout<<"input_height: "<<input_height<<endl;
//	cout<<"input_width: "<<input_width<<endl;
//	cout<<"filter_in_channel: "<<filter_in_channel<<endl;
//	cout<<"filter_out_channel: "<<filter_out_channel<<endl;
//	cout<<"filter_height: "<<filter_height<<endl;
//	cout<<"filter_width: "<<filter_width<<endl;
//	cout<<"pad_height: "<<settings->layer[layer_index].conv_setting.pad_height<<endl;
//	cout<<"pad_width: "<<settings->layer[layer_index].conv_setting.pad_width<<endl;
//	cout<<"vertical_stride: "<<settings->layer[layer_index].conv_setting.vertical_stride<<endl;
//	cout<<"horizontal_stride: "<<settings->layer[layer_index].conv_setting.horizontal_stride<<endl;
//	cout<<"dilation_height: "<<settings->layer[layer_index].conv_setting.dilation_height<<endl;
//	cout<<"dilation_width: "<<settings->layer[layer_index].conv_setting.dilation_width<<endl;
//	cout<<"output_channel: "<<output_channel<<endl;
//	cout<<"output_batch_size: "<<output_batch_size<<endl;
//	cout<<"output_height: "<<output_height<<endl;
//	cout<<"output_width: "<<output_width<<endl<<endl;
//	cout<<"batch_size(from cuDNN): "<<batch_size<<endl;
//	cout<<"channels(from cuDNN): "<<channels<<endl;
//	cout<<"height(from cuDNN): "<<height<<endl;
//	cout<<"width(from cuDNN): "<<width<<endl;






//	int image_bytes = batch_size * channels * height * width * sizeof(float);
//	float* d_input{nullptr};
//	cudaMalloc(&d_input, image_bytes);
//	cudaMemcpy(d_input, input, image_bytes, cudaMemcpyHostToDevice);



//	float* d_output{nullptr};
//	cudaMalloc(&d_output, image_bytes);
//	cudaMemset(d_output, 0, image_bytes);

//	const float kernel_template[3][3] = {
//
//	{1, 1, 1},
//	{1, -8, 1},
//	{1, 1, 1}
//
//	};
//
//	float h_kernel[filter_in_channel][filter_out_channel][filter_height][filter_width];
//	for (int kernel = 0; kernel < filter_in_channel; ++kernel) {
//	for (int channel = 0; channel < filter_out_channel; ++channel) {
//	  for (int row = 0; row < filter_height; ++row) {
//		for (int column = 0; column < filter_width; ++column) {
//		  h_kernel[kernel][channel][row][column] = kernel_template[row][column];
//		}
//	  }
//	}
//	}
//
//	float* d_kernel{nullptr};
//	cudaMalloc(&d_kernel, sizeof(h_kernel));
//	cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

	const float alpha = 1.0f, beta = 0.0f;
//	printf("*IN CONVOLUTION KERNEL*\nThe value of convolution_result_index is: %d\n", convolution_result_index);
	checkCudnnErr(cudnnConvolutionForward(convolution_settings.cudnn,
									 &alpha,
									 convolution_settings.input_descriptor,
									 d_input_2d[layer_index],
									 convolution_settings.kernel_descriptor,
									 filter_2d[convolution_result_index],
									 convolution_settings.convolution_descriptor,
									 convolution_settings.convolution_algorithm,
									 convolution_settings.d_workspace,
									 convolution_settings.workspace_bytes,
									 &beta,
									 convolution_settings.output_descriptor,
									 output_2d[convolution_result_index]));

	//cudaMemcpy(output, d_output, image_bytes, cudaMemcpyDeviceToHost);
//
//	printf("\n input print: \n");
//	float *temp_1 = new float[784];
//	cudaMemcpy(temp_1, d_input_2d[layer_index], 784*sizeof(float), cudaMemcpyDeviceToHost);
//	for(int ij=0;ij<784;ij++) printf(" %1.0f ", temp_1[ij]);
//	delete [] temp_1;
//
//	printf("\n filter print: \n");
//	float *temp_2 = new float[784000];
//	cudaMemcpy(temp_2, filter_2d[convolution_result_index], 784000*sizeof(float), cudaMemcpyDeviceToHost);
//	for(int ij=0;ij<784000;ij++) printf(" %1.1f|", temp_2[ij]);
//	delete [] temp_2;
//
//	printf("\n output print: \n");
//	float *temp = new float[1000];
//	cudaMemcpy(temp, output_2d[convolution_result_index], 1000*sizeof(float), cudaMemcpyDeviceToHost);
//	for(int ij=0;ij<1000;ij++){
//		probe[ij] = probe[ij]+temp[ij]/100;
//
//	}
//	delete [] temp;
//	float *temp_2 = new float[784000];
//	cudaMemcpy(temp_2, filter_2d[convolution_result_index], 784000*sizeof(float), cudaMemcpyDeviceToHost);
//	for(int ij=0;ij<1000;ij++){
//		float mean_temp = 0;
//		int addition = ij*784;
//		for(int ijk=0;ijk<784;ijk++){
//			mean_temp += temp_2[addition+ijk];
//			//printf("%f|", temp_2[ijk]);
//		}
//		probe[ij] += mean_temp/100;
//	}
//	delete [] temp_2;

	//cudaFree(d_output);
//	cudaFree(d_workspace);

//	cudnnDestroyTensorDescriptor(input_descriptor);
//	cudnnDestroyTensorDescriptor(output_descriptor);
//	cudnnDestroyFilterDescriptor(kernel_descriptor);
//	cudnnDestroyConvolutionDescriptor(convolution_descriptor);
//
//	cudnnDestroy(cudnn);

	return 1;
}

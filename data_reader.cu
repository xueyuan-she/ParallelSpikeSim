#include <iostream>
#include <time.h>
#include <vector>
#include <string>
#include <stdlib.h>
#include <streambuf>
#include <sstream>
#include <fstream>
#include <math.h>
#include <assert.h>
#include "cifar10_reader.hpp"
//#include "learning_options.cu"
//#include "mnist/mnist_reader_less.hpp"
#include <boost/filesystem.hpp>

#include "header.h"

using namespace std;

int to_int(char* p)
{
  return ((p[0] & 0xff) << 24) | ((p[1] & 0xff) << 16) |
         ((p[2] & 0xff) <<  8) | ((p[3] & 0xff) <<  0);
}

void read_filter_data(string image_file, float *image, int num, int pixel_num){
	ifstream file(image_file);
	string line;

	while(getline(file, line))
	{
		stringstream lineStream(line);
		string cell;

	}

}

void CIFAR_read_image_one_channel(float *image, int image_size, int channel, int data_set_choise){

	auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
	//uint8_t* cifar_training_images = new uint8_t[dataset.training_images.size()];
	//std::copy(dataset.training_images.begin(), dataset.training_images.end(), cifar_training_images);
	if(data_set_choise==1){
		int size = dataset.test_images.size();
		int index = 0;
		for (int i=0; i<size; ++i) {
			uint8_t *temp = dataset.test_images[i].data();
			for (int j=channel*image_size; j<(channel+1)*image_size; j++){
				float temp_data = float(temp[j])/255.0;
				image[index] =temp_data;
				index ++;
			}
		 }
	}
	else{
		int size = dataset.training_images.size();
		int index = 0;
		for (int i=0; i<size; ++i) {
			uint8_t *temp = dataset.training_images[i].data();
			for (int j=channel*image_size; j<(channel+1)*image_size; j++){
				float temp_data = float(temp[j])/255.0;
				image[index] =temp_data;
				index ++;
			}
		 }

		cimg_library::CImg<unsigned char> image("color_mid.jpg");
		uint8_t *temp = dataset.test_images[1].data();
		int img_k, img_i, img_j;
		for(img_k=0;img_k<3;img_k++){
			for (img_i=0;img_i<input_image_w;img_i++){
				for (img_j=0;img_j<input_image_l;img_j++){
					int count = img_k*input_image_w*input_image_l + img_i*input_image_l + img_j;
					image(img_j, img_i, 0, img_k) = temp[count];
				}
			}
		}
		string out_file_name = "cifar_read_test.jpg";
		image.save(out_file_name.c_str());

	}
}

void CIFAR_read_image(float *image, int image_size, int data_set_choise, bool if_gray_scale){

	auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
	//uint8_t* cifar_training_images = new uint8_t[dataset.training_images.size()];
	//std::copy(dataset.training_images.begin(), dataset.training_images.end(), cifar_training_images);
	if(if_gray_scale){
		cout<<"reading grayscale cifar"<<endl;
		int size = dataset.training_images.size();
		if(data_set_choise==1){
			size = dataset.test_images.size();
		}
		int index = 0;

		for (int i=0; i<size; ++i) {
			uint8_t *temp = dataset.training_images[i].data();
			if(data_set_choise==1){
				temp = dataset.test_images[i].data();
			}
			for (int j=0; j<image_size; j++){
				int rj = j;
				int gj = j+image_size;
				int bj = j+2*image_size;
				float temp_data = (float(temp[rj])+float(temp[gj])+float(temp[bj]))/3.0/255.0;
				image[index] =temp_data;
				index ++;
			}
		}

		cimg_library::CImg<unsigned char> test_image("color_mid.jpg");
		int img_k, img_i, img_j;
		for(img_k=0;img_k<3;img_k++){
			for (img_i=0;img_i<input_image_w;img_i++){
				for (img_j=0;img_j<input_image_l;img_j++){
					int count = img_i*input_image_l + img_j;
					test_image(img_i, img_j, 0, img_k) = int(image[count]*255);
				}
			}
		}
		string out_file_name = "cifar_read_test.jpg";
		test_image.save(out_file_name.c_str());

	}
	else{
		int size = dataset.training_images.size();
		if(data_set_choise==1){
			size = dataset.test_images.size();
		}
		printf("=====training image num: %d=====\n", size);
		int index = 0;
		for (int i=0; i<size; ++i) {
			uint8_t *temp = dataset.training_images[i].data();
			if(data_set_choise==1){
				temp = dataset.test_images[i].data();
			}
			for (int j=0; j<image_size; j++){
				float temp_data = float(temp[j])/255.0;
				image[index] =temp_data;
				index ++;
			}
		 }

		cimg_library::CImg<unsigned char> test_image("color.jpg");
		test_image.resize(input_image_w, input_image_l);
		uint8_t *temp = dataset.test_images[0].data();
		cout<<"input_dim: "<<sizeof(temp)<<endl;
		int img_k, img_i, img_j;
		for(img_k=0;img_k<3;img_k++){
			for (img_i=0;img_i<input_image_w;img_i++){
				for (img_j=0;img_j<input_image_l;img_j++){
					int count = img_k*input_image_w*input_image_l + img_i*input_image_l + img_j;
					test_image(img_i, img_j, 0, img_k) = temp[count];
				}
			}
		}
		string out_file_name = "cifar_read_test.jpg";
		test_image.save(out_file_name.c_str());
	}
}

void GTVIR_read_image(float *image, int image_size, int total_img_num){
	namespace fs = boost::filesystem;
	int folder_cnt = 1;
	int total_image_cnt = 0;
	int total_folder = 10;
	int pixel_cnt = 0;
	cout<<"Reading GTVIR image, No: "<<endl;
	for (int i=0; i<total_folder; i++){
		int folder_img_cnt = 1;
		string folder_name = "IR" + to_string(i+folder_cnt);

		string img_name_temp = "00000000" + to_string(folder_img_cnt);
		int cut_ind = img_name_temp.length()-6;
		string cur_dir = "IR_data/" + folder_name;
		cout<<cur_dir<<endl;
		//ifstream fs(full_dir.c_str());
		fs::path Path(cur_dir);
		fs::directory_iterator end_iter;
		for (fs::directory_iterator iter(Path); iter != end_iter; ++iter){
			if(iter->path().extension() == ".jpg"){
				img_name_temp = "00000000" + to_string(folder_img_cnt);
				cut_ind = img_name_temp.length()-6;
				string full_dir = "IR_data/" + folder_name + "/" + img_name_temp.substr(cut_ind, 6) + ".jpg";
				//cout<<to_string(total_image_cnt)<<' ';
				if (total_image_cnt%1000==0) cout<<to_string(total_image_cnt)<<' ';
				cimg_library::CImg<unsigned char> temp_image(full_dir.c_str());
				//temp_image.save("test.jpg");
				//cout<<"check this pixel:"<<to_string(temp_image(200, 300, 0, 0))<<endl;
				for (int wid=0; wid<input_image_w; wid++){
					for (int len=0; len<input_image_l; len++){
						image[pixel_cnt] = temp_image(wid, len, 0, 0)/255.0;
						pixel_cnt ++;
					}
				}
				folder_img_cnt ++;

				//cout<<full_dir<<endl;
				//fs.open(full_dir.c_str());
				total_image_cnt ++;

				if(total_image_cnt>=total_img_num){
					return;
				}
			}
		}

	}
}

void CIFAR_read_label(int *label, int data_set_choise){
	auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
	if(data_set_choise==1){
		int size = dataset.test_images.size();
		int index = 0;
		for (int i=0; i<size; ++i) {
			uint8_t temp = dataset.test_labels[i];
			label[index] = int(temp);
			index ++;
		 }
	}
	else{
		int size = dataset.training_images.size();
		int index = 0;
		for (int i=0; i<size; ++i) {
			uint8_t temp = dataset.training_labels[i];
			label[index] = int(temp);
			index ++;
		 }
	}
}

void MNIST_read_image(string image_file, float *image , int num){
	ifstream ifs(image_file.c_str(), std::ios::in | std::ios::binary);
	char p[4];

	ifs.read(p, 4);
	int magic_number = to_int(p);
	assert(magic_number == 0x803);

	ifs.read(p, 4);
	int m_size = to_int(p);
	//printf("m_size_is:%d\n",m_size);
	// limit
	if (num != 0 && num < m_size) m_size = num;

	ifs.read(p, 4);
	int m_rows = to_int(p);

	ifs.read(p, 4);
	int m_cols = to_int(p);
	int total_pixel = m_rows * m_cols;
	char* q = new char[m_rows * m_cols];

	//cout<<"num of col: "<<to_string(m_cols)<<"\n";
	//cout<<"num of row: "<<to_string(m_rows)<<'\n';


	for (int i=0; i<m_size; ++i) {
		//ifs.read(q, m_rows * m_cols);
	    //std::vector<double> image(m_rows * m_cols);

	    //====
	    for(int r=0;r<m_rows;++r)
	            {
	                //cout << '\t' << '[' ;
	                for(int c=0;c<m_cols;++c)
	                {
	                	int index = r*(m_rows) + c + i*(total_pixel);
	                    unsigned char temp = 0;
	                    if ( ! ifs.read((char*)&temp,sizeof(temp))){
	                    	cout<< "error opening file";
	                    }
	                    image[index] = float(temp)/(255.0);
	                    //cout << unsigned(temp) << ' ' << image[index] << ' '<< '|';
	                    //cout << ((temp == 0.0)? ' ' : '*');
	                }
	                //cout << ']' << endl;
	            }
	    //cout << ']' << endl;
	    //====

	 }

	 delete[] q;

	 ifs.close();
}

void MNIST_read_label(string label_file, int *label, int num){
	ifstream ifs(label_file.c_str(), std::ios::in | std::ios::binary);
	char p[4];

	ifs.read(p, 4);
	int magic_number = to_int(p);
	assert(magic_number == 0x801);

	ifs.read(p, 4);
	int size = to_int(p);
	  // limit
	if (num != 0 && num < size) size = num;

	for (int i=0; i<size; ++i) {
	    ifs.read(p, 1);
	    int label_read = p[0];
	    //printf("No.%d_label_is:%d\n",i,label_read);
	    label[i] = label_read;
	  }

	ifs.close();

}

void KAIST_PED_read_image(string image_path, float *image , int num){
	namespace fs = boost::filesystem;


	int pixel_cnt = 0;
	int total_image_cnt = 0;
	string cur_dir = "/hdd3/KAIST_PED/KAIST_PED/set00/V000/visible";

	//ifstream fs(full_dir.c_str());
	for (int path_i=0; path_i<2; path_i++){
		int folder_img_cnt = 0;
		if (path_i==1) cur_dir = "/hdd3/KAIST_PED/KAIST_PED/set09/V000/visible";

		fs::path Path(cur_dir);
		fs::directory_iterator end_iter;
		for (fs::directory_iterator iter(Path); iter != end_iter; ++iter){
			if(iter->path().extension() == ".jpg"){
				string img_name_temp = "00000" + to_string(folder_img_cnt);
				int cut_ind = img_name_temp.length()-5;
				string full_dir = cur_dir + "/I" + img_name_temp.substr(cut_ind, 5) + ".jpg";
				//cout<<full_dir<<endl;
				if (total_image_cnt%1000==0) cout<<to_string(total_image_cnt)<<' ';
				cimg_library::CImg<unsigned char> temp_image(full_dir.c_str());
				temp_image.resize(input_image_w, input_image_l);
				//temp_image.save("test.jpg");
				//cout<<"check this pixel:"<<to_string(temp_image(200, 300, 0, 0))<<endl;
				for(int img_k=0;img_k<input_image_channel;img_k++){
					for (int wid=0; wid<input_image_w; wid++){
						for (int len=0; len<input_image_l; len++){
							image[pixel_cnt] = temp_image(wid, len, 0, img_k)/255.0;
							pixel_cnt ++;
						}
					}
				}
				folder_img_cnt ++;

				//cout<<full_dir<<endl;
				//fs.open(full_dir.c_str());
				total_image_cnt ++;

				if(total_image_cnt>=num){
					return;
				}
			}
		}

	}



}


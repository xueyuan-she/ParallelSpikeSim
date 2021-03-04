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

void input_normalization(float *image, int input_dim, int toal_data_num){


}


void read_csv(string save_file_name, float *image, int total_img_num){
	bool normalize = false;


	int total_pixel_num = total_img_num *input_image_w*input_image_l*input_image_channel;
    int one_img_pixel_num = input_image_w*input_image_l*input_image_channel;
	ifstream file (save_file_name);
	string one_line;
	int read_cnt = 0;
	int line_cnt = 0;


	while (file.good()){
		getline(file, one_line);
		stringstream s(one_line);
		string val;
		while(getline(s, val, ',')){
			if (val.empty()) break;
			image[read_cnt] = stof(val);
//			cout<<image[read_cnt]<<" ";
			read_cnt ++;
			if (read_cnt>=total_pixel_num){
				break;
			}
			if ((read_cnt+1)%one_img_pixel_num==0) break;
		}
		line_cnt ++;
//		cout<<endl;
		if (line_cnt>=total_img_num) break;

	}
	cout<<"Total read pixel: "<<read_cnt<<endl;

	if (normalize){
		float input_max = 0.000001;

		for (int i=0; i<total_pixel_num; i++) {
			if(image[i]>input_max) input_max = image[i];
		}

		for (int i=0; i<total_pixel_num; i++) image[i] = image[i]/(input_max);



	}

	return;
}


void read_binary(string image_file, float *image, int num){
	cout<<"Using read_binary, reading images from: "<<image_file<<endl;
	bool if_normalize = false;

	//this is for imagenet
	//float norm_mean[3] = {0.485, 0.456, 0.406};
	//float norm_std[3] = {0.229, 0.224, 0.225};

	//this is for cifar
	float norm_mean[3] = {0.4914, 0.4822, 0.4465};
	float norm_std[3] = {0.2023, 0.1994, 0.2010};

	ifstream ifs(image_file.c_str(), std::ios::in | std::ios::binary);
	char p[4];

	ifs.read(p, 4);

	int magic_number = to_int(p);
	assert(magic_number == 0x803);

	ifs.read(p, 4);
	int m_size = to_int(p);
	printf("image number in file is:%d\n",m_size);
	// limit
	if (num != 0 && num < m_size) m_size = num;

	ifs.read(p, 4);
	int m_rows = to_int(p);

	ifs.read(p, 4);
	int m_cols = to_int(p);



	if(m_cols!=input_image_w || m_rows!=input_image_l){
		cout<<"=====Warning======: input image size mismatch"<<endl;
		cout<<"File rows: "<<m_rows<<" file col: "<<m_cols<<endl;
		//return;
	}
	m_cols=input_image_w; m_rows=input_image_l;
	//cout<<"num of col: "<<to_string(m_cols)<<"\n";
	//cout<<"num of row: "<<to_string(m_rows)<<'\n';
	char* q = new char[m_rows * m_cols];
	int total_pixel = 3*m_rows * m_cols;
	for (int i=0; i<m_size; ++i) {
		//ifs.read(q, m_rows * m_cols);
	    //std::vector<double> image(m_rows * m_cols);

	    //====
		for (int chan=0;chan<input_image_channel;chan++){
			for(int r=0;r<m_rows;++r)
					{
						//cout << '\t' << '[' ;
						for(int c=0;c<m_cols;++c)
						{
							int index = chan*m_rows*m_cols+ r*(m_rows) + c + i*(total_pixel);
							unsigned char temp = 0;
							if ( ! ifs.read((char*)&temp,sizeof(temp))){
								cout<< "error opening file";
								return;
							}
							image[index] = float(temp)/(255.0);
							//image[index] = 0;
							//image[0] = 1;
							if (if_normalize){
								image[index] = (image[index]-norm_mean[chan])/norm_std[chan];

								//if(image[index]<0) image[index]=0;
							}

							//cout << unsigned(temp) << ' ' << image[index] << ' '<< '|';
							//cout << ((temp == 0.0)? ' ' : '*');
						}
						//cout << ']' << endl;
					}
			//cout << ']' << endl;
			//====
		}

	 }

	 delete[] q;

	 ifs.close();


	 int check_data = 1;
	 if(check_data){
		cout<<"output one sample read data..."<<endl;
		int img_to_check = 24;
		cimg_library::CImg<unsigned char> test_image("color_mid.jpg");
		test_image.resize(input_image_w, input_image_l);
		int img_k, img_i, img_j;
		int count = img_to_check*3*input_image_w*input_image_l;
		for(img_k=0;img_k<3;img_k++){
			for (img_i=0;img_i<input_image_w;img_i++){
				for (img_j=0;img_j<input_image_l;img_j++){
					test_image(img_i, img_j, 0, img_k) = int(image[count]*255);
					count ++;
				}
			}
		}
		string out_file_name = "mnist_read_test.jpg";
		test_image.save(out_file_name.c_str());
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

void CIFAR_read_image(float *image, int image_size, int total_img_num, int data_set_choise, bool if_gray_scale){
	cout<<"Reading CIFAR images"<<endl;
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
			if(i>total_img_num) return;
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
//		uint8_t *temp = dataset.training_images[0].data();
//		cout<<"input_dim: "<<sizeof(temp)<<", "<<sizeof(temp[0])<<temp[0]<<temp[1]<<endl;
		for (int i=0; i<size; ++i) {
//			cout<<endl<<i<<endl;
			if(i>=total_img_num) break;
			uint8_t *temp = dataset.training_images[i].data();
			uint8_t label = dataset.training_labels[i];
			if(data_set_choise==1){
				temp = dataset.test_images[i].data();
				label = dataset.test_labels[i];

			}
			//if (i<100) printf("%d..", label);
			for (int j=0; j<image_size; j++){
				float temp_data = float(temp[j])/255.0;
//				if(i>19000)cout<<" "<<temp_data;
//				if(i>4000)cout<<index<<" ";
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
	bool load_from_csv = False;
	bool write_to_file = True;
	bool normalize = False;
	float norm_mean [3] = {0.4914, 0.4822, 0.4465};
	float norm_std [3] = {0.2023, 0.1994, 0.2010};

	string save_file_name = "GTVIR_DATA.csv";
	if(load_from_csv){
		int total_pixel_num = total_img_num *input_image_w*input_image_l*input_image_channel;

		ifstream file (save_file_name);
		string val;
		for(int i=0; i<total_pixel_num; i++){
			if (file.good()){
				getline(file, val, ',');
				image[i] = stof(val);
				//cout<<image[i]<<" ";
			}else{
				cout<<"loading from saved image file not working"<<endl;
				break;
			}
		}

		return;
	}

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
				temp_image.resize(input_image_w, input_image_l);
				//temp_image.save("test.jpg");
				//cout<<"check this pixel:"<<to_string(temp_image(200, 300, 0, 0))<<endl;
				for(int img_k=0;img_k<input_image_channel;img_k++){
					for (int wid=0; wid<input_image_w; wid++){
						for (int len=0; len<input_image_l; len++){
							image[pixel_cnt] = temp_image(wid, len, 0, img_k)/255.0;
							if(normalize){
								image[pixel_cnt] = (image[pixel_cnt] - norm_mean[img_k])/norm_std[img_k];
							}
							pixel_cnt ++;
						}
					}
				}
				folder_img_cnt ++;

				//cout<<full_dir<<endl;
				//fs.open(full_dir.c_str());
				total_image_cnt ++;

				if(total_image_cnt>=total_img_num){
					if(write_to_file){
					    ofstream myfile ((save_file_name));
					    if (myfile.is_open()){
					    	//myfile << "This is a new test\n";
					    	for(int i=0; i < pixel_cnt; i++){
					    		//printf("_%f_", log_v_host[i]);
					    		myfile << image[i] << ", ";
					    	}
					    	myfile.close();
					    }
					}
					return;
				}
			}
		}

	}
}

int IBM_DVS128_event_based(string file_name, Event_Camera_Input *events, int total_event, int total_img_num){

	//int total_event_num = total_img_num *input_image_w*input_image_l*input_image_channel;

	ifstream file (file_name);

	string val;
	//cout<<"max_string_size: "<<val.max_size()<<endl;

	int i = 0;
	int total_line = 0;
	while(getline(file, val)){
		//cout<<"reading a new line"<<endl;
		total_line ++;
		stringstream ss(val);
		while(ss.good()){
			string one_val;
			getline(ss, one_val, ',');
			events[i].loc_x = stoi(one_val);
			//cout<<'|'<<one_val<<" "<<events[i].loc_x<<'|';
			if (!ss.good()){
				//cout<<endl<<"end"<<endl;
				break;
			}

			getline(ss, one_val, ',');
			events[i].loc_y = stoi(one_val);
			//cout<<'|'<<one_val<<" "<<events[i].loc_y<<'|';
			if (!ss.good()){
				//cout<<endl<<"end"<<endl;
				break;
			}
			if(events[i].loc_x>input_image_w||events[i].loc_y>input_image_l) printf("___warning___reading data error");

			getline(ss, one_val, ',');
			events[i].time = stoul(one_val);
			//cout<<'|'<<one_val<<" "<<events[i].time<<'|';
			if (!ss.good()){
				//cout<<endl<<"end"<<endl;
				break;
			}

			getline(ss, one_val, ',');
			events[i].sign = stoi(one_val);
			events[i].valid = True;
			//cout<<'|'<<one_val<<" "<<events[i].sign<<'|';
			i ++;
			if (!ss.good()){
				//cout<<endl<<"end"<<endl;
				break;
			}
		}
	}


	cout<<"Total events read: "<<i<<", total line: "<<total_line<<endl;

//	for(int j=0; j<i+1; j++){
//		if (events[j].valid==False) {
//			//cout<<"|invalid event== no. "<<i<<" data: "<<events[i].loc_x<<" ,"<<events[i].loc_y<<" ,"<<events[i].time<<" ,"<<events[i].sign<<"|";
//			break;
//		}
//		cout<<"|event no. "<<j<<" data: "<<events[j].loc_x<<" ,"<<events[j].loc_y<<" ,"<<events[j].time<<" ,"<<events[j].sign<<"|";
//	}


	return i;

}

int IBM_DVS128_event_based_one_line(string file_name, Event_Camera_Input *events, int this_file_total_line, int target_line){

	//int total_event_num = total_img_num *input_image_w*input_image_l*input_image_channel;
	if (target_line>=this_file_total_line){
		cout<<"error: target line too big"<<endl;
		return 0;
	}
	ifstream file (file_name);

	string val;
	//cout<<"max_string_size: "<<val.max_size()<<endl;

	int i = 0;

    for(int i=0; i <= target_line - 1; ++i){
        file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    }

	if(getline(file, val)){
		//cout<<"reading a new line"<<endl;

		stringstream ss(val);
		while(ss.good()){
			string one_val;
			getline(ss, one_val, ',');
			events[i].loc_x = stoi(one_val);
			//cout<<'|'<<one_val<<" "<<events[i].loc_x<<'|';
			if (!ss.good()){
				//cout<<endl<<"end"<<endl;
				break;
			}

			getline(ss, one_val, ',');
			events[i].loc_y = stoi(one_val);
			//cout<<'|'<<one_val<<" "<<events[i].loc_y<<'|';
			if (!ss.good()){
				//cout<<endl<<"end"<<endl;
				break;
			}
			if(events[i].loc_x>input_image_w||events[i].loc_y>input_image_l) printf("___warning___reading data error");

			getline(ss, one_val, ',');
			events[i].time = stoul(one_val);
			//cout<<'|'<<one_val<<" "<<events[i].time<<'|';
			if (!ss.good()){
				//cout<<endl<<"end"<<endl;
				break;
			}

			getline(ss, one_val, ',');
			events[i].sign = stoi(one_val);
			events[i].valid = True;
			//cout<<'|'<<one_val<<" "<<events[i].sign<<'|';
			i ++;
			if (!ss.good()){
				//cout<<endl<<"end"<<endl;
				break;
			}
		}
	}else{
		cout<<"error: target line unreachable"<<endl;
		return 0;
	}

	//cout<<"Total events read: "<<i<<endl;

//	for(int j=0; j<i+1; j++){
//		if (events[j].valid==False) {
//			//cout<<"|invalid event== no. "<<i<<" data: "<<events[i].loc_x<<" ,"<<events[i].loc_y<<" ,"<<events[i].time<<" ,"<<events[i].sign<<"|";
//			break;
//		}
//		cout<<"|event no. "<<j<<" data: "<<events[j].loc_x<<" ,"<<events[j].loc_y<<" ,"<<events[j].time<<" ,"<<events[j].sign<<"|";
//	}
	return i;
}

int IBM_DVS128_event_based_count_line(string file_name){

	//int total_event_num = total_img_num *input_image_w*input_image_l*input_image_channel;
	ifstream file (file_name);
	string val;
	int total_line = 0;
	while(getline(file, val)){
		total_line ++;
	}
	return total_line;
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
	cout<<"reading images from: "<<image_file<<endl;
	bool if_normalize = false;

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



	if(m_cols!=input_image_w || m_rows!=input_image_l){
		cout<<"=====Warning======: input image size mismatch"<<endl;
		cout<<"File rows: "<<m_rows<<" file col: "<<m_cols<<endl;
		//return;
	}
	m_cols=input_image_w; m_rows=input_image_l;
	//cout<<"num of col: "<<to_string(m_cols)<<"\n";
	//cout<<"num of row: "<<to_string(m_rows)<<'\n';
	char* q = new char[m_rows * m_cols];
	int total_pixel = m_rows * m_cols;
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
	                    	return;
	                    }
	                    image[index] = float(temp)/(255.0);
	                    //image[index] = 0;
	                    //image[0] = 1;
	                    if (if_normalize){
	                    	image[index] = (image[index]-0.1307)/0.3081;

	                    	//if(image[index]<0) image[index]=0;
	                    }

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

	 int check_data = 1;
	 if(check_data){
		cout<<"output one sample read data..."<<endl;
		int img_to_check = 0;
		cimg_library::CImg<unsigned char> test_image("color_mid.jpg");
		test_image.resize(input_image_w, input_image_l);
		int img_k, img_i, img_j;
		for(img_k=0;img_k<3;img_k++){
			for (img_i=0;img_i<input_image_w;img_i++){
				for (img_j=0;img_j<input_image_l;img_j++){
					int count = img_i*input_image_l + img_j + img_to_check*input_image_w*input_image_l;

					test_image(img_i, img_j, 0, img_k) = int(image[count]*255);
				}
			}
		}
		string out_file_name = "mnist_read_test.jpg";
		test_image.save(out_file_name.c_str());
	 }

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


void DVS_read_image_8bit(string image_file, float *image , int num){
	cout<<"reading images from: "<<image_file<<endl;
	bool if_normalize = false;

	ifstream ifs(image_file.c_str(), std::ios::in | std::ios::binary);
	char p[4];


	ifs.read(p, 4);
	int m_cols = to_int(p);

	ifs.read(p, 4);
	int m_rows = to_int(p);

	ifs.read(p, 4);
	int magic_number = to_int(p);
	assert(magic_number == input_image_channel);

	ifs.read(p, 4);
	int m_size = to_int(p);
	//printf("m_size_is:%d\n",m_size);
	// limit

	if (num != 0 && num < m_size) m_size = num;


	if(m_cols!=input_image_w || m_rows!=input_image_l){
		cout<<"=====Warning======: input image size mismatch"<<endl;
		cout<<"File rows: "<<m_rows<<" file col: "<<m_cols<<endl;
		//return;
	}
	//m_cols=input_image_w; m_rows=input_image_l;
	cout<<"num of col: "<<to_string(m_cols)<<"\n";
	cout<<"num of row: "<<to_string(m_rows)<<'\n';
	cout<<"num of frames: "<<to_string(m_size)<<'\n';
	char* q = new char[m_rows * m_cols];
	int total_pixel = m_rows * m_cols * input_image_channel;
	char temp = 0;

	for (int i=0; i<m_size; ++i) {
		//ifs.read(q, m_rows * m_cols);
	    //std::vector<double> image(m_rows * m_cols);

	    //====
		for(int ii=0; ii<input_image_channel; ii++){
			for(int r=0;r<m_rows;++r)
					{
						//cout << '\t' << '[' ;
						for(int c=0;c<m_cols;++c)
						{
							long index = c + r*(m_cols) + ii*m_cols*m_rows + i*(total_pixel);
		                    unsigned char temp = 0;
		                    if ( ! ifs.read((char*)&temp,sizeof(temp))){
		                    	cout<< "error opening file";
		                    	return;
		                    }
		                    image[index] = temp;
//		                    cout<<image[index];
							//int shift_bit = 7-index%8;
							//if (index%8==0){
								//if ( ! ifs.read((char*)&temp,sizeof(temp))){
							//	if (! ifs.get(temp)){
							//		cout<< "error opening file";
							//		return;
							//	}
							//}
							//cout<<index<<" "<<index%8<<"|";
							//for (int bb=7; bb>=0; bb--){
							//cout<<((temp>>shift_bit)&1);
							//cout<<shift_bit;
							//cout<<int((temp & (1 << shift_bit)));
							//cout<<" ";
							//}
							//image[index] = float(temp);
							//image[index] = 0;
							//image[0] = 1;

							//cout << unsigned(temp) << ' ' << image[index] << ' '<< '|';
							//cout << ((temp == 0.0)? ' ' : '*');
						}
						//cout << ']' << endl;
					}

		}


	 }

	 delete[] q;

	 ifs.close();

	 int check_data = 1;
	 if(check_data){
		cout<<"output one sample read data..."<<endl;
		int img_to_check = 0;
		cimg_library::CImg<unsigned char> test_image("color_mid.jpg");
		test_image.resize(input_image_w, input_image_l);
		int img_k, img_i, img_j;
		for(img_k=0;img_k<3;img_k++){
			for (img_i=0;img_i<input_image_w;img_i++){
				for (img_j=0;img_j<input_image_l;img_j++){
					int count = img_i*input_image_l + img_j + img_to_check*input_image_w*input_image_l;

					test_image(img_i, img_j, 0, img_k) = int(image[count]*255);
				}
			}
		}
		string out_file_name = "mnist_read_test.jpg";
		test_image.save(out_file_name.c_str());
	 }

}


void NTU_skeleton_read_image(string image_file, float *image , int num, int offset){
	cout<<"reading images from: "<<image_file<<endl;
	bool if_normalize = false;

	ifstream ifs(image_file.c_str(), std::ios::in | std::ios::binary);
	char p[4];


	ifs.read(p, 4);
	int m_cols = to_int(p);

	ifs.read(p, 4);
	int m_rows = to_int(p);

	ifs.read(p, 4);
	int magic_number = to_int(p);
	printf("magic_number: %d", magic_number);
	assert(magic_number == input_image_channel);

	ifs.read(p, 4);
	int m_size = to_int(p);
	printf("m_size_is:%d\n",m_size);
	// limit

	if (num != 0 && num < m_size) m_size = num;


	if(m_cols!=input_image_w || m_rows!=input_image_l){
		cout<<"=====Warning======: input image size mismatch"<<endl;
		cout<<"File rows: "<<m_rows<<" file col: "<<m_cols<<endl;
		//return;
	}
	//m_cols=input_image_w; m_rows=input_image_l;
	cout<<"num of col: "<<to_string(m_cols)<<"\n";
	cout<<"num of row: "<<to_string(m_rows)<<'\n';
	cout<<"num of frames: "<<to_string(m_size)<<'\n';
	char* q = new char[m_rows * m_cols];
	int total_pixel = m_rows * m_cols * input_image_channel;
	long long pixel_offset = offset*total_pixel;
	//char temp = 0;
	long long offset_counter = 0;
	unsigned char temp = 0;
	long index = 0;
	long max_read = total_pixel*m_size;
	while (ifs.read((char*)&temp,sizeof(temp))){
		//unsigned char temp = 0;
		for (int bb=0; bb<8; bb++){
			int bit = (temp >> bb)&1;
			//if(bit) cout<<index<<" ";
			//cout<<shift_bit;
			//cout<<int((temp & (1 << shift_bit)));
			//cout<<" ";
			if (offset_counter>=pixel_offset){
				image[index] = bit;
				index ++;
			}
			offset_counter++;
			if (index>=max_read) break;
		}
		if (index>=max_read) break;
	}

	//cout<<image[index];
	//int shift_bit = 7-index%8;
	//if (index%8==0){
	//if ( ! ifs.read((char*)&temp,sizeof(temp))){
	//	if (! ifs.get(temp)){
	//		cout<< "error opening file";
	//		return;
	//	}
	//}
	//cout<<index<<" "<<index%8<<"|";

	//image[index] = float(temp);
	//image[index] = 0;
	//image[0] = 1;

	//cout << unsigned(temp) << ' ' << image[index] << ' '<< '|';
	//cout << ((temp == 0.0)? ' ' : '*');

	//cout << ']' << endl;


	 delete[] q;

	 ifs.close();

	 int check_data = 1;
	 if(check_data){
		cout<<"output one sample read data..."<<endl;
		int img_to_check = 1;
		cimg_library::CImg<unsigned char> test_image("color_mid.jpg");
		test_image.resize(input_image_w, input_image_l);
		int img_k, img_i, img_j;
		for(img_k=0;img_k<3;img_k++){
			for (img_i=0;img_i<input_image_w;img_i++){
				for (img_j=0;img_j<input_image_l;img_j++){
					int count = img_i*input_image_l + img_j + img_to_check*input_image_w*input_image_l;

					test_image(img_i, img_j, 0, img_k) = int(image[count]*255);
				}
			}
		}
		string out_file_name = "mnist_read_test.jpg";
		test_image.save(out_file_name.c_str());
	 }

}

void KAIST_PED_read_image(string image_path, float *image , int num){
	bool load_from_csv = False;
	bool write_to_file = False;

	string save_file_name = "KAISD_PED_DATA.csv";

	if(load_from_csv){
		int total_pixel_num = num *input_image_w*input_image_l*input_image_channel;

		ifstream file (save_file_name);
		string val;
		for(int i=0; i<total_pixel_num; i++){
			if (file.good()){
				getline(file, val, ',');
				image[i] = stof(val);
				//cout<<image[i]<<" ";
			}else{
				cout<<"loading from saved image file not working"<<endl;
				break;
			}
		}
		bool check_loaded = true;
		if(check_loaded){
			int i = 0;
			cimg_library::CImg<unsigned char> temp_image("Logo-Free.jpg");
			temp_image.resize(input_image_w, input_image_l);
			for(int img_k=0;img_k<input_image_channel;img_k++){
				for (int wid=0; wid<input_image_w; wid++){
					for (int len=0; len<input_image_l; len++){
						temp_image(wid, len, 0, img_k) = int(255*image[i]);
						i++;
					}
				}
			}
			temp_image.save("test.jpg");
			i += input_image_w*input_image_l*input_image_channel*5000;
			for(int img_k=0;img_k<input_image_channel;img_k++){
				for (int wid=0; wid<input_image_w; wid++){
					for (int len=0; len<input_image_l; len++){
						temp_image(wid, len, 0, img_k) = int(255*image[i]);
						i++;
					}
				}
			}
			temp_image.save("test_2.jpg");

		}

		return; 
	}

	bool normalize = True;
	float norm_mean [3] = {0.4914, 0.4822, 0.4465};
	float norm_std [3] = {0.2023, 0.1994, 0.2010};

	namespace fs = boost::filesystem;
	int pixel_cnt = 0;
	int total_image_cnt = 0;
	int start_idx = 0;
	string cur_dir = "/hdd3/KAIST_PED/KAIST_PED/set00/V000/visible";  //2245
	int pixel_num = input_image_w*input_image_l*input_image_channel;
	//ifstream fs(full_dir.c_str());
	for (int path_i=0; path_i<10; path_i++){
		int folder_img_cnt = 0;
		if (path_i==1) cur_dir = "/hdd3/KAIST_PED/KAIST_PED/set09/V000/visible"; //3500
		if (path_i==2) cur_dir = "/hdd3/KAIST_PED/KAIST_PED/set01/V000/visible"; //2254
		if (path_i==3) cur_dir = "/hdd3/KAIST_PED/KAIST_PED/set03/V000/visible"; //4400

		if (path_i==4) cur_dir = "/hdd3/KAIST_PED/KAIST_PED/set10/V000/visible"; //4708
		if (path_i==5) cur_dir = "/hdd3/KAIST_PED/KAIST_PED/set00/V001/visible"; //2799
		if (path_i==6) cur_dir = "/hdd3/KAIST_PED/KAIST_PED/set10/V001/visible"; //4200
		if (path_i==7) cur_dir = "/hdd3/KAIST_PED/KAIST_PED/set00/V001/visible"; //2799
		if (path_i==8) cur_dir = "/hdd3/KAIST_PED/KAIST_PED/set03/V001/visible"; //2299
		if (path_i==9) cur_dir = "/hdd3/KAIST_PED/KAIST_PED/set01/V001/visible"; //2299

		fs::path Path(cur_dir);
		fs::directory_iterator end_iter;
		for (fs::directory_iterator iter(Path); iter != end_iter; ++iter){
			if(iter->path().extension() == ".jpg"){
				string img_name_temp = "00000" + to_string(folder_img_cnt);
								int cut_ind = img_name_temp.length()-5;
								string full_dir = cur_dir + "/I" + img_name_temp.substr(cut_ind, 5) + ".jpg";
								//cout<<full_dir<<endl;
								if (total_image_cnt%1000==0) cout<<to_string(total_image_cnt)<<' ';
								if (!(access( full_dir.c_str(), F_OK ) != -1)) cout<<"Wrong file name!"<<endl;
								cimg_library::CImg<unsigned char> temp_image(full_dir.c_str());
								temp_image.resize(input_image_w, input_image_l);
								//temp_image.save("test.jpg");
								//cout<<"check this pixel:"<<to_string(temp_image(200, 300, 0, 0))<<endl;
								for(int img_k=0;img_k<input_image_channel;img_k++){
									for (int wid=0; wid<input_image_w; wid++){
										for (int len=0; len<input_image_l; len++){
											image[pixel_cnt] = temp_image(wid, len, 0, img_k)/255.0;
											if(normalize){
												image[pixel_cnt] = (image[pixel_cnt] - norm_mean[img_k])/norm_std[img_k];
											}
											pixel_cnt ++;
										}
									}
								}
								folder_img_cnt ++;

								//cout<<full_dir<<endl;
								//fs.open(full_dir.c_str());
								total_image_cnt ++;

								if(total_image_cnt>=num){
									if(write_to_file){
									    ofstream myfile ((save_file_name));
									    if (myfile.is_open()){
									    	//myfile << "This is a new test\n";
									    	for(int i=0; i < pixel_cnt; i++){
									    		//printf("_%f_", log_v_host[i]);
									    		myfile << image[i] << ", ";
									    	}
									    	myfile.close();
									    }
									}
									return;
								}
			}
		}

	}



}

void read_sine_seq(string image_file, float *image, int num){
	ifstream fin(image_file);
	//fin.open(image_file, ios::in);
	int inx = 0;
	string word;
    while (fin.good()) {
        getline(fin, word, ',');
		int temp_word = stoi(word);
		image[inx] = temp_word;
		//cout<<temp_word<<", ";
		inx += 1;

    }
}

void imageNET_read_image(string folder_to_read, float *image , int num){
	bool load_from_csv = False;
	bool write_to_file = False;
	string save_file_name = "imageNET_DATA.csv";
	cout<<"Reading imageNet data"<<endl;

	if(load_from_csv){
		int total_pixel_num = num *input_image_w*input_image_l*input_image_channel;

		ifstream file (save_file_name);
		string val;
		for(int i=0; i<total_pixel_num; i++){
			if (file.good()){
				getline(file, val, ',');
				image[i] = stof(val);
				//cout<<image[i]<<" ";
			}else{
				cout<<"loading from saved image file not working"<<endl;
				break;
			}
		}
		bool check_loaded = True;
		if(check_loaded){
			int i = 0;
			cimg_library::CImg<unsigned char> temp_image("Logo-Free.jpg");
			temp_image.resize(input_image_w, input_image_l);
			for(int img_k=0;img_k<input_image_channel;img_k++){
				for (int wid=0; wid<input_image_w; wid++){
					for (int len=0; len<input_image_l; len++){
						temp_image(wid, len, 0, img_k) = int(255*image[i]);
						i++;
					}
				}
			}
			temp_image.save("test.jpg");
			i += input_image_w*input_image_l*input_image_channel*5000;
			for(int img_k=0;img_k<input_image_channel;img_k++){
				for (int wid=0; wid<input_image_w; wid++){
					for (int len=0; len<input_image_l; len++){
						temp_image(wid, len, 0, img_k) = int(255*image[i]);
						i++;
					}
				}
			}
			temp_image.save("test_2.jpg");

		}

		return;
	}


	bool normalize = True;
	float norm_mean [3] = {0.485, 0.456, 0.406};
	float norm_std [3] = {0.229, 0.224, 0.225};

	namespace fs = boost::filesystem;
	int pixel_cnt = 0;
	int total_image_cnt = 0;
	int start_idx = 0;
	string cur_dir = "/data_hdd/data/imagenet/raw-data/train";  //2245
	int pixel_num = input_image_w*input_image_l*input_image_channel;
	//ifstream fs(full_dir.c_str());

	float input_w_l_ratio = input_image_w/input_image_l;

	int folder_img_cnt = 0;
	cur_dir = "/data_hdd/data/imagenet/raw-data/train/" + folder_to_read;
//	cout<<"Reading data from: "<<cur_dir<<endl;
	fs::path Path(cur_dir);
	fs::directory_iterator end_iter;
	int folder_image_cnt = 0;
	for (fs::directory_iterator iter(Path); iter != end_iter; ++iter){

//			cout<<iter->path()<<endl;

		if(folder_image_cnt>100){
			break;
		}

		if(iter->path().extension() == ".JPEG"){
			string full_dir = iter->path().string();
//								cout<<full_dir<<endl;
//			if (total_image_cnt%1000==0) cout<<to_string(total_image_cnt)<<' images read';
			if (!(access( full_dir.c_str(), F_OK ) != -1)) cout<<"Wrong file name!"<<endl;
			cimg_library::CImg<unsigned char> temp_image(full_dir.c_str());
			int x0,x1,y0,y1;
			float image_w_l_ratio = (temp_image.width()+0.0)/temp_image.height();
			if (image_w_l_ratio>=input_w_l_ratio){
				int pixel_to_crop = (image_w_l_ratio - input_w_l_ratio)*temp_image.height()/2;
				x0 = pixel_to_crop;
				x1 = temp_image.width() - pixel_to_crop;
				y0 = 0;
				y1 = temp_image.height();
			}else{
				int pixel_to_crop = (input_w_l_ratio - image_w_l_ratio)*temp_image.height()/2;
				y0 = pixel_to_crop;
				y1 = temp_image.height() - pixel_to_crop;
				x0 = 0;
				x1 = temp_image.width();
			}
			temp_image.crop(x0, y0, x1, y1);
			temp_image.resize(input_image_w, input_image_l);
//								temp_image.save("test.jpg");
			//cout<<"check this pixel:"<<to_string(temp_image(200, 300, 0, 0))<<endl;
			for(int img_k=0;img_k<input_image_channel;img_k++){
				for (int wid=0; wid<input_image_w; wid++){
					for (int len=0; len<input_image_l; len++){
						image[pixel_cnt] = temp_image(wid, len, 0, img_k)/255.0;
						if(normalize){
							image[pixel_cnt] = (image[pixel_cnt] - norm_mean[img_k])/norm_std[img_k];
						}
						pixel_cnt ++;
					}
				}
			}
			folder_image_cnt ++;
			folder_img_cnt ++;

			//cout<<full_dir<<endl;
			//fs.open(full_dir.c_str());
			total_image_cnt ++;

			if(total_image_cnt>=num){
				if(write_to_file){
					ofstream myfile ((save_file_name));
					if (myfile.is_open()){
						//myfile << "This is a new test\n";
						for(int i=0; i < pixel_cnt; i++){
							//printf("_%f_", log_v_host[i]);
							myfile << image[i] << ", ";
						}
						myfile.close();
					}
				}
				return;
			}
		}
	}
}


void read_polygon(string folder_to_read, float *image, int num){
	namespace fs = boost::filesystem;
	int pixel_cnt = 0;
	int total_image_cnt = 0;
	int start_idx = 0;
	string cur_dir = "/home/xshe6/Documents/CUDA/Spike_CNN/Debug";  //2245
	int pixel_num = input_image_w*input_image_l*input_image_channel;
	//ifstream fs(full_dir.c_str());

	float input_w_l_ratio = input_image_w/input_image_l;

	cur_dir += folder_to_read;
	cout<<"Reading data from: "<<cur_dir<<", total num: "<<num<<endl;
	fs::path Path(cur_dir);

	typedef vector<fs::path> vec;
	vec v;
	copy(fs::directory_iterator(Path), fs::directory_iterator(),  back_inserter(v));
	sort(v.begin(), v.end());
	cout<<"path vector size: "<<v.size()<<endl;

	fs::directory_iterator end_iter;
	int folder_image_cnt = 0;
	for(vec::const_iterator iter=v.begin();iter!=v.end(); ++iter){

//			cout<<iter->path()<<endl;


		fs::path temp = *iter;

		//cout<<temp.string()<<endl;
		if(temp.extension() == ".png"){
			string full_dir = temp.string();
			//cout<<"Reading images from "<<full_dir<<endl;
//			if (total_image_cnt%1000==0) cout<<to_string(total_image_cnt)<<' images read';
			if (!(access( full_dir.c_str(), F_OK ) != -1)) cout<<"Wrong file name!"<<endl;
			cimg_library::CImg<unsigned char> temp_image(full_dir.c_str());
			temp_image.resize(input_image_w, input_image_l);
//								temp_image.save("test.jpg");
			//cout<<"check this pixel:"<<to_string(temp_image(200, 300, 0, 0))<<endl;
			for(int img_k=0;img_k<input_image_channel;img_k++){
				for (int wid=0; wid<input_image_w; wid++){
					for (int len=0; len<input_image_l; len++){
						image[pixel_cnt] = temp_image(wid, len, 0, img_k)/255.0;
						pixel_cnt ++;
					}
				}
			}
			folder_image_cnt ++;

			//cout<<full_dir<<endl;
			//fs.open(full_dir.c_str());
			total_image_cnt ++;
			//cout<<total_image_cnt<<endl;
			if(total_image_cnt>=num){
				cout<<"Ending from overflowing data memory"<<endl;
				return;
			}
		}
	}

}

void read_one_image(string dir_to_read, float *image, int num){
	namespace fs = boost::filesystem;
	int pixel_cnt = 0;
	int total_image_cnt = 0;
	int start_idx = 0;
	string cur_dir = "/home/xshe6/Documents/CUDA/Spike_CNN/Debug";  //2245
	int pixel_num = input_image_w*input_image_l*input_image_channel;
	//ifstream fs(full_dir.c_str());

	float input_w_l_ratio = input_image_w/input_image_l;

	//cur_dir += folder_to_read;
	cout<<"Reading data from: "<<dir_to_read<<endl;
	//fs::path Path(dir_to_read);
	//fs::directory_iterator end_iter;
	string full_dir = dir_to_read;
	//cout<<"Reading images from "<<full_dir<<endl;

	if (!(access( full_dir.c_str(), F_OK ) != -1)) cout<<"Wrong file name!"<<endl;
	cimg_library::CImg<unsigned char> temp_image(full_dir.c_str());
	temp_image.resize(input_image_w, input_image_l);
//								temp_image.save("test.jpg");
	//cout<<"check this pixel:"<<to_string(temp_image(200, 300, 0, 0))<<endl;
	for(int img_k=0;img_k<input_image_channel;img_k++){
		for (int wid=0; wid<input_image_w; wid++){
			for (int len=0; len<input_image_l; len++){
				image[pixel_cnt] = temp_image(wid, len, 0, img_k)/255.0;
				pixel_cnt ++;
			}
		}
	}

}

void read_sc2(string image_file, float *image, int num){


	ifstream fin(image_file);
	//fin.open(image_file, ios::in);
	int inx = 0;
	string one_line;
    while (std::getline(fin, one_line)) {

    	std::string delimiter = ",";

    	size_t pos = 0;
    	std::string token;
    	while ((pos = one_line.find(delimiter)) != std::string::npos) {
    	    token = one_line.substr(0, pos);
//    	    std::cout << token << " ";
    		int temp_word = stoi(token);
    		image[inx] = temp_word;
    		inx += 1;
    	    one_line.erase(0, pos + delimiter.length());
    	    if(inx>=num-input_image_channel) {
    	    	image[input_image_channel] = -1;
    	    	return;
    	    }

    	}

//    	std::cout << std::endl;
    }
    for(int i=0; i<input_image_channel; i++) image[inx+i]=-1;
}

void read_sc2_2(string image_file, float *image, int input_neuron_num){
//this reading in: building type as channels, time sequence as x,y

	ifstream fin(image_file);
	//fin.open(image_file, ios::in);

	int line_cnt = 0;
	string one_line;
    while (std::getline(fin, one_line)) {

    	std::string delimiter = ",";

    	size_t pos = 0;
    	std::string token;
    	int inx = 0;
    	while ((pos = one_line.find(delimiter)) != std::string::npos) {
    	    token = one_line.substr(0, pos);
//    	    std::cout << token << " ";
    		int temp_word = stoi(token);
    		if((line_cnt+input_image_w*input_image_l*inx)<input_neuron_num) image[line_cnt+input_image_w*input_image_l*inx] = 3*temp_word;
    		inx += 1;
    	    one_line.erase(0, pos + delimiter.length());
    	}
    	line_cnt++;
//    	std::cout << std::endl;
    }

}


void read_sc2_3(string image_file, float *image, int input_neuron_num){
//this reading in: building type as channels, time sequence as x,y

	ifstream fin(image_file);
	//fin.open(image_file, ios::in);

	int index_cnt = 0;
	string one_line;
    while (std::getline(fin, one_line)) {

    	std::string delimiter = ",";

    	size_t pos = 0;
    	std::string token;

    	while ((pos = one_line.find(delimiter)) != std::string::npos) {
    	    token = one_line.substr(0, pos);
//    	    std::cout << token << " ";
    		int temp_word = stoi(token);
    		if(index_cnt<input_neuron_num){
    			image[index_cnt] = 6*temp_word;
    		}else{
    			return;
    		}
    		index_cnt += 1;
    	    one_line.erase(0, pos + delimiter.length());
    	}


    }

}


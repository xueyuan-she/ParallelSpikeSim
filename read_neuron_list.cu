#include "header.h"
#include <iostream>
#include <string>
#include <fstream>
#include<stdlib.h>

int reset_weight(Neuron *NeuronList, float start_depth, float end_depth, int reset_method, int network_size){
	cout<<"resetting weight, start depth: "<<start_depth<<" end depth: "<<end_depth<<endl;
	if(reset_method==1){//within this depth, reset all weight to 0.5



		for (int index=0; index<network_size; index ++){
			if((NeuronList[index].param[7]<start_depth||NeuronList[index].param[7]>end_depth)){
				continue;
			}
			for (int connection_index=0; connection_index<MAX_CONNECTION; connection_index++){
				if(NeuronList[index].connected_in[connection_index] > 0.1){

					NeuronList[index].connected_weight[connection_index] = 0.5;
				}
			}


		}
	}
	return 0;

}


int normalize_weight(Neuron *NeuronList, float start_depth, float end_depth, int norm_method, int network_size){
	cout<<"normalizing weight, start depth: "<<start_depth<<" end depth: "<<end_depth<<endl;
	if(norm_method==1){//within this depth, normalize all weight to 0 mean unit variance



		for (int index=0; index<network_size; index ++){

			float mean = 0;
			float sum = 0;
			float std = 0;
			int valid_connection = 0;
			if((NeuronList[index].param[7]<start_depth||NeuronList[index].param[7]>end_depth)){
				continue;
			}
			for (int connection_index=0; connection_index<MAX_CONNECTION; connection_index++){
				if(NeuronList[index].connected_in[connection_index] > 0.1){
					sum += NeuronList[index].connected_weight[connection_index];
					valid_connection ++;
				}
			}
			mean = sum/valid_connection;

			for (int connection_index=0; connection_index<MAX_CONNECTION; connection_index++){
				if(NeuronList[index].connected_in[connection_index] > 0.1){
					std += (NeuronList[index].connected_weight[connection_index]-mean)*(NeuronList[index].connected_weight[connection_index]-mean);
				}
			}
			std /= valid_connection;
			std = sqrt(std);
			cout<<"Index "<<index<<" connection size: "<<valid_connection<<" mean: "<<mean<<" std: "<<std<<endl;
			for (int connection_index=0; connection_index<MAX_CONNECTION; connection_index++){
				if(NeuronList[index].connected_in[connection_index] > 0.1){

					NeuronList[index].connected_weight[connection_index] = (NeuronList[index].connected_weight[connection_index]-mean)/std;
				}
			}


		}

		for (int index=0; index<network_size; index ++){

			float mean = 0;
			float sum = 0;
			float std = 0;
			int valid_connection = 0;
			if((NeuronList[index].param[7]<start_depth||NeuronList[index].param[7]>end_depth)){
				continue;
			}
			for (int connection_index=0; connection_index<MAX_CONNECTION; connection_index++){
				if(NeuronList[index].connected_in[connection_index] > 0.1){
					sum += NeuronList[index].connected_weight[connection_index];
					valid_connection ++;
				}
			}
			mean = sum/valid_connection;

			for (int connection_index=0; connection_index<MAX_CONNECTION; connection_index++){
				if(NeuronList[index].connected_in[connection_index] > 0.1){
					std += (NeuronList[index].connected_weight[connection_index]-mean)*(NeuronList[index].connected_weight[connection_index]-mean);
				}
			}
			std /= valid_connection;
			std = sqrt(std);
			cout<<"Index "<<index<<" connection size: "<<valid_connection<<" mean: "<<mean<<" std: "<<std<<endl;


		}

	}
	return 0;

}
int read_neuron_list(Neuron *NeuronList, int neuron_type, string file_name){
	int i = 0;//number of neurons
	int j = 0;//number of connected_in neurons
	int k = 0;
	std::string item;
	ifstream file_read;
	file_read.open(file_name.c_str());
	//printf("==0==\n");
	int flag_0 = 0;
	int flag_1 = 0;
	int flag_2 = 0;
	int line_count = 0;
	int param_num = 8;
	int state_num = 8;
	while(file_read >> item){
		//printf("ring,");
		//line_count ++;
		//j = 0;
		if(strcmp(item.c_str(), " ") == 0){

		}
		else if(strcmp(item.c_str(), ";") == 0){
			flag_1 = 1;
			flag_0 = 0;
			j = 0;
		}
		else if(strcmp(item.c_str(), "|") == 0){
			flag_1 = 2;
			flag_0 = 0;
			k = 0;
		}
		else if(strcmp(item.c_str(), ".") == 0){
			flag_1 = 0;
			NeuronList[i].connected_in[j+1] = 0;
			line_count++;
//			cout<<"_index_"<<i<<"-";
//			cout<<NeuronList[i].index<<"-";
//			printf("%d|", NeuronList[i].type);
			i++;
		}
		else if (flag_1 == 0){
			//printf("rIZH.");
			switch (flag_0){
				case 0: NeuronList[i].index = atoi(item.c_str());
						//printf("-%d", NeuronList[i].index);

				break;
				case 1: NeuronList[i].type = atoi(item.c_str());

				break;
//				case 2: NeuronList[i].param[0] = strtof(item.c_str(), NULL);
//				break;
//				case 3: NeuronList[i].state[0] = strtof(item.c_str(), NULL);
//				break;
			}

			if(flag_0<=param_num+1&&flag_0>1){
				NeuronList[i].param[flag_0-2] = strtof(item.c_str(), NULL);
			}else if(flag_0>param_num+1){
				NeuronList[i].state[flag_0-(param_num+2)] = strtof(item.c_str(), NULL);
			}
			flag_0 ++;
		}
		else if (flag_1 == 1){
			//printf("==3==\n");
			if (flag_2 == 0){
				NeuronList[i].connected_in[j] = atoi(item.c_str());
				//string stream = to_string(NeuronList[i].connected_in[j]);
				//stream = "Neuron No. is" + to_string(i) + ";Connected in index is: " + to_string(j) + ";Connected in is: " + to_string(NeuronList[i].connected_in[j]);
				//cout<<"= "<<stream<<" =";
				flag_2 = 1;
			}else{
				NeuronList[i].connected_weight[j] = strtof(item.c_str(), NULL);
//				printf("-%f",NeuronList[i].connected_weight[j]);
				flag_2 = 0;
				j++;
			}

			//printf("\n");
			/*
			string stream = to_string(j);
			cout<<stream;
			cout<<"connected:"<<item.c_str()<<endl;
			*/

		}
		else if (flag_1 == 2){
			NeuronList[i].local_inhibition[k] = atoi(item.c_str());
			k++;
		}



	}

	cout<<"total_line_is: "<<to_string(line_count)<<endl;

	file_read.close();

	return 0;

}

int read_neuron_list_special(Neuron *NeuronList, int duplicate_layer, CNN_struct *settings, string file_name){
	//need to increase counting on param[7](depth number)
	cout<<"Special network loading function from file: "<<file_name<<endl;

	int repeat_start_neuron_index = 0;
	int repeat_layer_total_neuron =  settings->layer[duplicate_layer].neuron_num/2;
	for(int i=0; i<duplicate_layer; i++){
		repeat_start_neuron_index += settings->layer[i].neuron_num;
	}
	int repeat_end_neuron_index = repeat_start_neuron_index + settings->layer[duplicate_layer].neuron_num/2;
	int repeat_layer_depth_num = settings->layer[duplicate_layer].depth/2;

	repeat_start_neuron_index = repeat_start_neuron_index - settings->layer[0].neuron_num;
	repeat_end_neuron_index = repeat_end_neuron_index - settings->layer[0].neuron_num;

	cout<<"Parameters: repeat_start_neuron_index-"<<repeat_start_neuron_index<<" repeat_layer_total_neuron-"<<repeat_layer_total_neuron<< \
			" repeat_end_neuron_index-"<<repeat_end_neuron_index<<" repeat_layer_depth_num-"<<repeat_layer_depth_num<<endl;

	int i = 0;//number of neurons
	int j = 0;//number of connected_in neurons
	int k = 0;
	std::string item;
	ifstream file_read;
	file_read.open(file_name.c_str());
	//printf("==0==\n");
	int flag_0 = 0;
	int flag_1 = 0;
	int flag_2 = 0;
	int line_count = 0;
	int param_num = 8;
	int state_num = 8;

	while(file_read >> item){
		//printf("ring,");
		//line_count ++;
		//j = 0;

		if(strcmp(item.c_str(), " ") == 0){

		}
		else if(strcmp(item.c_str(), ";") == 0){
			flag_1 = 1;
			flag_0 = 0;
			j = 0;
		}
		else if(strcmp(item.c_str(), "|") == 0){
			flag_1 = 2;
			flag_0 = 0;
			k = 0;
		}
		else if(strcmp(item.c_str(), ".") == 0){
			flag_1 = 0;
			NeuronList[i].connected_in[j+1] = 0;
			if(i<repeat_end_neuron_index&&i>=repeat_start_neuron_index) NeuronList[i+repeat_layer_total_neuron].connected_in[j+1] = 0;
			line_count++;
//			cout<<"_index_"<<i<<"-";
//			cout<<NeuronList[i].index<<"-";
//			printf("%d|", NeuronList[i].type);
			i++;
			if(i==repeat_end_neuron_index) i=i+repeat_layer_total_neuron;
		}
		else if (flag_1 == 0){
			//printf("rIZH.");
			switch (flag_0){
				case 0: {
					NeuronList[i].index = atoi(item.c_str());

					if(i<repeat_end_neuron_index&&i>=repeat_start_neuron_index){
						NeuronList[i+repeat_layer_total_neuron].index = NeuronList[i].index + repeat_layer_total_neuron;
					}else if(i>=repeat_end_neuron_index){
						NeuronList[i].index = NeuronList[i].index + repeat_layer_total_neuron;
					}

				}
				break;
				case 1: {
					NeuronList[i].type = atoi(item.c_str());
					if(i<repeat_end_neuron_index&&i>=repeat_start_neuron_index){
						NeuronList[i+repeat_layer_total_neuron].type = NeuronList[i].type;
					}
				}
				break;
//				case 2: NeuronList[i].param[0] = strtof(item.c_str(), NULL);
//				break;
//				case 3: NeuronList[i].state[0] = strtof(item.c_str(), NULL);
//				break;
			}

			if(flag_0<=param_num+1&&flag_0>1){
				NeuronList[i].param[flag_0-2] = strtof(item.c_str(), NULL);
				if(i<repeat_end_neuron_index&&i>=repeat_start_neuron_index){
					NeuronList[i+repeat_layer_total_neuron].param[flag_0-2] = NeuronList[i].param[flag_0-2];
					if((flag_0-2)==7) NeuronList[i+repeat_layer_total_neuron].param[flag_0-2] += repeat_layer_depth_num;
				}else if(i>=repeat_end_neuron_index){
					if((flag_0-2)==7) NeuronList[i].param[flag_0-2] += repeat_layer_depth_num;
				}

			}else if(flag_0>param_num+1){
				NeuronList[i].state[flag_0-(param_num+2)] = strtof(item.c_str(), NULL);
				if(i<repeat_end_neuron_index&&i>=repeat_start_neuron_index) NeuronList[i+repeat_layer_total_neuron].state[flag_0-(param_num+2)] = NeuronList[i].state[flag_0-(param_num+2)];
			}
			flag_0 ++;
		}
		else if (flag_1 == 1){
			//printf("==3==\n");
			if (flag_2 == 0){
				NeuronList[i].connected_in[j] = atoi(item.c_str());
				if(i<repeat_end_neuron_index&&i>=repeat_start_neuron_index) NeuronList[i+repeat_layer_total_neuron].connected_in[j] = NeuronList[i].connected_in[j];
				//string stream = to_string(NeuronList[i].connected_in[j]);
				//stream = "Neuron No. is" + to_string(i) + ";Connected in index is: " + to_string(j) + ";Connected in is: " + to_string(NeuronList[i].connected_in[j]);
				//cout<<"= "<<stream<<" =";
				flag_2 = 1;
			}else{
				NeuronList[i].connected_weight[j] = strtof(item.c_str(), NULL);
				if(i<repeat_end_neuron_index&&i>=repeat_start_neuron_index) NeuronList[i+repeat_layer_total_neuron].connected_weight[j] = NeuronList[i].connected_weight[j];
//				printf("-%f",NeuronList[i].connected_weight[j]);
				flag_2 = 0;
				j++;
			}

			//printf("\n");
			/*
			string stream = to_string(j);
			cout<<stream;
			cout<<"connected:"<<item.c_str()<<endl;
			*/

		}
		else if (flag_1 == 2){
			NeuronList[i].local_inhibition[k] = atoi(item.c_str());
			if(i<repeat_end_neuron_index&&i>=repeat_start_neuron_index) NeuronList[i+repeat_layer_total_neuron].local_inhibition[k] = NeuronList[i].local_inhibition[k];
			k++;
		}



	}

	cout<<"total_line_is: "<<to_string(line_count)<<endl;

	file_read.close();

	return 0;
	cout<<"Network Reading Done"<<endl;

}

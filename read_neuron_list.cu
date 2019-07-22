#include "header.h"
#include <iostream>
#include <string>
#include <fstream>
#include<stdlib.h>

int read_neuron_list(Neuron *NeuronList, int neuron_type, string file_name){
	int i = 0;//number of neurons
	int j = 0;//number of connected_in neurons
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



	}

	cout<<"total_line_is: "<<to_string(line_count)<<endl;

	file_read.close();

	return 0;

}

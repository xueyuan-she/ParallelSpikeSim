#include "header.h"
#include <iostream>
#include <string>
#include <fstream>
#include<stdlib.h>

using namespace std;

int write_neuron_list(Neuron *NeuronList, string file_name, int network_size){
	cout<<"Writing network to file"<<endl;
	ofstream file_write;
	file_write.open(file_name);
	//printf("==0==\n");


	for(int i=0;i<network_size;i++){
		//cout<<i<<" ";
		int para_num = 8;
		int state_num = 8;
		string each_neuron;

//		switch (NeuronList[i].type){
//			case 0: //IZH
//			para_num = 5;
//			state_num = 3;
//			break;
//			case 1://Stoch
//			para_num = 1;
//			state_num = 1;
//			break;
//			case 2://LIF
//			para_num = 6;
//			state_num = 3;
//			break;
//			case 3://HH
//			para_num = 7;
//			state_num = 4;
//			break;
//			case 4://signal
//			para_num = 1;
//			state_num = 3;
//			break;
//		}

		each_neuron = each_neuron + to_string(NeuronList[i].index) + ' ' + to_string(NeuronList[i].type) + ' ';
		//printf("-%d", NeuronList[i].type);
		for(int param_i=0;param_i<para_num;param_i++){
			each_neuron = each_neuron + to_string(NeuronList[i].param[param_i]) + ' ';
		}

		for(int state_i=0;state_i<state_num;state_i++){
			each_neuron = each_neuron + to_string(NeuronList[i].state[state_i]) + ' ';
		}
		each_neuron = each_neuron + ';' + ' ';

		int connected_in_i = 0;
		//cout << to_string(NeuronList[i].connected_in[connected_in_i])<<"\n";
		while(NeuronList[i].connected_in[connected_in_i] > 0.1){
			//cout << to_string(NeuronList[i].connected_in[connected_in_i]);
			each_neuron = each_neuron + to_string(NeuronList[i].connected_in[connected_in_i]) + ' ' + to_string(NeuronList[i].connected_weight[connected_in_i]) + ' ';
			connected_in_i ++;
		}
		//cout<<each_neuron;
		each_neuron = each_neuron + '|' + ' ';
		connected_in_i = 0;
		while(NeuronList[i].local_inhibition[connected_in_i] > 1){
			//cout << to_string(NeuronList[i].connected_in[connected_in_i]);
			each_neuron = each_neuron + to_string(NeuronList[i].local_inhibition[connected_in_i]) + ' ';
			connected_in_i ++;
		}
		//cout<<each_neuron;
		each_neuron = each_neuron + '.' + ' ';
		file_write << each_neuron << '\n';

	}

	file_write.close();
	cout<<"File write complete\n";
	return 0;

}

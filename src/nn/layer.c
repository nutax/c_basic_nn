#include "nn/layer.h"

void nn_layer_forward(float output[], const float input[], const float weight[], int inputs, int outputs, void (*act)(float*, int)){
    int i, j, neuron;
    for(i = 0; i<outputs; ++i){
        output[i] = 0;
        neuron = i*inputs;
        for(j = 0; j<inputs; ++j){
            output[i] += input[j] * weight[neuron+j];
        }
    }
    act(output, outputs);
}
void nn_layer_output_error(float output_e[], const float answer[], const float output[], int outputs){    int i;
    for(i = 0; i<outputs; ++i){
        output_e[i] = answer[i] - output[i];
    }
}
void nn_layer_input_error(float input_e[], const float output_e[], const float weight[], int inputs, int outputs){
    int i, j, neuron;
    for(i = 0; i<inputs; ++i){
        input_e[i] = 0;
    }
    for(i = 0; i<outputs; ++i){
        neuron = i*inputs;
        for(j = 0; j < inputs; ++j){
            input_e[j] += output_e[i]*weight[neuron+j];
        }
    }
}
void
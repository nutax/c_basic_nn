#include "nn/layer.h"

void nn_layer_create(float **free_mem, struct nn_layer *layer, int inputs, int outputs){
    int i;

    layer->outputs = outputs;

    layer->weight = *free_mem;
    (*free_mem) += outputs*inputs;

    layer->output = *free_mem;
    (*free_mem) += outputs;

    layer->error = *free_mem;
    (*free_mem) += outputs;

    for(i = 0; i<inputs*outputs; ++i){
        layer->weight[i] = ((float)rand())/((float)RAND_MAX);
    }
}
void nn_layer_forward(float output[], const float input[], const float weight[], int inputs, int outputs, void (*act)(float*, int)){
    int i, j, neuron;
    inputs += 1; //bias
    for(i = 0; i<outputs; ++i){
        neuron = i*inputs;
        output[i] = weight[neuron]; //bias
        for(j = 1; j<inputs; ++j){
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
void nn_layer_update(float weight[], float output[], const float input[], const float output_e[], int inputs, int outputs, float rate, void (*dact)(float*, int)){
    int i, j, neuron;
    dact(output, outputs);
    const float *output_d = output;
    for(i = 0; i<outputs; ++i){
        neuron = i*inputs;
        for(j = 0; j < inputs; ++j){
            weight[neuron+j] = input[j]*output_d[i]*rate;
        }
    }
}
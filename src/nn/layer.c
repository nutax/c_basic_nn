#include "nn/layer.h"

void nn_layer_create(float **free_mem, struct nn_layer *layer, int outputs, enum NN_LAYER_TYPE type, struct nn_layer *prev_layer, void (*act)(float*, int), void (*dact)(float*, int)){
    int i, inputs;

    layer->type = type;
    
    outputs += ((int)(type != NN_LAYER_OUTPUT) & 1);
    layer->outputs = outputs;

    if(type != NN_LAYER_INPUT){
        layer->act = act;
        layer->dact = dact;

        inputs = prev_layer->outputs;

        layer->input = prev_layer->output;
        layer->input_e = prev_layer->output_e;

        layer->weight = *free_mem;
        (*free_mem) += outputs*inputs;

        layer->output = *free_mem;
        (*free_mem) += outputs;

        layer->output_e = *free_mem;
        (*free_mem) += outputs;

        for(i = 0; i<inputs*outputs; ++i){
            layer->weight[i] = ((float)rand())/((float)RAND_MAX);
        }
    }else{
        layer->output = *free_mem;
        (*free_mem) += outputs;
    }
    layer->output[outputs-1] = 1;
}
void nn_layer_forward(struct nn_layer *layer){
    int i, j, neuron;
    
    int const inputs = layer->inputs;
    int const outputs = layer->outputs;
    int const last = outputs - 1;

    float const * const input = layer->input;
    float const * const weight = layer->weight;

    float * const output = layer->output;
    
    for(i = 0; i<last; ++i){
        neuron = i*inputs;
        output[i] = 0;
        for(j = 0; j<inputs; ++j){
            output[i] += input[j] * weight[neuron+j];
        }
    }
    if(layer->type == NN_LAYER_OUTPUT){
        neuron = last*inputs;
        for(j = 0; j<inputs; ++j){
            output[last] += input[j] * weight[neuron+j];
        }
    }
    layer->act(output, outputs);
}
void nn_layer_output_error(struct nn_layer *layer, float const answer[]){
    int i;
    int const outputs = layer->outputs;

    float const * const output = layer->output;
    float * const output_e = layer->output_e;

    for(i = 0; i<outputs; ++i){
        output_e[i] = answer[i] - output[i];
    }
}
void nn_layer_input_error(struct nn_layer *layer){
    int i, j, neuron;

    int const inputs = layer->inputs;
    int const outputs = layer->outputs;

    float const * const output_e = layer->output_e;
    float const * const weight = layer->weight;
    
    float * const input_e = layer->input_e;
    
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
void nn_layer_update(struct nn_layer *layer, int rate){
    int i, j, neuron;

    int const inputs = layer->inputs;
    int const outputs = layer->outputs;

    float const * const input = layer->input;
    
    float * const output = layer->output;
    float * const weight = layer->weight;

    layer->dact(output, outputs);
    float const * const output_d = output;

    for(i = 0; i<outputs; ++i){
        neuron = i*inputs;
        for(j = 0; j < inputs; ++j){
            weight[neuron+j] = input[j]*output_d[i]*rate;
        }
    }
}
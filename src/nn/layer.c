#include "nn/layer.h"

void nn_layer_create(float **free_mem, struct nn_layer *layer, int outputs, enum NN_LAYER_TYPE type, struct nn_layer const *prev_layer, void (*act)(float*, int), void (*dact)(float*, int)){
    int i, inputs;

    layer->type = type;
    
    outputs += ((int)(type != NN_LAYER_OUTPUT) & 1);
    layer->outputs = outputs;

    if(type != NN_LAYER_INPUT){
        layer->act = act;
        layer->dact = dact;

        inputs = prev_layer->outputs;

        layer->inputs = inputs;
        layer->input = prev_layer->output;
        layer->input_delta = prev_layer->output_delta;

        layer->weight = *free_mem;
        (*free_mem) += outputs*inputs;

        layer->output = *free_mem;
        (*free_mem) += outputs;

        layer->output_delta = *free_mem;
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
void nn_layer_output_load(struct nn_layer *layer, float const output[]){
    memcpy(layer->output, output, sizeof(float)*(layer->outputs - 1));
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
void nn_layer_output_delta(struct nn_layer *layer, float const answer[]){
    int i;
    int const outputs = layer->outputs;

    float const * const output = layer->output;
    float * const output_delta = layer->output_delta;

    memcpy(output_delta, output, sizeof(float)*outputs);
    layer->dact(output_delta, outputs);
    for(i = 0; i<outputs; ++i){
        output_delta[i] *= (answer[i] - output[i]);
    }
}
void nn_layer_input_delta(struct nn_layer *layer){
    int i, j, neuron;

    int const inputs = layer->inputs;
    int const outputs = layer->outputs;

    float const * const input = layer->input;
    float const * const output_delta = layer->output_delta;
    float const * const weight = layer->weight;

    float * const input_delta = layer->input_delta;

    float input_delta_aux[inputs];
    memset(input_delta_aux, 0, sizeof(float)*inputs);

    for(i = 0; i<outputs; ++i){
        neuron = i*inputs;
        for(j = 0; j < inputs; ++j){
            input_delta_aux[j] += output_delta[i]*weight[neuron+j];
        }
    }

    memcpy(input_delta, input, sizeof(float)*inputs);
    layer->dact(input_delta, inputs);
    for(i = 0; i < inputs; ++i){
        input_delta[i] *= input_delta_aux[i];
    }
}
void nn_layer_update(struct nn_layer *layer, float rate){
    int i, j, neuron;

    int const inputs = layer->inputs;
    int const outputs = layer->outputs;

    float const * const input = layer->input;
    float const * const output_delta = layer->output_delta;
    
    float * const weight = layer->weight;

    for(i = 0; i<outputs; ++i){
        neuron = i*inputs;
        for(j = 0; j < inputs; ++j){
            weight[neuron+j] = input[j]*output_delta[i]*rate;
        }
    }
}
void nn_layer_print(struct nn_layer *layer){
    int i;
    static char const * const spacer = " ------ ";
    int const inputs = layer->inputs;
    int const outputs = layer->outputs;
    float const * const input = layer->input;
    float const * const output = layer->output;
    float const * const input_delta = layer->input_delta;
    float const * const output_delta = layer->output_delta;

    printf("\n%sLayer: %p%s", spacer, layer, spacer);
    printf("\nInputs:  %d", inputs);
    printf("\nOutputs:  %d", outputs);
    printf("\n\nInput ptr:  %p", input);
    printf("\nInput arr:  ");
    for(i = 0; i<inputs; ++i){
        printf("%f  ", input[i]);
    }
    printf("\n\nOutput ptr:  %p", output);
    printf("\nOutput arr:  ");
    for(i = 0; i<outputs; ++i){
        printf("%f  ", output[i]);
    }
    printf("\n\nODelta ptr:  %p", output_delta);
    printf("\nODelta arr:  ");
    for(i = 0; i<outputs*((layer->type != NN_LAYER_INPUT) & 1); ++i){
        printf("%f  ", output_delta[i]);
    }
    printf("\n");
}

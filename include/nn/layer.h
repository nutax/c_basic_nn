#ifndef _NN_LAYER_H_
#define _NN_LAYER_H_

#include <stdlib.h>

enum NN_LAYER_TYPE{NN_LAYER_INPUT, NN_LAYER_HIDDEN, NN_LAYER_OUTPUT};

struct nn_layer;

void nn_layer_create(float **free_mem, struct nn_layer *layer, int outputs, enum NN_LAYER_TYPE type, struct nn_layer *prev_layer, void (*act)(float*, int), void (*dact)(float*, int));
void nn_layer_forward(struct nn_layer *layer);
void nn_layer_output_error(struct nn_layer *layer, float const answer[]);
void nn_layer_input_error(struct nn_layer *layer);
void nn_layer_update(struct nn_layer *layer, int rate);


struct nn_layer{
    enum NN_LAYER_TYPE type;
    int inputs;
    int outputs;
    void (*act)(float*, int);
    void (*dact)(float*, int);
    float *input_e;
    float *input;
    float *weight;
    float *output;
    float *output_e;
};

#endif
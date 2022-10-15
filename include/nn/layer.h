#ifndef _NN_LAYER_H_
#define _NN_LAYER_H_

void nn_layer_forward(const float x[], const float w[], float y[], int inputs, int outputs, void (*act)(float*, int));
void nn_layer_output_delta(const float ans[], const float y[], float output_d[], int outputs);
void nn_layer_input_delta(const float d[], const float w[], float input_d[], int inputs, int outputs);
void nn_layer_update(const float forward_d[], const float forward_y[], const float y[], float w[], int inputs, int outputs, float rate, void (*dact)(float*, int));


#endif
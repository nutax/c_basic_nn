#ifndef _NN_LAYER_H_
#define _NN_LAYER_H_

void nn_layer_forward(float output[], const float input[], const float weight[], int inputs, int outputs, void (*act)(float*, int));
void nn_layer_output_error(float output_e[], const float answer[], const float output[], int outputs);
void nn_layer_input_error(float input_e[], const float output_e[], const float weight[], int inputs, int outputs);
void nn_layer_update(float weight[], float output[], const float input[], const float output_e[], int inputs, int outputs, float rate, void (*dact)(float*, int));
#endif
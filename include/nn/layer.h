#ifndef _NN_LAYER_H_
#define _NN_LAYER_H_

void nn_layer_forward(const float x[], const float w[], float y[], int inputs, int outputs, void (*act)(float*, int));
void nn_layer_error(const float r[], const float y[], float e[], int outputs);
void nn_layer_delta(const float fd[], const float w[], float d[], int inputs, int outputs);
void nn_layer_update(const float fd[], const float fy[], const float y[], float w[], int inputs, int outputs, float rate, void (*dact)(float*, int));


#endif
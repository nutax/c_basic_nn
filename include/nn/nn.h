#ifndef _NN_H_
#define _NN_H_

#include <math.h>

void nn_forward(const float x[], const float w[], float y[], int inputs, int outputs, void (*act)(float*, int));
void nn_error(const float r[], const float y[], float d[], int outputs);
void nn_delta(const float fd[], const float w[], float d[], int inputs, int outputs);
void nn_update(const float fd[], const float fy[], const float y[], int inputs, int outputs, float rate, void (*dact)(float*, int));

void nn_sigmoid(float y[], int size);
void nn_dsigmoid(float y[], int size);




#endif
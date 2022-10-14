#include "nn/nn.h"

void nn_forward(float x[], float w[], float y[], int inputs, int outputs, void (*act)(float*, int)){
    int o, h, i;
    for(o = 0; o<outputs; ++o){
        y[o] = 0;
        h = o*inputs;
        for(i = 0; i<inputs; ++i){
            y[o] += x[i] * w[h+i];
        }
    }
    act(y, outputs);
}
void nn_delta(float x[], float w[], float y[], int inputs, int outputs, void (*dact)(float*, int)){
    int o, h, i;
    
}
void nn_sigmoid(float y[], int size){
    int i;
    for(i = 0; i<size; ++i){
        y[i] = 1/(1+exp(-y[i]));
    }
}
void nn_dsigmoid(float y[], int size){
    int i;
    for(i = 0; i<size; ++i){
        y[i] = y[i]*(1-y[i]);
    }
}
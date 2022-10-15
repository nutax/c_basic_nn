#include "nn/layer.h"

void nn_layer_forward(const float x[], const float w[], float y[], int inputs, int outputs, void (*act)(float*, int)){
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
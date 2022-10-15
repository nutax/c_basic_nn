#include "nn/layer.h"

void nn_layer_forward(const float x[], const float w[], float y[], int inputs, int outputs, void (*act)(float*, int)){
    int i, j, h;
    for(i = 0; i<outputs; ++i){
        y[i] = 0;
        h = i*inputs;
        for(j = 0; j<inputs; ++j){
            y[i] += x[j] * w[h+j];
        }
    }
    act(y, outputs);
}
void nn_layer_error(const float r[], const float y[], float e[], int outputs){
    int i;
    for(i = 0; i<outputs; ++i){
        e[i] = r[i] - y[i];
    }
}
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
void nn_layer_output_delta(const float ans[], const float y[], float output_d[], int outputs){
    int i;
    for(i = 0; i<outputs; ++i){
        output_d[i] = ans[i] - y[i];
    }
}
void nn_layer_input_delta(const float d[], const float w[], float input_d[], int inputs, int outputs){
    int i, j, h;
    for(i = 0; i<inputs; ++i){
        input_d[i] = 0;
    }
    for(i = 0; i<outputs; ++i){
        h = i*inputs;
        for(j = 0; j < inputs; ++j){
            input_d[j] += d[i]*w[h+j];
        }
    }
}
void
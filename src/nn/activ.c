#include "nn/activ.h"

void nn_activ_sigmoid(float y[], int size){
    int i;
    for(i = 0; i<size; ++i){
        y[i] = 1/(1+exp(-y[i]));
    }
}
void nn_activ_dsigmoid(float y[], int size){
    int i;
    for(i = 0; i<size; ++i){
        y[i] = y[i]*(1-y[i]);
    }
}
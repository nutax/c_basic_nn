#include "core/main.h"


float const learning_rate = 0.1;
int const epochs = 1000;
int const rows = 4;
int const cols = 2;
float const train_input[rows*cols] = {1, 1, 1, 0, 0, 1, 0, 0};
float const train_output[rows] = {0, 1, 1, 0}; // XOR problem
int order[rows] = {0, 1, 2, 3};

float mem[1024];
float *free_mem = mem;
struct nn_layer layers[3];


int main(int argc, char **argv){
    int i, j, row;

    srand(time(NULL));

    nn_layer_create(&free_mem, layers+0, 2, NN_LAYER_INPUT, NULL, NULL, NULL);
    nn_layer_create(&free_mem, layers+1, 2, NN_LAYER_HIDDEN, layers+0, nn_activ_sigmoid, nn_activ_dsigmoid);
    nn_layer_create(&free_mem, layers+2, 1, NN_LAYER_OUTPUT, layers+1, nn_activ_sigmoid, nn_activ_dsigmoid);
    
    for(i = 0; i<epochs; ++i){
        shuffle(order, rows);
        for(j = 0; j<rows; ++j){
            row = order[j];

            //Foward
            nn_layer_output_load(layers+0, train_input+row*cols);
            nn_layer_forward(layers+1);
            nn_layer_forward(layers+2);

            //Backward
            nn_layer_output_error(layers+2, train_output+row);
            nn_layer_input_error(layers+2);
            nn_layer_update(layers+2, learning_rate);
            nn_layer_update(layers+1, learning_rate);
        }
    }

    


    hello();
    return EXIT_SUCCESS;
}
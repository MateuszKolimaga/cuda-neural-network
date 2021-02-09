#include <iostream>
#include <fstream>
#include <math.h>
#include <string>
#include <ctime>
#include <cstdlib>
#include <cuda.h>
#include <stdio.h>

const int n_x = 784; //Rozmiar warstwy wejsciowej, jednoczesnie rozmiar badanych obrazow (28x28)
const int n_h = 7;   //Ilosc neuronow w warstwie ukrytej
const int n_y = 1;   //Ilosc neuronow na wyjsciu sieci
float learning_rate = 0.075; //Predkosc uczenia
const int train_samples = 209; //Ilosc obrazow przechodzacych przez siec w fazie treningu
const int test_samples = 50;   //Ilosc obrazow przechodzacych przez siec w fazie testu
int num_samples = train_samples; //Aktualna ilosc obrazow przechodzacych przez siec
const int iter_num = 100;  //Ilosc przejsc zestawu treningowego przez siec
const int print_freq = 20;  //Czestotliwosc wyswietlania wartosci funkcji kosztu

//Struktura przechowujaca poszczegolne gradienty
struct grad_data {  
    float dW1[n_h][n_x];
    float dW2[n_y][n_h];
    float db1;
    float db2;
    float dA0[n_x][train_samples];
    float dA1[n_h][train_samples];
    float dA2[n_y][train_samples];
    float dZ1[n_h][train_samples];
    float dZ2[n_y][train_samples];
};

//Struktura przechowujaca parametry oraz wyjscia poszczegolnych warstw
struct param_data { 
    float train_x[n_x][train_samples];
    float test_x[n_x][test_samples];
    float train_y[n_y][train_samples];
    float test_y[n_y][test_samples];
    float W1[n_h][n_x];
    float W2[n_y][n_h];
    float b1;
    float b2;
    float A1[n_h][train_samples];
    float A2[n_y][train_samples];
    float Z1[n_h][train_samples];
    float Z2[n_y][train_samples];

    //Tablice pomocnicze
    float AT0[train_samples][n_x];
    float AT1[train_samples][n_h];
    float WT1[n_x][n_h];
    float WT2[n_h][n_y];
};

//Funkcje obslugujace struktury
void load_data(param_data&, grad_data&);
void delete_data(param_data&, grad_data&);

//Funkcja obliczajaca Z1 oraz dW1 
__global__ void kernel(float *arr1,float *arr2, float *arr_out, float *b, int *size_1, int *size_2)
{
    int x = blockIdx.x;
    int y = blockIdx.y;

    if((*size_1) == train_samples) { //Wtedy liczone jest dW1
        arr_out[x + (*size_2)*y] = 0;
        for(int i = 0; i < (*size_1) ; i++) 
            arr_out[x +(*size_2)*y] += (arr1[i + (*size_1)*y] * arr2[x + (*size_2)*i]) / train_samples;

    } else {                        //Wtedy liczone jest Z1
        arr_out[x + (*size_2)*y] = *b; 
        for(int i = 0; i < (*size_1) ; i++) 
            arr_out[x +(*size_2)*y] += (arr1[i + (*size_1)*y] * arr2[x + (*size_2)*i]); 

    } 
}

//Funkcja aktywacji warstwy wyjsciowej (Sigmoid)
void sigmoid(param_data &parameters) {
    int rows = 1;
    int cols = num_samples;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            parameters.A2[i][j] = 1 / (1 + exp(-parameters.Z2[i][j]));
        }
    }
}

//Funkcja aktywacji warstwy ukrytej (ReLu)
void relu(param_data &parameters) {
    int rows = n_h;
    int cols = num_samples;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (parameters.Z1[i][j] <= 0) parameters.A1[i][j] = 0;
            else parameters.A1[i][j] = parameters.Z1[i][j];
        }
    }
}

//Funkcja wyznaczajaca dZ1 za pomoca pochodnej z nieliniowej funkcji aktywacji (ReLu) oraz dA1
void relu_backward(param_data &parameters, grad_data &grads) {
    int rows = n_h;
    int cols = num_samples;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (parameters.Z1[i][j] <= 0) grads.dZ1[i][j] = 0;
            else grads.dZ1[i][j] = grads.dA1[i][j];
        }
    }
}

//Funkcja wyznaczajaca dZ2 za pomocą pochodnej z nieliniowej funkcji aktywacji (Sigmoid) oraz dA2 
void sigmoid_backward(param_data &parameters, grad_data &grads) {
    int rows = 1;
    int cols = num_samples;

    float s;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            s = 1 / (1 + exp(-parameters.Z2[i][j]));
            grads.dZ2[i][j] = grads.dA2[i][j] * s * (1 - s);
        }
    }
}

//Z1 = W1*X + b1    <-- liczone na GPU
void linear_forward_relu(param_data &parameters) {
    int rows_w = n_h;
    int cols_w = n_x;
    int rows_z = n_h;
    int cols_z = num_samples;
    float *arr1,*arr2,*arr_out, *b;
    int *size_1, *size_2;

    cudaMalloc((void **)&arr1, rows_w*cols_w*sizeof(float));
    cudaMalloc((void **)&arr2, cols_w*cols_z*sizeof(float));
    cudaMalloc((void **)&arr_out, rows_z*cols_z*sizeof(float));
    cudaMalloc((void **)&b, sizeof(float));
    cudaMalloc((void **)&size_1, sizeof(int));
    cudaMalloc((void **)&size_2, sizeof(int));

    cudaMemcpy(arr1,parameters.W1, rows_w*cols_w*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(arr2,parameters.train_x, n_x*cols_z*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b,&parameters.b1, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(size_1,&n_x, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(size_2,&num_samples, sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(cols_z,rows_z);

    kernel<<<grid,1>>>(arr1,arr2,arr_out,b,size_1,size_2);
    //cudaDeviceSynchronize();

    cudaMemcpy(parameters.Z1,arr_out,rows_z*cols_z*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(arr1);
    cudaFree(arr2);
    cudaFree(arr_out);
    cudaFree(b);
    cudaFree(size_1);
    cudaFree(size_2);

    // Alternatywny kod na CPU:
    // for (int i_z = 0; i_z < rows_z; i_z++) {
    //     for (int j_z = 0; j_z < cols_z; j_z++) {
    //         for (int j_w = 0; j_w < cols_w; j_w++) {
    //             parameters.Z1[i_z][j_z] += parameters.W1[i_z][j_w] * parameters.train_x[j_w][j_z];
    //         }
    //         parameters.Z1[i_z][j_z] += parameters.b1;
    //     }
    // } 
}

//Z2 = W2*A1 + b2  
void linear_forward_sigm(param_data &parameters) {
    int rows_w = n_y;
    int cols_w = n_h;
    int rows_z = n_y;
    int cols_z = num_samples;

    for (int i_z = 0; i_z < rows_z; i_z++) {
        for (int j_z = 0; j_z < cols_z; j_z++) {
            for (int j_w = 0; j_w < cols_w; j_w++) {
                parameters.Z2[i_z][j_z] += parameters.W2[i_z][j_w] * parameters.A1[j_w][j_z];
            }
            parameters.Z2[i_z][j_z] += parameters.b2;
        }
    }
}

//Funkcja wybierajaca tryb aktywacji
void linear_activation_forward(param_data &parameters, std::string activation) { 
    if (activation.compare("sigmoid") == 0) {
        linear_forward_sigm(parameters);
        sigmoid(parameters);
    }
    else {
        linear_forward_relu(parameters);
        relu(parameters);
    }
}

//Funkcja obliczajaca wartosc kosztu po pojedynczym przejsciu zestawu treningowego przez siec
float compute_cost(param_data &parameters) { 
    float cost = 0;
    float m = train_samples;

    for (int i = 0; i < m; i++) {
        cost += (-1 / m) * ( parameters.train_y[0][i] * log(parameters.A2[0][i]) + (1 - parameters.train_y[0][i]) * log(1 - parameters.A2[0][i]));
    }
    return cost;
}

//dW1 = (dZ1 * X.T) / train_samples  <-- liczone na GPU
//dA0 = (W1).T * dZ1                 <-- nie musi być liczone
//db1 = sum(dZ1) / train_samples     <-- liczone na CPU
void linear_backward_relu(param_data &parameters, grad_data &grads) { 
    int rows_dw1 = n_h;
    int cols_dw1 = n_x;
    int rows_dz1 = n_h;
    int cols_dz1 = train_samples;
    int rows_da0 = n_x;
    int cols_da0 = train_samples;
    int cols_wt1 = n_h;

    for (int i = 0; i < rows_da0; i++) {
        for (int j = 0; j < cols_da0; j++) {
            parameters.AT0[j][i] = parameters.train_x[i][j];
        }
    }

    for (int i = 0; i < rows_dw1; i++) {
        for (int j = 0; j < cols_dw1; j++) {
            parameters.WT1[j][i] = parameters.W1[i][j];
        }
    }

    for (int i = 0; i < rows_dz1; i++) {
        for (int j = 0; j < cols_dz1; j++) {
            grads.db1 += grads.dZ1[i][j] / train_samples;
        }
    }

    float *arr1,*arr2,*arr_out, *b;
    int *size_1, *size_2;

    cudaMalloc((void **)&arr1, rows_dz1*cols_dz1*sizeof(float));
    cudaMalloc((void **)&arr2, cols_da0*rows_da0*sizeof(float));
    cudaMalloc((void **)&arr_out, rows_dw1*cols_dw1*sizeof(float));
    cudaMalloc((void **)&b, sizeof(float));
    cudaMalloc((void **)&size_1, sizeof(int));
    cudaMalloc((void **)&size_2, sizeof(int));

    cudaMemcpy(arr1, grads.dZ1, rows_dz1*cols_dz1*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(arr2, parameters.AT0, cols_da0*rows_da0*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b, &parameters.b1, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(size_1, &train_samples, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(size_2, &n_x, sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(cols_dw1,rows_dw1);

    kernel<<<grid,1>>>(arr1,arr2,arr_out,b,size_1,size_2);
   // cudaDeviceSynchronize();

    cudaMemcpy(grads.dW1,arr_out,rows_dw1*cols_dw1*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(arr1);
    cudaFree(arr2);
    cudaFree(arr_out);
    cudaFree(b);
    cudaFree(size_1);
    cudaFree(size_2);

    //Alternatywna wersja na CPU:

    // for (int i_dw1 = 0; i_dw1 < rows_dw1; i_dw1++) {
    //     for (int j_dw1 = 0; j_dw1 < cols_dw1; j_dw1++) {
    //         for (int j_dz1 = 0; j_dz1 < cols_dz1; j_dz1++) {
    //             grads.dW1[i_dw1][j_dw1] += (grads.dZ1[i_dw1][j_dz1] * parameters.AT0[j_dz1][j_dw1]) / float(train_samples);
    //         }
    //     }
    // }

    // for (int i_da0 = 0; i_da0 < rows_da0; i_da0++) {
    //     for (int j_da0 = 0; j_da0 < cols_da0; j_da0++) {
    //         for (int j_wt1 = 0; j_wt1 < rows_dz1; j_wt1++) {
    //             grads.dA0[i_da0][j_da0] += parameters.WT1[i_da0][j_wt1] * grads.dZ1[j_wt1][j_da0];
    //         }
    //     }
    // }
}

//dW2 = dZ2 * (A1).T / train_samples   
//dA1 = (W2).T * dZ2                        <-- Wszystko liczone na CPU    
//db2 = sum(dZ2) /train_samples                
void linear_backward_sigm(param_data &parameters, grad_data &grads) {
    int rows_dw2 = n_y;
    int cols_dw2 = n_h;
    int rows_dz2 = n_y;
    int cols_dz2 = train_samples;
    int rows_da1 = n_h;
    int cols_da1 = train_samples;
    int cols_wt2 = n_y;


    for (int i = 0; i < rows_da1; i++) {
        for (int j = 0; j < cols_da1; j++) {
            parameters.AT1[j][i] = parameters.A1[i][j];
        }
    }

    for (int i = 0; i < rows_dw2; i++) {
        for (int j = 0; j < cols_dw2; j++) {
            parameters.WT2[j][i] = parameters.W2[i][j];
        }
    }


    for (int i = 0; i < rows_dz2; i++) {
        for (int j = 0; j < cols_dz2; j++) {
            grads.db2 += grads.dZ2[i][j] / train_samples;
        }
    }

    for (int i_dw2 = 0; i_dw2 < rows_dw2; i_dw2++) {
        for (int j_dw2 = 0; j_dw2 < cols_dw2; j_dw2++) {
            for (int j_dz2 = 0; j_dz2 < cols_dz2; j_dz2++) {
                grads.dW2[i_dw2][j_dw2] += (grads.dZ2[i_dw2][j_dz2] * parameters.AT1[j_dz2][j_dw2]) / train_samples;
            }
        }
    }


    for (int i_da1 = 0; i_da1 < rows_da1; i_da1++) {
        for (int j_da1 = 0; j_da1 < cols_da1; j_da1++) {
            for (int j_wt2 = 0; j_wt2 < cols_wt2; j_wt2++) {
                grads.dA1[i_da1][j_da1] += parameters.WT2[i_da1][j_wt2] * grads.dZ2[j_wt2][j_da1];
            }
        }
    }

}

//Funkcja wybierajaca tryb obliczania gradientow
void linear_activation_backward(param_data &parameters, grad_data &grads, std::string activation) {
    if (activation.compare("relu") == 0) {
        relu_backward(parameters, grads);
        linear_backward_relu(parameters, grads);
    }
    else {
        sigmoid_backward(parameters, grads);
        linear_backward_sigm(parameters, grads);
    }
}

//Aktualizowanie parametrów sieci po jednej iteracji
void update_parameters(param_data &parameters, grad_data &grads) {
    int rows_W1 = n_h;
    int cols_W1 = n_x;
    int rows_W2 = n_y;
    int cols_W2 = n_h;

    for (int i = 0; i < rows_W1; i++) {
        for (int j = 0; j < cols_W1; j++) {
            parameters.W1[i][j] -= learning_rate * grads.dW1[i][j];
        }
    }
    for (int i = 0; i < rows_W2; i++) {
        for (int j = 0; j < cols_W2; j++) {
            parameters.W2[i][j] -= learning_rate * grads.dW2[i][j];
        }
    }

    parameters.b1 -= learning_rate * grads.db1;
    parameters.b2 -= learning_rate * grads.db2;
    
}

//Glowna funkcja przechodzaca przez siec
void two_layer_model(param_data &parameters, grad_data &grads) {
    float cost = 0;

    for (int i = 0; i < iter_num + 1; i++) {
        delete_data(parameters, grads);

        linear_activation_forward(parameters, "relu");
        linear_activation_forward(parameters, "sigmoid");

        cost = compute_cost(parameters);

        for (int j = 0; j < train_samples; j++) {
            grads.dA2[0][j] = -((parameters.train_y[0][j] / parameters.A2[0][j]) - ((1. - parameters.train_y[0][j]) / (1. - parameters.A2[0][j])));
        }

        linear_activation_backward(parameters, grads, "sigmoid");
        linear_activation_backward(parameters, grads, "relu");

        update_parameters(parameters, grads);

        if (i % print_freq == 0) {
            std::cout << "Koszt po iteracji " << i << ": " << cost << "\n\n";
        }
    }

}

//Sprawdzanie skutecznosci sieci
void accuracy_check_train(param_data &parameters){
    float accuracy = 0;

    linear_activation_forward(parameters, "relu");
    linear_activation_forward(parameters, "sigmoid");

    for (int j = 0; j < train_samples; j++) {
            if(parameters.A2[0][j] >= 0.5 && parameters.train_y[0][j] == 1) accuracy += 1;
            else if(parameters.A2[0][j] < 0.5 && parameters.train_y[0][j] == 0) accuracy += 1;
    }

    std::cout << "Accuracy (training): " << accuracy / train_samples << "\n";

    num_samples = test_samples;
    accuracy = 0;

    linear_activation_forward(parameters, "relu");
    linear_activation_forward(parameters, "sigmoid");

    for (int j = 0; j < test_samples; j++) {
        if(parameters.A2[0][j] >= 0.5 && parameters.test_y[0][j] == 1) accuracy += 1;
        else if(parameters.A2[0][j] < 0.5 && parameters.test_y[0][j] == 0) accuracy += 1;
    }

    std::cout << "Accuracy (test): " << accuracy / test_samples << "\n";
}


int main() {
    param_data parameters;
    grad_data grads;

    load_data(parameters, grads);

    two_layer_model(parameters, grads);

    accuracy_check_train(parameters);

    return 0;
}

void load_data(param_data &parameters, grad_data &grads) {
    srand(time(NULL));

    std::cout << "Ladowanie zestawu treningowego i testowego.\n";

    std::string path = "train_x.txt";
    std::ifstream input(path.c_str());
    for (int i = 0; i < n_x; i++) 
        for (int j = 0; j < train_samples; j++) input >> parameters.train_x[i][j];

    path = "test_x.txt";
    std::ifstream input2(path.c_str());
    for (int i = 0; i < n_x; i++) 
        for (int j = 0; j < test_samples; j++) input2 >> parameters.test_x[i][j];

    
    std::cout << "Wczytano zestaw treningowy i testowy.\n";

    path = "train_y.txt";
    std::ifstream input3(path.c_str());
    for (int j = 0; j < train_samples; j++) input3 >> parameters.train_y[0][j];
    

    path = "test_y.txt";
    std::ifstream input4(path.c_str());
    for (int j = 0; j < test_samples; j++) input4 >> parameters.test_y[0][j];
    

    std::cout << "Wczytano zestaw klas.\n";

    for (int i = 0; i < n_h; i++) 
        for (int j = 0; j < n_x; j++) parameters.W1[i][j] = (rand()%10000 - 5000) * 0.000001;

    for (int i = 0; i < n_y; i++) 
        for (int j = 0; j < n_h; j++) parameters.W2[i][j] = (rand()%10000 - 5000) * 0.000001;

    parameters.b1 = 0;
    parameters.b2 = 0;
    grads.db1 = 0;
    grads.db2 = 0;
   
    for (int i = 0; i < n_h; i++) 
        for (int j = 0; j < train_samples; j++) parameters.Z1[i][j] = 0;

    for (int i = 0; i < n_y; i++) 
        for (int j = 0; j < train_samples; j++) parameters.Z2[i][j] = 0;
}

void delete_data(param_data& parameters, grad_data& grads) {

    for (int i = 0; i < n_h; i++) 
        for (int j = 0; j < train_samples; j++) 
            parameters.Z1[i][j] = 0;

    for (int i = 0; i < n_y; i++) 
        for (int j = 0; j < train_samples; j++) 
            parameters.Z2[i][j] = 0;
  
    for (int i = 0; i < n_h; i++) 
        for (int j = 0; j < train_samples; j++) 
            parameters.A1[i][j] = 0;

    for (int i = 0; i < n_y; i++) 
        for (int j = 0; j < train_samples; j++) 
            parameters.A2[i][j] = 0;

    for (int i = 0; i < n_h; i++) 
        for (int j = 0; j < n_x; j++) 
            grads.dW1[i][j] = 0;

    for (int i = 0; i < n_y; i++) 
        for (int j = 0; j < n_h; j++) 
            grads.dW2[i][j] = 0;

    for (int i = 0; i < n_h; i++) 
        for (int j = 0; j < train_samples; j++) 
            grads.dA1[i][j] = 0;

    for (int i = 0; i < n_y; i++) 
        for (int j = 0; j < train_samples; j++) 
            grads.dA2[i][j] = 0;

    for (int i = 0; i < n_h; i++) 
        for (int j = 0; j < train_samples; j++) 
            grads.dZ1[i][j] = 0;

    for (int i = 0; i < n_y; i++) 
        for (int j = 0; j < train_samples; j++) 
            grads.dZ2[i][j] = 0;

    for (int i = 0; i < n_x; i++) 
        for (int j = 0; j < train_samples; j++)
            grads.dA0[i][j] = 0;

    grads.db1 = 0;
    grads.db2 = 0;

}

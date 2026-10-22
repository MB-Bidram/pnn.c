#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define d_model 4


struct Model {
    float W[d_model][d_model]; // 4 x 4 matrix, right?
    float bias;
};

struct TData {
    float V[4];
    int label;
};

float PNN(struct Model* model, float x[], int size);

// Mathematic operators (Functions)

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float frand(void) {
    return (float)rand() / (float)RAND_MAX;
}

// Memory

void* alloc(int size) {
    void* ptr = malloc(size);
    if (!ptr) { perror("Memory allocation failed\n"); exit(1); }

    return ptr;
}

// PNN

struct Model* Model_New(int size) {
    srand(time(NULL));
    
    struct Model* model = alloc(sizeof(struct Model));

    for (int j=0; j<size; j++) {
        for (int i=0; i<size; i++) {
            model->W[j][i] = frand();
        }
    }

    model->bias = frand();

    return model;
}

struct Model* Model_Train(struct TData x[], float lr, int epoch, int size) {
    struct Model* model = Model_New(d_model);

    // for epoch
    for (int e=0; e<epoch; e++) {
        printf("Epoch: %d:\n", e+1);
        // Training loop
        for (int i=0; i<size; i++) {
            // Predict models answer
            float p = PNN(model, x[i].V, d_model);

            // Calculate the error (loss)
            float loss = p - x[i].label;

            printf("  Loss: %f\n", loss);

            // Gradient
            for (int j1=0; j1<4; j1++) {
                for (int j=0; j<4; j++) {
                    float gradW = loss * x[i].V[j];
                    model->W[j1][j] -= lr * gradW;
                }
            }

            model->bias -= lr * loss;
        }
    }

    return model;
}

float PNN(struct Model* model, float x[], int size) {
    float y = 0;

    // Scaler Dot Product (SDP, cool name right?)
    for (int j=0; j<size; j++) {
        for (int i=0; i<size; i++) {
            y += model->W[j][i] * x[i];
        }
    }

    y += model->bias;

    return sigmoid(y);
}

int main(void) {
    float x[4] = {1, 1, 1, 1};

    struct TData tdata[] = {
        {{1, 1, 1, 1}, 1},
        {{1, 0, 0, 1}, 0},
        {{1, 1, 1, 0}, 0},
        {{0, 0, 0, 0}, 1},
        {{1, 0, 1, 1}, 0},
        {{0, 1, 0, 0}, 0},
        {{0, 0, 0, 1}, 0}
    };

    struct Model* pnn = Model_Train(tdata, 0.05, d_model, 7);
    float y = PNN(pnn, x, d_model);
    printf("%.2f%%\n", y * 100);


    free(pnn);

    return 0;
}

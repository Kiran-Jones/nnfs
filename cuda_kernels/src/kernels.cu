#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../include/kernels.h"


__global__ void dense_forward_kernel(int batch,
                                     int in_dim,
                                     int out_dim,
                                     const float *__restrict__ X,
                                     const float *__restrict__ W,
                                     const float *__restrict__ b,
                                     float *__restrict__ Y) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < batch && col < out_dim) {
        float sum = 0.0f;
        int x_offset = row * in_dim;
        for (int k = 0; k < in_dim; ++k) {
            sum += X[x_offset + k] * W[k * out_dim + col];
        }
        Y[row * out_dim + col] = sum + b[col];
    }
}


__global__ void relu_forward_kernel(int N,
                                    const float *__restrict__ X,
                                    float *__restrict__ Y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        Y[idx] = X[idx] > 0.0f ? X[idx] : 0.0f;
    }
}


__global__ void relu_backward_kernel(int N,
                                     const float* __restrict__ X,
                                     const float* __restrict__ dY,
                                     float*       __restrict__ dX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float mask = X[idx] > 0.0f; // 1 if X[idx] > 0, else 0
        dX[idx] = mask * dY[idx]; // dY[idx] if mask == 1, else 0
    }
}
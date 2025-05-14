#pragma once

#include <cuda_runtime.h>

__global__ void dense_forward_kernel(int batch,
                                     int in_dim,
                                     int out_dim,
                                     const float *__restrict__ X,
                                     const float *__restrict__ W,
                                     const float *__restrict__ b,
                                     float *__restrict__ Y);

__global__ void relu_forward_kernel(int N,
                                    const float *__restrict__ X,
                                    float *__restrict__ Y);

__global__ void relu_backward_kernel(int N,
                                     const float* __restrict__ X,
                                     const float* __restrict__ dY,
                                     float*       __restrict__ dX);

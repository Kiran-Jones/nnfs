#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cuda_runtime.h>

#include "../include/kernels.h"
#include <cstring>

namespace py = pybind11;


py::array_t<float> dense_forward(py::array_t<float> X, py::array_t<float> W, py::array_t<float> b) {
    auto X_buf = X.request();
    auto W_buf = W.request();
    auto b_buf = b.request();
    
    int batch = static_cast<int>(X_buf.shape[0]);
    int in_dim = static_cast<int>(X_buf.shape[1]);
    int out_dim = static_cast<int>(W_buf.shape[1]);

    const float* host_X = static_cast<const float*>(X_buf.ptr);
    const float* host_W = static_cast<const float*>(W_buf.ptr);
    const float* host_b = static_cast<const float*>(b_buf.ptr);

    // Allocate GPU memory
    float *device_X = nullptr, *device_W = nullptr, *device_b = nullptr, *device_Y = nullptr;
    cudaMalloc(&device_X, batch * in_dim * sizeof(float));
    cudaMalloc(&device_W, in_dim * out_dim * sizeof(float));
    cudaMalloc(&device_b, out_dim * sizeof(float));
    cudaMalloc(&device_Y, batch * out_dim * sizeof(float));

    // Copy input data to allocated space on GPU
    cudaMemcpy(device_X, host_X, batch * in_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_W, host_W, in_dim * out_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, out_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Call dense kernel
    dim3 block(16, 16);
    dim3 grid((out_dim + block.x - 1) / block.x, (batch + block.y - 1) / block.y);
    dense_forward_kernel<<<grid, block>>>(batch, in_dim, out_dim, device_X, device_W, device_b, device_Y);
    cudaDeviceSynchronize();

    // Create array to hold result
    py::array_t<float> Y({batch, out_dim});
    auto Y_buf = Y.request();
    float* host_Y = static_cast<float*>(Y_buf.ptr);

    // Copy result back to host
    cudaMemcpy(host_Y, device_Y, batch * out_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(device_X); cudaFree(device_W); cudaFree(device_b); cudaFree(device_Y);

    return Y;
}


py::array_t<float> relu_forward(py::array_t<float> X) {
    auto X_buf = X.request();
    size_t N = X_buf.size;

    const float* host_X = static_cast<const float*>(X_buf.ptr);

    // Allocate GPU memory
    float *device_X = nullptr, *device_Y = nullptr;
    cudaMalloc(&device_X, N * sizeof(float));
    cudaMalloc(&device_Y, N * sizeof(float));

    // Copy input data to GPU
    cudaMemcpy(device_X, host_X, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    relu_forward_kernel<<<blocks, threads>>>(static_cast<int>(N), device_X, device_Y);
    cudaDeviceSynchronize();

    // Prepare output array with same shape and strides as input
    py::array_t<float> Y(X_buf.shape, X_buf.strides);
    auto Y_buf = Y.request();
    float* host_Y = static_cast<float*>(Y_buf.ptr);

    // Copy result from GPU
    cudaMemcpy(host_Y, device_Y, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_X); cudaFree(device_Y);

    return Y;
}


py::array_t<float> relu_backward(py::array_t<float> X, py::array_t<float> dY) {
    auto X_buf = X.request();
    auto dY_buf = dY.request();
    size_t N = X_buf.size;

    const float* host_X = static_cast<const float*>(X_buf.ptr);
    const float* host_dY = static_cast<const float*>(dY_buf.ptr);

    // Allocate GPU memory
    float *device_X = nullptr, *device_dY = nullptr, *device_dX = nullptr;
    cudaMalloc(&device_X, N * sizeof(float));
    cudaMalloc(&device_dY, N * sizeof(float));
    cudaMalloc(&device_dX, N * sizeof(float));

    // Copy input data to GPU
    cudaMemcpy(device_X, host_X, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_dY, host_dY, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    relu_backward_kernel<<<blocks, threads>>>(static_cast<int>(N), device_X, device_dY, device_dX);
    cudaDeviceSynchronize();

    // Prepare output array with same shape and strides as input
    py::array_t<float> dX(X_buf.shape, X_buf.strides);
    auto dX_buf = dX.request();
    float* host_dX = static_cast<float*>(dX_buf.ptr);

    // Copy result from GPU
    cudaMemcpy(host_dX, device_dX, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_X); cudaFree(device_dY); cudaFree(device_dX);
 
    return dX;
}

// export module as cuda_kernels
PYBIND11_MODULE(cuda_kernels, m) {
    m.def("dense_forward_cuda", &dense_forward, "Dense forward (CUDA)", 
        py::arg("X"), py::arg("W"), py::arg("b"));
          
    m.def("relu_forward_cuda", &relu_forward, "ReLU forward (CUDA)",
        py::arg("X"));

    m.def("relu_backward_cuda", &relu_backward, "ReLU backward (CUDA)",
        py::arg("X"), py::arg("dY"));
}

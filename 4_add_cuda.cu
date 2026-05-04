#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void add(int *A, int *B, int *C, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        C[tid] = A[tid] + B[tid];
    }
}

int main() {
    int N = 8;   // can increase for large vectors
    int size = N * sizeof(int);

    int *A = new int[N];
    int *B = new int[N];
    int *C = new int[N];

    // Initialize
    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i * 2;
    }

    int *dA, *dB, *dC;
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    add<<<blocks, threads>>>(dA, dB, dC, N);

    cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);

    cout << "Result:\n";
    for (int i = 0; i < N; i++)
        cout << C[i] << " ";

    cout << endl;

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
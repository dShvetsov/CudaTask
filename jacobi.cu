#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuda_fp16.h>

#define EPS 0.0000001f
#define SIZE 1024
#define BIG_VALUE 65536
#define BLOCK_SIZE 256

// generate random matrix
void getMatrix(float* matrix, unsigned size)
{
  for (unsigned i = 0; i < size; i++) {
    for (unsigned j = 0; j < size; j++) {
      matrix[i + size * j] = (((float)rand() / RAND_MAX) - 0.5f) * 10;
    }
  }

  // add diagonal predominance
  for (unsigned i = 0; i < size; i++) {
    matrix[i + size * i] += 1000;
  }
}

// generate random vector
void get_f(float *f, unsigned size)
{
  for (unsigned i = 0; i < size; i++) {
    f[i] = (((float)rand() / RAND_MAX) - 0.5f) * 10;
  }
}

// compute Matrix B, and return result in argument
void  computeBMatrix(float *A, unsigned size)
{
  float *inverse_D = (float *)calloc(size, sizeof(float));

  //D = diag(A)
  //compute D^-1
  for (unsigned i = 0; i < size; i++) {
    inverse_D[i] = 1.0f / A[i + size * i];
  }

  //compute D^-1 * A
  for (unsigned i = 0; i < size; i++) { //columns
    for (unsigned j = 0; j < size; j++) { //lines
      A[i + size * j] = inverse_D[j] * A[i + size * j];
    }
  }

  //compute I - D^-1 * A
  for (unsigned i = 0; i < size; i++) {
	for (unsigned j = 0; j < size; j++) {
	  float tmp = (i == j) ? 1 : 0;
	  A[i + size * j] = tmp - A[i + size * j];
	}
  }

  free(inverse_D);
}

// compute vector g and return in argument
void compute_g(float *A, float* b, unsigned size)
{
  // g = diag(A) ^ -1 * b;
  for (unsigned i = 0; i < size; i++) {
    b[i] = (1.0f / A[i + size * i]) * b[i];
  }
}

// metric is : max(|Ax_i - f_i|)
float precision(float *matrix, float *f, float* x, unsigned size)
{
  float max = 0;
  for (unsigned i = 0; i < size; i++) {
    float cur = 0;
    for (unsigned j = 0; j < size; j++) {
      cur += matrix[j + size * i] * x[j];
    }
    cur = fabs(f[i] - cur);
    max = max < cur ? cur : max;
  }
  return max;
}

// cuda kernel
__global__ void jacobi(float* B, float* g, float* x, unsigned size, float* x_next)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float x_curr = 0;
  for (int i = 0; i < size; i++) {
    x_curr += B[i + idx * size] * x[i];
  }
  x_next[idx] = x_curr + g[idx];
}


int main()
{
  float eps = EPS;
  unsigned size = SIZE;

  // alloc memory on CPU
  float *host_A = (float *)malloc(size * size * sizeof(float));
  float *host_B = (float *)malloc(size * size * sizeof(float));
  float *host_f = (float *)malloc(size * sizeof(float));
  float *host_g = (float *)malloc(size * sizeof(float));
  float *host_x = (float *)calloc(size, sizeof(float)); // start with null vector

  getMatrix(host_A, size);
  get_f(host_f, size);

  memcpy(host_B, host_A, size * size * sizeof(float));
  memcpy(host_g, host_f, size * sizeof(float));

  compute_g(host_A, host_g, size);
  computeBMatrix(host_B, size);

  // alloc memory on GPU
  float *dev_B, *dev_g, *dev_x_prev, *dev_x_next;
  cudaMalloc((void **)&dev_B, size * size * sizeof(float));
  cudaMalloc((void **)&dev_g, size * sizeof(float));
  cudaMalloc((void **)&dev_x_prev, size * sizeof(float));
  cudaMalloc((void **)&dev_x_next, size * sizeof(float));

  //copy memory from CPU to GPU
  cudaMemcpy(dev_x_prev, host_x, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_x_next, dev_x_prev, size * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaMemcpy(dev_B, host_B, size * size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_g, host_g, size * sizeof(float), cudaMemcpyHostToDevice);

  float p = 0; //precision
  float prev_p = BIG_VALUE; //previos precision
  // when precision stop decrease stop compute
  do {
    for (int i = 0; i < 512; i++) {
      //swap vectors
      float *tmp = dev_x_next;
      dev_x_next = dev_x_prev;
      dev_x_prev = tmp;

	  jacobi <<<dim3(size / BLOCK_SIZE), dim3(BLOCK_SIZE)>>> (dev_B, dev_g, dev_x_prev, size, dev_x_next);
	  cudaDeviceSynchronize();
    }
	//get result
    cudaMemcpy(host_x, dev_x_next, size * sizeof(float), cudaMemcpyDeviceToHost);

	prev_p = p;
	p = precision(host_A, host_f, host_x, size);
	printf("precision : %f \n", p);
  } while (p > eps && p < prev_p);

  printf("success\n");

  //free memory on CPU
  free(host_A);
  free(host_B);
  free(host_f);
  free(host_g);
  free(host_x);

  //free memory on GPU
  cudaFree(dev_B);
  cudaFree(dev_g);
  cudaFree(dev_x_prev);
  cudaFree(dev_x_next);

#ifdef _WIN32
  // stop console
  // only Windows needed
  scanf("\n");
#endif
  return 0;
}


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

texture<float, 2, cudaReadModeElementType> texMatrix;

// generate random matrix
void getMatrix(float* matrix, unsigned size)
{
  if (matrix == NULL){
	  fprintf(stderr, "getMatrix doesn't get matrix\n");
	  exit(-1);
  }
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
  if (get_f == NULL){
	  fprintf(stderr, "get_f doen't get vector\n");
	  exit(-1);
  }
  for (unsigned i = 0; i < size; i++) {
    f[i] = (((float)rand() / RAND_MAX) - 0.5f) * 10;
  }
}

// compute Matrix B, and return result in argument
void  computeBMatrix(float *A, unsigned size)
{
  if (A == NULL){
	  fprintf(stderr, "computeBMatrix doesn't get matrix\n");
	  exit(-1);
  }
  float *inverse_D = (float *)calloc(size, sizeof(float));
  if (inverse_D == NULL){
	  fprintf(stderr, "error on allocate memory\n");
	  exit(-1);
  }
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
  if (A == NULL || b == NULL){
	  fprintf(stderr, "compute_g get invalud arguments\n");
	  exit(-1);
  }
  // g = diag(A) ^ -1 * b;
  for (unsigned i = 0; i < size; i++) {
    b[i] = (1.0f / A[i + size * i]) * b[i];
  }
}

// metric is : max(|Ax_i - f_i|)
float precision(float *matrix, float *f, float* x, unsigned size)
{
  if (matrix == NULL || f == NULL || x == NULL){
	  fprintf(stderr, "precision get invalid parametr\n");
	  exit(-1);
  }

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
//with transpose matrix B
__global__ void jacobi(float* g, float* x, unsigned size, float* x_next)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float x_curr = 0;

  __shared__ float shared_x[SIZE];

  for (int i = 0; i < SIZE / blockDim.x; i++){
	  int loc_idx = blockDim.x * i + threadIdx.x;
	  shared_x[loc_idx] = x[loc_idx];
  }
  __syncthreads();

#pragma uroll 16
  for (int i = 0; i < size; i++) {
    // here loc_i is useless, becouse matrix B
    float tmp = tex2D(texMatrix, idx + 0.5f, i + 0.5f);
    x_curr += tmp* shared_x[i];
  }
  x_next[idx] = x_curr + g[idx];
}

void transpose(float* matrix, unsigned size)
{
	for (unsigned i = 0; i < size; i++){
		for (unsigned j = 0; j < i; j++){
			//swap value
			float tmp = matrix[i + j * size];
			matrix[i + j * size] = matrix[j + i * size];
			matrix[j + i * size] = tmp;
		}
	}
}

void checkGPUOperation()
{
	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess){
		fprintf(stderr, "Cuda Error : %s\n", cudaGetErrorString(code));
		exit(-1);
	}
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
  if (host_A == NULL || host_B == NULL || host_f == NULL
	  || host_g == NULL || host_x == NULL){
    fprintf(stderr, "error on allocate memory\n");
	return -1;
  }

  getMatrix(host_A, size);
  get_f(host_f, size);

  memcpy(host_B, host_A, size * size * sizeof(float));
  memcpy(host_g, host_f, size * sizeof(float));

  compute_g(host_A, host_g, size);
  computeBMatrix(host_B, size);
  transpose(host_B, size); // transpose for optimization

  // alloc memory on GPU
  float *dev_g, *dev_x_prev, *dev_x_next;
  cudaMalloc((void **)&dev_g, size * sizeof(float));
  checkGPUOperation();
  cudaMalloc((void **)&dev_x_prev, size * sizeof(float));
  checkGPUOperation();
  cudaMalloc((void **)&dev_x_next, size * sizeof(float));
  checkGPUOperation();

  // make texture

  cudaChannelFormatDesc channel =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

  cudaArray *cuArray;
  cudaMallocArray(&cuArray, &channel, size, size);
  checkGPUOperation();

  cudaMemcpyToArray(cuArray, 0, 0, host_B, size * size * sizeof(float),
                    cudaMemcpyHostToDevice);
  checkGPUOperation();

  cudaBindTextureToArray(texMatrix, cuArray);
  checkGPUOperation();

  //copy memory from CPU to GPU
  cudaMemcpy(dev_x_prev, host_x, size * sizeof(float), cudaMemcpyHostToDevice);
  checkGPUOperation();
  cudaMemcpy(dev_x_next, dev_x_prev, size * sizeof(float), cudaMemcpyDeviceToDevice);
  checkGPUOperation();
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

	  jacobi <<<dim3(size / BLOCK_SIZE), dim3(BLOCK_SIZE)>>> (dev_g, dev_x_prev, size, dev_x_next);
	  cudaDeviceSynchronize();
    }
	//get result
    cudaMemcpy(host_x, dev_x_next, size * sizeof(float), cudaMemcpyDeviceToHost);
	checkGPUOperation();

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


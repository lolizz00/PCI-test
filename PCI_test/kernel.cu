
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <conio.h>
#include <time.h>
#include <Windows.h>

/* Матрицы как матрицы */
typedef struct
{
	int width;
	int height;
	double* elements;
} Matrix;


/*Случайные числа  */
void randomDouble(double* ptr, int memsize)
{
	for (int i = 0;i <  memsize / sizeof(double); i++)
	{
		ptr[i] = (double)rand() / RAND_MAX * 1.0 - 1.0;
	}
}

/* Само вычисление на GPU */
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
	
	float Cvalue = 0.0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row > A.height || col > B.width) 
	{
		return;
	}
	for (int e = 0; e < A.width; ++e)
	{
		Cvalue += (A.elements[row * A.width + e]) * (B.elements[e * B.width + col]);
	}

	C.elements[row * C.width + col] = Cvalue;
}




unsigned status;




int main()
{

	/* Ищем все устройства, которые можно грузить */
	printf("CUDA devices:\n");
	int nDevices = 0;
	cudaGetDeviceCount(&nDevices);
	if (!nDevices)
	{
		printf("Empty!\n");
		system("pause");
		return 0;
	}
	for (int i = 0; i < nDevices; i++)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("\tDevice Number: %d\n", i);
		printf("\tDevice name: %s\n", prop.name);
		printf("\tPCI bus ID of the device: %d\n", prop.pciBusID);
		printf("\tPCI device ID of the device: %d", prop.pciDeviceID);
	}

	/* Выбираем нужное */

	while (1)
	{
		printf("\nEnter device number:"); scanf("%d", &nDevices);
		status = cudaSetDevice(nDevices);

		if (status == cudaSuccess)
		{
			break;
		}
		else
		{
			printf("Wrong number!\n");
		}
	}


	// ---


	int size;
	int mem_size;
	
	/* Размер матрицы, которая будет считаться */
	while (1)
	{
		printf("Enter matrix size for mul: ");
		scanf("%d", &size);

		if (size <= 2)
		{
			printf("Wrong size!\n");
		}
		else
		{
			break;
		}
	}


	mem_size = size * size * sizeof(double);

	Matrix A;
	Matrix B;
	Matrix C;

	Matrix HOST;

	A.height = size; A.width = size;
	B.width = size; B.height = size;
	C.width = size; C.height = size;
	HOST.width = size; HOST.height = size;

	/* Выделили память */
	status |= (unsigned)cudaMalloc(&A.elements, mem_size);
	status |= (unsigned)cudaMalloc(&B.elements, mem_size);
	status |= (unsigned)cudaMalloc(&C.elements, mem_size);

	if (status)
	{
		printf("Cannot allocate memory on GPU!\n");
		system("pause");
		return 0;
	}
	else
	{
		printf("Memory on GPU succ. allocated.\n");
	}

	

	HOST.elements = (double*)malloc(mem_size);

	if (!HOST.elements)
	{
		printf("Cannot allocate memory on HOST!\n");
		system("pause");
		return 0;
	}
	else
	{
		printf("Memory on HOST succ. allocated.\n");
	}


	// ---

	printf("Data size:%d\n", mem_size);
	randomDouble(HOST.elements, mem_size);
	printf("Data generated succ.\n");

	// ----
	
	printf("Begin work...\n");
	while (1)
	{

		unsigned tm = clock();

		/* Гоняем данные туда-сюда */
		cudaMemcpy(A.elements, HOST.elements, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(A.elements, HOST.elements, mem_size, cudaMemcpyHostToDevice);

		/* Считаем матрицы, что бы грелась ГПУ */
		MatMulKernel << <size, size >> >(A, B, C);
		cudaMemcpy(HOST.elements, C.elements, mem_size, cudaMemcpyDeviceToHost);


		tm = clock() - tm;

		/* Выводим время */
		printf("Runtime: %u ms\n", tm);


	}

	/* Память чистить лень, тут можно обойтись и без этого */

	printf("\n");
	system("pause");
    return 0;
}
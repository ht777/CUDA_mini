// CMPE297-6 HW2
// CUDA version Rabin-Karp

#include<stdio.h>
#include<iostream>


//#include <string.h>
//#include <assert.h>
//#include <stdlib.h>
#include <cuda_runtime.h>


__host__ __device__ bool mcmp(char* in,int j,char* patt,int plen)
	{ 
		int i=0;
		while (in[j] == patt[i]) 
		{
			if (i>plen)
			break;
			i++;
			j++;
		}
		if(patt[i]=='\0')
		return 0;
		else
		return -1;
    	}
/*ADD CODE HERE: Implement the parallel version of the sequential Rabin-Karp*/
__global__ void 
findIfExistsCu(char* input, int input_length, char* pattern, int pattern_length, int patHash, int* result,int* runtime)
{ 
	unsigned long long start_time=clock64();
	int inph, k=0;
	int tid = threadIdx.x;

//printf ("data: %c\n%s\n", input[1], pattern);
	for (inph=0,k=tid; k<pattern_length+tid; k++)
		{
		inph = (inph*256 + input[k]) % 997;
         	}
	if ((inph==patHash)&& (mcmp (input,tid,pattern,pattern_length)==0))
		{result[tid] = 1;}

	unsigned long long stop_time=clock64();
	runtime[tid]=(unsigned long long)(stop_time-start_time);
}

int main()
{
	// host variables
	char input[] = "HEABAL"; 	/*Sample Input*/
	char pattern[] = "AB"; 		/*Sample Pattern*/
	int patHash = 0; 			/*hash for the pattern*/

	int* runtime; 				/*Exection cycles*/
	int pattern_length = 2;		/*Pattern Length*/
	int input_length = 6; 		/*Input Length*/
	int* result;//input_length-pattern_length
	// device variables
	//char* d_input;
	//char* d_pattern;
	//int* d_result;
	//int* d_runtime;
        int d_patHash;

	// measure the execution time by using clock() api in the kernel as we did in Lab3
	int runtime_size = (input_length-pattern_length+1)*sizeof(int);/*FILL CODE HERE*/;

	result = (int *) malloc((input_length-pattern_length+1)*sizeof(int));
	runtime = (int *) malloc(runtime_size);
	    cudaError_t err = cudaSuccess;




	/*Calculate the hash of the pattern*/
	for (int i = 0; i < pattern_length; i++)
    {
        patHash = (patHash * 256 + pattern[i]) % 997;
    }

	/*ADD CODE HERE: Allocate memory on the GPU and copy or set the appropriate values from the HOST*/
	
	char* d_input = NULL;	
    err = cudaMalloc((void**)&d_input, input_length*sizeof(char)); 
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device input (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	char* d_pattern = NULL;	
    err = cudaMalloc((void**)&d_pattern,pattern_length*sizeof(char)); 
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device pattern (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	int* d_result = NULL;
    err = cudaMalloc((void**)&d_result, (input_length-pattern_length+1)*sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device result (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
//patHash is not allocated yet!!!

    // Copy the host input matrix A and B in host memory to the device input matrices in
    // device memory
	// TODO : Add proper mem copy APIs according to the memory that matrix A and B will be stored
	// -->
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_input, input, input_length*sizeof(char), cudaMemcpyHostToDevice);// FILL HERE
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy input from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_pattern, pattern, pattern_length*sizeof(char), cudaMemcpyHostToDevice);// FILL HERE
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy pattern from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int* d_runtime;
     int r_size = (input_length-pattern_length)*sizeof(int);
    //unsigned long long* runtime = (unsigned long long*)malloc(r_size);
    memset(runtime, 0, r_size);
    err = cudaMalloc((void**)&d_runtime, (input_length-pattern_length+1)*sizeof(int)); 
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device runtime (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


	/*ADD CODE HERE: Launch the kernel and pass the arguments*/
	int blocksPerGrid =1 ;
   	int threadsPerBlock = (input_length-pattern_length+1);
	findIfExistsCu<<<blocksPerGrid,threadsPerBlock>>>(d_input,input_length,d_pattern,pattern_length,patHash,d_result, d_runtime);	

	/*ADD CODE HERE: Copy the execution times from the GPU memory to HOST Code*/		
	
	
	/*RUN TIME calculation*/

    cudaMemcpy(runtime, d_runtime, r_size, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    unsigned long long elapsed_time = 0;
    for(int i = 0; i <= input_length-pattern_length; i++)
        if(elapsed_time < runtime[i])
            elapsed_time = runtime[i];

    printf("Kernel Execution Time: %llu cycles\n", elapsed_time);
	printf("Total cycles: %llu \n", elapsed_time);
	printf("Kernel Execution Time: %llu cycles\n", elapsed_time);


	/*ADD CODE HERE: COPY the result and print the result as in the HW description*/

	
    // Copy the device result matrix in device memory to the host result matrix
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");

    err = cudaMemcpy(result, d_result, (input_length-pattern_length+1)*sizeof(int), cudaMemcpyDeviceToHost);
    if (err)
    {
        fprintf(stderr, "Failed to copy result from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaThreadSynchronize();

   printf("Input string = %s\n", input);
   printf("Pattern = %s\n", pattern);

   for (int i=0; i<=input_length-pattern_length; i++)
	 printf("Pos:%d Result:%d\n",i,result[i]);


   cudaFree(d_input);
    cudaFree(d_pattern);
    cudaFree(d_result);
#ifdef TM
    cudaFree(d_runtime);
#endif
	// <--

	// Free host memory

	free(result);
#ifdef TM
	free(runtime);
#endif
	
	return 0;
}


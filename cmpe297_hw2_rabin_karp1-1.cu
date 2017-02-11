// CMPE297-6 HW2
// CUDA version Rabin-Karp

#include<stdio.h>
#include<iostream>

__host__ __device__ bool memory_compare(char* in,int j,char* patt,int plen)
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
findIfExistsCu(char* input, int input_length, char* pattern, int pattern_length, int patHash, int* result, int j,int* temp_res)
{ 
	//unsigned long long start_time=clock64();
	int tid=threadIdx.x;
	
	int k=0;
	int inph;	
	//printf(" in gpu:%s\n",pattern);
	
	for(inph=0,k=tid; k<pattern_length+tid; k++)
	{
		inph= (inph * 256 + input[k]) % 997;//input hash
		
	}
	if((inph==patHash) && memory_compare(input,tid,pattern,pattern_length)==0)
	result[tid]=1;
	
	if(result[tid]==1)
	{	
		*temp_res=1;
		//printf(" in gpu mres:%d\n",*mres);
	}


	//printf(" in gpu:%d\n",result[tid]);
	
	//unsigned long long stop_time=clock64();
	//runtime[tid]=(unsigned long long)(stop_time-start_time);
}

int main()
{	
	cudaError_t err = cudaSuccess;
	// host variables
	char input[] = "REPLACE THE CUPS IN TIME"; 	/*Sample Input*/
	char *pattern[3]={"REPLACE","TODAY FOR","JUST NOW"};
	int patHash[3] = {0}; 			/*hash for the pattern*/
	int* result; 
	int* result1;

	//int* runtime; 				/*Exection cycles*/
	int pattern_length[3]={7, 9, 15};		/*Pattern Length*/
	int input_length = 25; 		/*Input Length*/

	 printf("Searching for a Multiple pattern in a single string\n");
 	 printf("Input String = %s\n",input);
	
	/*Calculate the hash of the pattern*/
	for (int j = 0; j < 3; j++)
	{	result = (int *) malloc((input_length-pattern_length[j]+1)*sizeof(int));	
		char *temp=pattern[j];
		for (int i = 0; i < pattern_length[j]; i++)
    		{
       			 patHash[j] = (patHash[j] * 256 + temp[i]) % 997;
    		}
	

	/*ADD CODE HERE: Allocate memory on the GPU and copy or set the appropriate values from the HOST*/
	char* d_input = NULL;	
    err = cudaMalloc((void**)&d_input, input_length*sizeof(char)); 
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device input (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	char* d_pattern[3] ={0,0,0} ;	
    err = cudaMalloc((void**)&d_pattern[j], pattern_length[j]*sizeof(char)); 
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device pattern (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	static int* d_result =NULL;	
    err = cudaMalloc((void**)&d_result, ((input_length-pattern_length[j]+1)* sizeof(int))); 
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device result (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    static int* d_temp_res =NULL;	
    err = cudaMalloc((void**)&d_temp_res, 3* sizeof(int)); 
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device temp_res (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	//int* d_runtime;
	//int r_size= (input_length-pattern_length)*sizeof(int);
	//int runtime=(int*)malloc(r_size);
	//memset(runtime, 0, r_size);
   // err = cudaMalloc((void**)&d_runtime, (input_length-pattern_length+1)*sizeof(int)); 
    //if (err != cudaSuccess)
    //{
      //  fprintf(stderr, "Failed to allocate device runtime (error code %s)!\n", cudaGetErrorString(err));
        //exit(EXIT_FAILURE);
    //}
	
	
	err = cudaMemcpy(d_input, input, input_length*sizeof(char), cudaMemcpyHostToDevice);// FILL HERE
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to input from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
		err = cudaMemcpy(d_pattern[j], pattern[j], pattern_length[j]*sizeof(char), cudaMemcpyHostToDevice);// FILL HERE
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to pattern from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	

	/*ADD CODE HERE: Launch the kernel and pass the arguments*/
		
    int blocksPerGrid =1 ;// FILL HERE
    int threadsPerBlock = input_length-pattern_length[j]+1;// FILL HERE
	

    findIfExistsCu<<<blocksPerGrid,threadsPerBlock>>>(d_input,input_length,d_pattern[j],pattern_length[j],patHash[j],d_result,j,d_temp_res);
          
		


/*ADD CODE HERE: COPY the result and print the result as in the HW description*/
	err=cudaMemcpy(result, d_result,sizeof(int), cudaMemcpyDeviceToHost);
if(err)
printf("Result Copy From Device to Host failed");



	err=cudaMemcpy(result1, d_temp_res,sizeof(int), cudaMemcpyDeviceToHost);
if(err)
printf("temp_res Copy From Device to Host failed");

	if(*result1)
		printf("Pattern:%s was found\n",pattern[j]);
		else
		printf("Pattern:%s not found\n",pattern[j]);
}

 
  

		
	
	return 0;
}


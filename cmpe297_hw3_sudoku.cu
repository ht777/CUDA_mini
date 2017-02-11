// Simple SUDOKU probram in CUDA
// cmpe297_hw3_easysudoku.cu

#include<stdio.h>
#include<string.h>
#include <cuda_runtime.h>

const int big_2x[9][9] = {{1, 1, 1, 1, 1, 1, 1, 1, 1},
			  {1, 1, 1, 1, 1, 1, 1, 1, 1},
			  {1, 1, 1, 1, 1, 1, 1, 1, 1},
			  {1, 1, 1, 1, 1, 1, 1, 1, 1},
			  {1, 1, 1, 1, 1, 1, 1, 1, 1},
			  {1, 1, 1, 1, 1, 1, 1, 1, 1},
			  {1, 1, 1, 1, 1, 1, 1, 1, 1},
			  {1, 1, 1, 1, 1, 1, 1, 1, 1},
			  {1, 1, 1, 1, 1, 1, 1, 1, 1}};

// input 9x9 sudoku : 
// - 1~9 : valid values 
// - 0 : no value is decided
const int input_sdk[9][9] =  {{0, 7, 0, 0, 6, 5, 0, 8, 0},
			  {6, 0, 0, 0, 3, 0, 4, 0, 0},
			  {0, 2, 0, 0, 4, 0, 7, 0, 0},
			  {8, 6, 0, 0, 0, 2, 5, 7, 0},
			  {0, 0, 7, 4, 0, 6, 1, 0, 0},
			  {0, 5, 2, 3, 0, 0, 0, 6, 4},
			  {0, 0, 8, 0, 2, 0, 0, 3, 0},
			  {0, 0, 5, 0, 8, 0, 0, 0, 1},
			  {0, 4, 0, 7, 1, 0, 0, 5, 0}};
typedef struct {
	int val[9][9]; // values that each entry can get
	int num_options[9][9]; // number of values that each entry can get
	int not_in_cell[9][9];	// values not in each 3x3 cell
	int not_in_row[9][9];	// values not in each row
	int not_in_col[9][9];	// values not in each column
} stContext;
stContext context;

void initialize_all();
void print_all();

#define WIDTH	9

#define IS_OPTION(row, col, k) \
			((context->not_in_row[row][k] == 1) && \
			(context->not_in_col[col][k] == 1) && \
			(context->not_in_cell[row/3+(col/3)*3][k] == 1))? 1 : 0;
#define FINISHED()	(memcmp(context.num_options, big_2x, sizeof(big_2x)) == 0? 1: 0)


// rule: numbers should be unique in each sub-array, each row, and each column
__global__ void k_Sudoku(stContext *context, unsigned long long* runtime)
{
    int col = threadIdx.x;
    int row = threadIdx.y;

	unsigned long long start_time=clock64();
 

//finding unique elements and storing it in the context
    if(context->num_options[row][col] > 1)
	{
		// Find values that are not in the row, col, and the 
		// 3x3 cell that (row, col) is belonged to.
		int value = 0, temp;
		context->num_options[row][col] = 0;

		for(int k = 0; k < 9; k++)
		{
			temp = IS_OPTION(row, col, k);

			if(temp == 1)
			{
				context->num_options[row][col]++;
				value = k;

			}
		}

		// If the above loop found only one value, 
		// set the value to (row, col)
		if(context->num_options[row][col] == 1)
		{
			context->not_in_row[row][value] = 0;
			context->not_in_col[col][value] = 0;
			context->not_in_cell[(row)/3+((col)/3)*3][value] = 0;
			context->val[row][col] = value+1;

		}
	}

	unsigned long long stop_time=clock64();
	runtime[row*WIDTH+col]=(unsigned long long)(stop_time-start_time);//runtime for each thread



}

int main(int argc, char **argv)
{
    cudaError_t err;

    initialize_all();
    print_all();

    
   unsigned long long net_runtime=0;//stores the total execution time
    stContext *k_context=NULL;
	
    err = cudaMalloc((void**)&k_context, sizeof(stContext)); // TODO: Allocate context in GPU device memory 
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


	while(!FINISHED())// call the kernel until all the elements are found while copying the updated context each time the kernel needs to be called
{
    err = cudaMemcpy(k_context, &context, sizeof(stContext), cudaMemcpyHostToDevice);// TODO: Copy the input/updated context to GPU
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
/*-------------runtime related memory allocation----------*/
    unsigned long long* d_runtime;
    int r_size = WIDTH*WIDTH*sizeof(unsigned long long);
    unsigned long long* runtime = (unsigned long long*)malloc(r_size);
    memset(runtime, 0, r_size);
    cudaMalloc((void**)&d_runtime, r_size);
/*-------------------------xxxxxxxxx-----------------------*/
  
    // Assign as many threads as the matrix size so that
    // each thread can deal with one entry of the matrix
    dim3 dimGrid(WIDTH,WIDTH, 1);
    dim3 dimBlock(1, 1, 1);
    // Call the kernel function
    k_Sudoku<<<dimBlock,dimGrid>>>(k_context, d_runtime);


    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    cudaThreadSynchronize();

    printf("Copy output data from the CUDA device to the host memory\n");//copying the updated context from GPU to CPU
    err = cudaMemcpy(&context,k_context, sizeof(stContext), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy result from device to host (error code %s)!\n", cudaGetErrorString(err));
     }
    cudaMemcpy(runtime, d_runtime, r_size, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    
    unsigned long long elapsed_time = 0;
    for(int i = 0; i < WIDTH*WIDTH; i++)
        if(elapsed_time < runtime[i])
            elapsed_time = runtime[i];//highest execution time among all the simultaneously running threads
    net_runtime += elapsed_time;// calculates the total execution time, each time when the kernel is executed

}
    cudaThreadSynchronize();
	
    // Print the result
    print_all();


    printf("Kernel Execution Time: %llu cycles\n", net_runtime);// prints the total executing times


    // Free the device memory
    err = cudaFree(k_context);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free gpu data (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	//free(d_runtime);

	
    getchar();
	
    return 0;
}

void initialize_all()
{
    int i, j;

    memcpy(context.not_in_cell,big_2x, sizeof(big_2x));	
    memcpy(context.not_in_row,big_2x, sizeof(big_2x));	
    memcpy(context.not_in_col,big_2x, sizeof(big_2x));	
		
    for(i=0; i<9; i++){
    	for(j=0; j<9; j++){
            if(input_sdk[i][j] == 0)
            {
                context.val[i][j] = 0;
                context.num_options[i][j]=9;
            }
            else
            {
                context.val[i][j] = input_sdk[i][j];
                context.num_options[i][j] = 1;
                context.not_in_cell[i/3+(j/3)*3][input_sdk[i][j]-1] = 0;
                context.not_in_col[j][input_sdk[i][j]-1] = 0;
                context.not_in_row[i][input_sdk[i][j]-1] = 0;
            }
        }
    }
}


void print_all()
{
    int i, j, k;

    for(i=0; i<9; i++){
        for(j=0; j<9; j++){
            if(context.val[i][j] == 0)
                fprintf(stdout, "  %1d   ", context.val[i][j]);  
            else
                fprintf(stdout, " *%1d*  ", context.val[i][j]);  
            if((j==2)||(j==5)){
                fprintf(stdout, "| ");	
            }
        }
        fprintf(stdout, "\n");	
        if((i==2)||(i==5)){
            for(k=0; k<69; k++){
                fprintf(stdout, "-");	
            }
            fprintf(stdout, "\n");	
        }
    }
    fprintf(stdout, "\n");
}


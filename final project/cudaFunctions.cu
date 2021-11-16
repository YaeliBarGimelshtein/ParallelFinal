#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <limits.h>
#include "cudaHeader.h"


enum constants {WEIGHTS = 4,  RES = 3, A_ROWS = 9, A_COLS = 5, B_ROWS = 11 , B_COLS = 7};

__device__ __constant__ int a_rows = 9;
__device__ __constant__ int b_rows = 11;
__device__ __constant__ int a_cols = 5;
__device__ __constant__ int b_cols = 7;
__device__ __constant__ int res_components = 3;

__device__ __constant__ char levelA[A_ROWS][A_COLS] = {"NDEQ", "MILV", "FYW", "NEQK", "QHRK", "HY", "STA", "NHQK", "MILF"};
__device__ __constant__ char levelB[B_ROWS][B_COLS] = {"SAG", "ATV", "CSA", "SGND", "STPA", "STNK", "NEQHRK", "NDEQHK", "SNDEQK", "HFY", "FVLIM"};

__device__ int cuda_strlen(char* seq)
{
    int counter = 0;
    while(*seq != '\0')
    {
        counter++;
        seq++;
    }

    return counter;
}

__device__ int check_level_A_cuda(char from_seq1, char from_seq2)
{
    char from_seq1_new = from_seq1;
    char from_seq2_new = from_seq2;

    //UPPER CASE
    if(from_seq1 >= 'a' && from_seq1 <='z')
        from_seq1_new = from_seq1 - 32;
    
    if(from_seq2 >= 'a' && from_seq2 <='z')
        from_seq2_new = from_seq2 - 32;

    int counter_1 = 0, counter_2 = 0;
    for (int i = 0; i < a_rows; i++)
    {
        for (int j = 0; j < cuda_strlen(levelA[i]); j++)
        {
            if(from_seq1_new == levelA[i][j])
                counter_1++;
            
            if(from_seq2_new == levelA[i][j])
                counter_2++;
        }

        if(counter_1 > 0 && counter_2 > 0)
            return 1;

        counter_1 = 0;
        counter_2 = 0;
        
    }
    return 0;
}



__device__ int check_level_B_cuda(char from_seq1, char from_seq2)
{
    char from_seq1_new = from_seq1;
    char from_seq2_new = from_seq2;

    //UPPER CASE
    if(from_seq1 >= 'a' && from_seq1 <='z')
        from_seq1_new = from_seq1 - 32;
    
    if(from_seq2 >= 'a' && from_seq2 <='z')
        from_seq2_new = from_seq2 - 32;

    int counter_1 = 0, counter_2 = 0;
    for (int i = 0; i < b_rows; i++)
    {
        for (int j = 0; j < cuda_strlen(levelB[i]); j++)
        {
            if(from_seq1_new == levelB[i][j])
                counter_1++;
            
            if(from_seq2_new == levelB[i][j])
                counter_2++;
        }

        if(counter_1 > 0 && counter_2 > 0)
            return 1;
        
        counter_1 = 0;
        counter_2 = 0;
        
    }
    return 0;
}



__global__ void return_score_offset_mutant(char* seq1, char* seq2 , int* lenght_seq2 , int* weights, int* start, int* finish, int* res)
{
    int possible_mutants = *lenght_seq2;
    int curr_score = 0;
    int index = 0;

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int offset =  thread_id + *start;
    int thread_index_for_res_array = thread_id * res_components; 

    int dollars = 0, precent = 1, hashes = 2, spaces = 3; 

    if(offset <= *finish)
    {
        for (int mutant = 0; mutant < possible_mutants; mutant++) //all mutants
        {
            index = offset;
            for (int index_seq2 = 0; index_seq2 < *lenght_seq2; index_seq2++)
            {
                //MUTATION POSSIBLE ONLY IF NOT THE FINAL OFFSET
                if(offset < *finish)
                {
                    if (index_seq2 == mutant && mutant != 0)
                        index++;
                }
    
                
                if (seq1[index] == seq2[index_seq2])
                    curr_score += weights[dollars];
                
                else if (check_level_A_cuda(seq1[index], seq2[index_seq2]))
                    curr_score -= weights[precent];
                
                else if (check_level_B_cuda(seq1[index], seq2[index_seq2]))
                    curr_score -= weights[hashes];
                
                else
                    curr_score -= weights[spaces];

                index++;
            }
            
            if(curr_score > res[thread_index_for_res_array])
            {
                res[thread_index_for_res_array] = curr_score;
                res[thread_index_for_res_array + 1] = offset;
                res[thread_index_for_res_array + 2] = mutant;
            }
            curr_score = 0;
        }
    } 
}

int* return_cuda_score_offset_mutant(char* seq1, char* seq2 , int* weights, int offset_start, int offset_finish)
{
    //CREATE THE ARRAY THAT WILL RETURN THE BEST RESULT
    int* result = (int*)calloc(sizeof(int), RES);

    //DATA NEEDED
    int offset_size = offset_finish - offset_start + 1; //--> num of cuda threads
    int num_of_blocks = (offset_size / NUM_THREADS_PER_BLOCK);
    if (offset_size % NUM_THREADS_PER_BLOCK != 0)
        num_of_blocks ++;
    
    int lenght_of_res = (RES) * offset_size; //for each offset the gpu finds the best mutant

    //ALLOCATE DATA TO CUDA MEMORY
    char* cuda_seq1, *cuda_seq2;
    int* cuda_weights, *cuda_offset_start, *cuda_offset_finish ,*cuda_seq2_lenght;
    int* cuda_res, *res = (int*)calloc(sizeof(int) , lenght_of_res);
    int seq2_lenght = (strlen(seq2));

    for (int i = 0; i < lenght_of_res; i++) // get the array ready
    {
        res[i] = INT_MIN;
    }

        //sizes to allocate
    int size_for_cuda_seq1 = sizeof(char) * (strlen(seq1));
    int size_for_cuda_seq2 = sizeof(char) * seq2_lenght;
    int size_for_cuda_weights = sizeof(int) * (WEIGHTS);
    int size_for_cuda_int = sizeof(int);
    int size_for_cuda_res = sizeof(int) * lenght_of_res;

        //allocate
    cudaMalloc((void**)&cuda_seq1, size_for_cuda_seq1);
    cudaMalloc((void**)&cuda_seq2, size_for_cuda_seq2);
    cudaMalloc((void**)&cuda_weights, size_for_cuda_weights);
    cudaMalloc((void**)&cuda_offset_start, size_for_cuda_int);
    cudaMalloc((void**)&cuda_offset_finish, size_for_cuda_int);
    cudaMalloc((void**)&cuda_res, size_for_cuda_res);
    cudaMalloc((void**)&cuda_seq2_lenght, size_for_cuda_int);



    //COPY INPUT INTO DEVICE
    cudaMemcpy(cuda_seq1, seq1, size_for_cuda_seq1, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_seq2, seq2, size_for_cuda_seq2, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_weights, weights, size_for_cuda_weights, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_offset_start, &offset_start, size_for_cuda_int, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_offset_finish, &offset_finish, size_for_cuda_int, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_seq2_lenght, &seq2_lenght, size_for_cuda_int, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_res, res, size_for_cuda_res, cudaMemcpyHostToDevice);
    
    
    //LUNCH KERNEL
    return_score_offset_mutant<<<num_of_blocks, NUM_THREADS_PER_BLOCK>>>(cuda_seq1, cuda_seq2, cuda_seq2_lenght, cuda_weights, cuda_offset_start, cuda_offset_finish ,cuda_res);

    //COPY RESULT BACK TO HOST
    cudaMemcpy(res, cuda_res, size_for_cuda_res, cudaMemcpyDeviceToHost);
    
    //GET THE BIGGEST SCORE
    result[0] = INT_MIN; 

    for (int i = 0; i < lenght_of_res; i += 3)
    {
        if(result[0] < res[i])
        {
            result[0] = res[i];
            result[1] = res[i+1];
            result[2] = res[i+2];
        }
    }
    
    //FREE
    cudaFree(cuda_seq1);
    cudaFree(cuda_seq2);
    cudaFree(cuda_weights);
    cudaFree(cuda_offset_start);
    cudaFree(cuda_offset_finish);
    cudaFree(cuda_res);
    cudaFree(cuda_seq2_lenght);

    return result;
}

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <omp.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include "cudaHeader.h"

enum constants {ROOT = 0, WEIGHTS = 4, MAX_SEQ1 = 3000, MAX_SEQ2 = 2000, RES_INFO = 3, A_ROWS = 9, A_COLS = 5, B_ROWS = 11 , B_COLS = 7,
DOLLARS = 0 , PRECENT = 1, HASHES = 2, SPACES = 3};

char levelA[A_ROWS][A_COLS] = {"NDEQ", "MILV", "FYW", "NEQK", "QHRK", "HY", "STA", "NHQK", "MILF"};
  
char levelB[B_ROWS][B_COLS] = {"SAG", "ATV", "CSA", "SGND", "STPA", "STNK", "NEQHRK", "NDEQHK", "SNDEQK", "HFY", "FVLIM"};



int check_level_A(char from_seq1, char from_seq2)
{
    char from_seq1_new = toupper(from_seq1);
    char from_seq2_new = toupper(from_seq2);
    int counter_1 = 0, counter_2 = 0;
    for (int i = 0; i < A_ROWS; i++)
    {
        for (int j = 0; j < strlen(levelA[i]); j++)
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


int check_level_B(char from_seq1, char from_seq2)
{
    char from_seq1_new = toupper(from_seq1);
    char from_seq2_new = toupper(from_seq2);
    int counter_1 = 0, counter_2 = 0;
    for (int i = 0; i < B_ROWS; i++)
    {
        for (int j = 0; j < strlen(levelB[i]); j++)
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


int* return_omp_score_offset_mutant(char* seq1, char* seq2 , int* weights, int offset_max)
{
    int lenght_seq2 = strlen(seq2);
    int possible_mutants = lenght_seq2;
    int curr_score = 0;
    int index_seq1 = 0;

    int* res = (int*)malloc(sizeof(int) * RES_INFO);
    res[0] = INT_MIN;

#pragma omp parallel for shared(seq1, seq2, weights, lenght_seq2, possible_mutants) firstprivate(index_seq1, curr_score) 
    for (int offset = 0; offset < offset_max; offset++) //all offsets
    {
        for (int mutant = 0; mutant < possible_mutants; mutant++) //all mutants
        {
            index_seq1 = offset;
            for (int index_seq2 = 0; index_seq2 < lenght_seq2; index_seq2++)
            {
                if (index_seq2 == mutant && mutant != 0)
                    index_seq1++;
                
                if (seq1[index_seq1] == seq2[index_seq2])
                    curr_score +=weights[DOLLARS];
                
                else if (check_level_A(seq1[index_seq1], seq2[index_seq2]))
                    curr_score -= weights[PRECENT];
                
                else if (check_level_B(seq1[index_seq1], seq2[index_seq2]))
                    curr_score -= weights[HASHES];
                
                else
                    curr_score -= weights[SPACES];

                index_seq1++;
            }
            #pragma omp critical
            {
                if(curr_score > res[0])
                {
                    res[0] = curr_score;
                    res[1] = offset;
                    res[2] = mutant;
                }
            }
            curr_score = 0;
        }
    }
    //for the case that none letters is the same --> thread that catches first puts lowest score, so change manauly
    if(res[0] == -WEIGHTS * weights[SPACES]) 
    {
        res[1] = 0;
        res[2] = 0;
    }
    return res;
}



void workerProcess(int num_of_seq, char* seq1, int* weights)
{
    char* seq2 = (char*)malloc(sizeof(char) * MAX_SEQ2);
    int* res = (int*)malloc(sizeof(int) * RES_INFO);

    int* omp_res = (int*)calloc(sizeof(int) , RES_INFO);
    int* cuda_res = (int*)calloc(sizeof(int) , RES_INFO);
    
    int tag = 0;
    MPI_Status status;
    

    do
    {
        MPI_Recv(seq2,MAX_SEQ2,MPI_CHAR,ROOT,MPI_ANY_TAG,MPI_COMM_WORLD,&status); //RECEIVE WORK
        tag = status.MPI_TAG;
        
        if(tag != num_of_seq + 1) //NO SIGNAL TO DIE
        {
            //SPLIT WORK BETWEEN OMP AND CUDA
            int offsets = strlen(seq1) - strlen(seq2);
            int cuda_offset = offsets / 2 ; 
            int omp_offset = offsets / 2 + offsets % 2;

            //GET MAX SIZE FROM CUDA AND OMP
            omp_res = return_omp_score_offset_mutant(seq1, seq2, weights, omp_offset);
            cuda_res = return_cuda_score_offset_mutant(seq1, seq2, weights, omp_offset, offsets); //cuda off set start at number of omp offsets
            
            if(omp_res[0] < cuda_res[0])
                res = cuda_res;
            else
                res = omp_res;
            
            MPI_Send(&tag, 1, MPI_INT, ROOT, tag, MPI_COMM_WORLD); //SEND THE LOCATION
            MPI_Recv(&tag, 1, MPI_INT, ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, &status); //GET APPROVED 
            MPI_Send(res, RES_INFO, MPI_INT, ROOT, tag, MPI_COMM_WORLD); //SEND THE DATA  
        }
        
    } while (tag != num_of_seq + 1);
    
    //FREE DATA
    free(res);
    free(seq2);
}


void masterProcess(int num_of_seq, char** all_seq, char* seq1, int num_proc)
{
    //SET THE NUMBERS
    int jobs_total = num_of_seq;
    int num_workers = num_proc - 1;
    int send_and_rec = 0, jobs_sent = 0;
    int worker_id;
    MPI_Status status;

    //GATHER ALL RESULTS OF OFFSET, MUTATION AND SCORE
    int** res = (int**)malloc(sizeof(int*) * num_of_seq);

    for (int i = 0; i < num_of_seq; i++)
    {
        res[i] = (int*)malloc(sizeof(int) * RES_INFO);
    }
    

    //START WORKERS
    for(worker_id = 1; worker_id < num_proc; worker_id++)
    {
        MPI_Send(all_seq[jobs_sent], MAX_SEQ2, MPI_CHAR, worker_id, jobs_sent, MPI_COMM_WORLD);
        jobs_sent ++;
        send_and_rec ++;
    }
    
    //RECIVE AND SEND MORE WORK
    while(send_and_rec != 0)
    {
        //GET THE DATA AND SEND MORE
        int location = 0;
        MPI_Recv(&location, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status); //GET THE LOCATION
        MPI_Send(&location, 1, MPI_INT, status.MPI_SOURCE, location, MPI_COMM_WORLD);
        MPI_Recv(res[location], RES_INFO, MPI_INT, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status); //GET THE DATA
        send_and_rec --;
        
        if((jobs_sent < jobs_total))
        {
            MPI_Send(all_seq[jobs_sent], MAX_SEQ2, MPI_CHAR, status.MPI_SOURCE, jobs_sent, MPI_COMM_WORLD);
            jobs_sent ++;
            send_and_rec ++;
        }
    }
    
    //SEND THE SIGNAL FOR PROCESSES TO DIE
    int stop = num_of_seq + 1;
    for (worker_id = 1; worker_id < num_proc; worker_id++)
    {
        MPI_Send(&stop, 1 ,MPI_INT, worker_id, stop, MPI_COMM_WORLD);
    }
    
    //PRINT THE RES
    int m = 0;
    printf("-----------------------------------Parallel Program-----------------------------------------------\n");
    printf("\n");
    printf("Seq1 is %s\n", seq1);
    printf("\nHere are all the best matches for the strings:\n");
    
    for (int i = 0; i < num_of_seq; i++)
    {
        m = res[i][2];
        printf("seq2 = %s\n", all_seq[i]);
        if(res[i][2] == 0)
            m = strlen(all_seq[i]);
        printf("offset (n) = %d , mutation (k) = %d , score = %d \n", res[i][1], m, res[i][0]);
        printf("\n");
    }
    
    //FREE DATA
    for (int i = 0; i < num_of_seq; i++)
    {
        free(res[i]);
    }
    free(res);
}





int main(int argc, char *argv[])
{
    double start = time(NULL);
    //GENERAL DATA NEEDED
    int my_rank, num_procs, number_of_sequences;
    char* seq1 = (char*)malloc(sizeof(char) * MAX_SEQ1);
    int weights[WEIGHTS];
    char** all_seq;

    
    //MPI INIT
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs); 


    if(my_rank == ROOT)
    {
        //GET ALL THE INPUT

        //GET THE WEIGHTS
        for (int i = 0; i < WEIGHTS; i++)
        {
            scanf("%d",&weights[i]);
        }

        //GET SEQ1
        scanf("%s", seq1);

        //GET NUMBER OF SEQ'S
        scanf("%d", &number_of_sequences);

        //GET ALL OTHER SEQ'S
            //MALLOC **
        all_seq = (char**)malloc(sizeof(char*) * number_of_sequences);
            //MALOC *
        for (int i = 0; i < number_of_sequences; i++)
        {
            all_seq[i] = (char*)malloc(sizeof(char) * MAX_SEQ2);
        }
            //GET FROM FILE
        for (int i = 0; i < number_of_sequences; i++)
        {
            scanf("%s",all_seq[i]);
        }
        
        //MAKE SURE DATA IS BIF ENOUGH
         if(number_of_sequences < num_procs - 1)
        {
            printf("The program needs at least the %d inputs to work\n", num_procs - 1);
            MPI_Finalize();
            return 0;
        }
    }
    
   

    //BROADCAST SEQ1 AND WEIGHTS TO ALL PROC
    MPI_Bcast(seq1, MAX_SEQ1, MPI_CHAR, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(weights, WEIGHTS, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&number_of_sequences, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    
    //START THE LOGIC --------------------> MASTER WORKER DYNAMIC ARCHITECTURE 
    if(my_rank == ROOT)
    {
        masterProcess(number_of_sequences, all_seq, seq1, num_procs);
    }
    else
    {
        workerProcess(number_of_sequences, seq1, weights);
    }
    
    
    
    //FREE ALL DATA
    
    if(my_rank == ROOT)
    {
        for (int i = 0; i < number_of_sequences; i++)
        {
            free(all_seq[i]);
        }
        free(all_seq);
    }

    free(seq1);
    
    if(my_rank == ROOT)
        printf("the time it took is %lf seconds\n", time(NULL) - start);
    
    MPI_Finalize();
    return 0;
}

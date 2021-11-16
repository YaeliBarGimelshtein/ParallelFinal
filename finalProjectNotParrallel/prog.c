#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <ctype.h>
#include <time.h>


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


    for (int offset = 0; offset <= offset_max; offset++) //all offsets
    {
        for (int mutant = 0; mutant < possible_mutants; mutant++) //all mutants
        {
            index_seq1 = offset;
            for (int index_seq2 = 0; index_seq2 < lenght_seq2; index_seq2++)
            {
                //MUTATION POSSIBLE ONLY IF NOT THE FINAL OFFSET
                if(offset != offset_max)
                {
                    if (index_seq2 == mutant && mutant != 0)
                        index_seq1++;
                }
                
                
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

            
            if(curr_score > res[0])
            {
                res[0] = curr_score;
                res[1] = offset;
                res[2] = mutant;
            }
            
            curr_score = 0;
        }
    }
    return res;
}





int main(int argc, char *argv[])
{
    double start = time(NULL);
    //GENERAL DATA NEEDED
    int number_of_sequences;
    char* seq1 = (char*)malloc(sizeof(char) * MAX_SEQ1);
    int weights[WEIGHTS];
    char** all_seq;


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

    //RESULTS
    int** res = (int**)malloc(sizeof(int*) * number_of_sequences);

    for (int i = 0; i < number_of_sequences; i++)
    {
        res[i] = (int*)malloc(sizeof(int) * RES_INFO);
    }

    for (int i = 0; i < number_of_sequences; i++)
    {
        int offset_max = strlen(seq1) - strlen(all_seq[i]);
        res[i] = return_omp_score_offset_mutant(seq1, all_seq[i], weights, offset_max);
    }
    
    //PRINT THE RES
    printf("-----------------------------------Non Parallel Program-----------------------------------------------\n");
    printf("\n");
    printf("Seq1 is %s\n", seq1);
    printf("\nHere are all the best matches for the strings:\n");
    
    for (int i = 0; i < number_of_sequences; i++)
    {
        printf("seq2 = %s\n", all_seq[i]);
        if(res[i][2] == 0) //if mutation = 0 then change it to max lenght
            res[i][2] = strlen(all_seq[i]);
        printf("offset (n) = %d , mutation (k) = %d , score = %d \n", res[i][1], res[i][2], res[i][0]);
        printf("\n");
    }
    
    
    //FREE ALL DATA
    for (int i = 0; i < number_of_sequences; i++)
    {
        free(all_seq[i]);
    }
    free(all_seq);

    for (int i = 0; i < number_of_sequences; i++)
    {
        free(res[i]);
    }
    free(res);
    

    free(seq1);
    printf("the time it took is %lf seconds\n", time(NULL) - start);
    return 0;
}

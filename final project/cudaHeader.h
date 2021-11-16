#pragma once

#define NUM_THREADS_PER_BLOCK 256


int* return_cuda_score_offset_mutant(char* seq1, char* seq2 , int* weights, int offset_start, int offset_finish);

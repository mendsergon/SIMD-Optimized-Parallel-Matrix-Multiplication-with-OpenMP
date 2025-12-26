#ifndef MATRIX_MULT_SIMD_H
#define MATRIX_MULT_SIMD_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include <immintrin.h>

float** create_matrix(int rows, int cols);
void free_matrix(float** matrix, int rows);
void initialize_matrix(float** matrix, int rows, int cols);
void initialize_matrix_random(float** matrix, int rows, int cols);
void print_matrix(float** matrix, int rows, int cols);
void matrix_multiply_serial(float** A, float** B, float** C, int M, int N, int W);
void matrix_multiply_parallel_tasks_simd_avx(float** A, float** B, float** C, int M, int N, int W, int num_threads);
void matrix_multiply_parallel_tasks_simd_sse(float** A, float** B, float** C, int M, int N, int W, int num_threads);
void matrix_multiply_parallel_tasks_simd_avx_optimized(float** A, float** B, float** C, int M, int N, int W, int num_threads);
void matrix_multiply_parallel_for_simd_avx(float** A, float** B, float** C, int M, int N, int W, int num_threads);
void check_simd_support();
void transpose_matrix(float** src, float** dst, int rows, int cols);

#endif

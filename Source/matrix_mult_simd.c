#include "matrix_mult_simd.h"
#include <string.h>

// Helper function to sum all elements in an AVX vector (8 floats)
static inline float horizontal_sum_avx(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);    // Extract low 128 bits
    __m128 vhigh = _mm256_extractf128_ps(v, 1); // Extract high 128 bits
    vlow = _mm_add_ps(vlow, vhigh);             // Add high and low parts
    __m128 shuf = _mm_movehdup_ps(vlow);        // Duplicate odd elements
    __m128 sums = _mm_add_ps(vlow, shuf);       // Add even and odd
    shuf = _mm_movehl_ps(shuf, sums);           // Move high to low
    sums = _mm_add_ss(sums, shuf);              // Add remaining elements
    return _mm_cvtss_f32(sums);                 // Convert to float
}

// Helper function to sum all elements in an SSE vector (4 floats)
static inline float horizontal_sum_sse(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);           // Duplicate odd elements
    __m128 sums = _mm_add_ps(v, shuf);          // Add even and odd
    shuf = _mm_movehl_ps(shuf, sums);           // Move high to low
    sums = _mm_add_ss(sums, shuf);              // Add remaining elements
    return _mm_cvtss_f32(sums);                 // Convert to float
}

// Create matrix with 32-byte alignment for AVX
float** create_matrix(int rows, int cols) {
    float** matrix = (float**)aligned_alloc(32, rows * sizeof(float*));
    if (matrix == NULL) {
        fprintf(stderr, "Memory allocation failed for matrix rows\n");
        exit(1);
    }
    
    for (int i = 0; i < rows; i++) {
        matrix[i] = (float*)aligned_alloc(32, cols * sizeof(float));
        if (matrix[i] == NULL) {
            fprintf(stderr, "Memory allocation failed for matrix columns\n");
            exit(1);
        }
    }
    
    return matrix;
}

// Free aligned matrix memory
void free_matrix(float** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Initialize matrix with sequential values (1, 2, 3...)
void initialize_matrix(float** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = (float)(i * cols + j + 1);
        }
    }
}

// Initialize matrix with random values between 0.0 and 9.9
void initialize_matrix_random(float** matrix, int rows, int cols) {
    static int seeded = 0;
    if (!seeded) {
        srand(time(NULL));
        seeded = 1;
    }
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = (float)(rand() % 100) / 10.0f;
        }
    }
}

// Print first 5x5 elements of matrix for debugging
void print_matrix(float** matrix, int rows, int cols) {
    int max_rows = (rows > 5) ? 5 : rows;
    int max_cols = (cols > 5) ? 5 : cols;
    
    for (int i = 0; i < max_rows; i++) {
        for (int j = 0; j < max_cols; j++) {
            printf("%8.2f ", matrix[i][j]);
        }
        if (cols > max_cols) printf("...");
        printf("\n");
    }
    if (rows > max_rows) printf("... (%d more rows)\n", rows - max_rows);
}

// Check which SIMD instructions are supported by the CPU
void check_simd_support() {
    printf("SIMD Support Check:\n");
    
    #ifdef __AVX2__
    printf("  AVX2: Supported\n");
    #else
    printf("  AVX2: Not supported\n");
    #endif
    
    #ifdef __AVX__
    printf("  AVX: Supported\n");
    #else
    printf("  AVX: Not supported\n");
    #endif
    
    #ifdef __SSE4_2__
    printf("  SSE4.2: Supported\n");
    #else
    printf("  SSE4.2: Not supported\n");
    #endif
}

// Serial matrix multiplication for comparison (baseline)
void matrix_multiply_serial(float** A, float** B, float** C, int M, int N, int W) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < W; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// Parallel matrix multiplication using OpenMP tasks with AVX SIMD (8 floats per vector)
void matrix_multiply_parallel_tasks_simd_avx(float** A, float** B, float** C, int M, int N, int W, int num_threads) {
    omp_set_num_threads(num_threads);
    
    // Initialize result matrix to 0 in parallel
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < W; j++) {
            C[i][j] = 0.0f;
        }
    }
    
    // Create tasks for each column of B
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int j = 0; j < W; j++) {
                #pragma omp task firstprivate(j)
                {
                    const int SIMD_WIDTH = 8;  // AVX processes 8 floats
                    
                    for (int i = 0; i < M; i++) {
                        __m256 sum_vec = _mm256_setzero_ps();
                        int k;
                        
                        // Main SIMD loop - process 8 elements at a time
                        for (k = 0; k <= N - SIMD_WIDTH; k += SIMD_WIDTH) {
                            // Load 8 consecutive elements from row i of A
                            __m256 a_vec = _mm256_load_ps(&A[i][k]);
                            
                            // Load 8 elements from column j of B (non-contiguous, need to gather)
                            float b_temp[8];
                            for (int t = 0; t < 8; t++) {
                                b_temp[t] = B[k + t][j];
                            }
                            __m256 b_vec = _mm256_load_ps(b_temp);
                            
                            // Fused multiply-add: sum_vec += a_vec * b_vec
                            sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
                        }
                        
                        // Sum all elements in the AVX vector
                        float sum = horizontal_sum_avx(sum_vec);
                        
                        // Process remaining elements (N % 8)
                        for (; k < N; k++) {
                            sum += A[i][k] * B[k][j];
                        }
                        
                        C[i][j] = sum;
                    }
                }
            }
        }
    }
}

// Parallel matrix multiplication using OpenMP tasks with SSE SIMD (4 floats per vector)
void matrix_multiply_parallel_tasks_simd_sse(float** A, float** B, float** C, int M, int N, int W, int num_threads) {
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < W; j++) {
            C[i][j] = 0.0f;
        }
    }
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int j = 0; j < W; j++) {
                #pragma omp task firstprivate(j)
                {
                    const int SIMD_WIDTH = 4;  // SSE processes 4 floats
                    
                    for (int i = 0; i < M; i++) {
                        __m128 sum_vec = _mm_setzero_ps();
                        int k;
                        
                        for (k = 0; k <= N - SIMD_WIDTH; k += SIMD_WIDTH) {
                            __m128 a_vec = _mm_load_ps(&A[i][k]);
                            
                            float b_temp[4];
                            for (int t = 0; t < 4; t++) {
                                b_temp[t] = B[k + t][j];
                            }
                            __m128 b_vec = _mm_load_ps(b_temp);
                            
                            sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(a_vec, b_vec));
                        }
                        
                        float sum = horizontal_sum_sse(sum_vec);
                        
                        for (; k < N; k++) {
                            sum += A[i][k] * B[k][j];
                        }
                        
                        C[i][j] = sum;
                    }
                }
            }
        }
    }
}

// Optimized version: transpose B first for contiguous memory access in SIMD loads
void matrix_multiply_parallel_tasks_simd_avx_optimized(float** A, float** B, float** C, int M, int N, int W, int num_threads) {
    omp_set_num_threads(num_threads);
    
    // Transpose B so we can load contiguous memory in SIMD
    float** Bt = create_matrix(W, N);
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < W; j++) {
            Bt[j][i] = B[i][j];
        }
    }
    
    // Initialize result matrix
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < W; j++) {
            C[i][j] = 0.0f;
        }
    }
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int j = 0; j < W; j++) {
                #pragma omp task firstprivate(j)
                {
                    const int SIMD_WIDTH = 8;
                    
                    for (int i = 0; i < M; i++) {
                        __m256 sum_vec = _mm256_setzero_ps();
                        int k;
                        
                        // Now Bt[j][k] is contiguous memory - perfect for SIMD
                        for (k = 0; k <= N - SIMD_WIDTH; k += SIMD_WIDTH) {
                            __m256 a_vec = _mm256_load_ps(&A[i][k]);
                            __m256 b_vec = _mm256_load_ps(&Bt[j][k]);  // Contiguous load
                            sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
                        }
                        
                        float sum = horizontal_sum_avx(sum_vec);
                        
                        for (; k < N; k++) {
                            sum += A[i][k] * Bt[j][k];
                        }
                        
                        C[i][j] = sum;
                    }
                }
            }
        }
    }
    
    free_matrix(Bt, W);
}

// Parallel for loop version with AVX SIMD (for comparison with tasks)
void matrix_multiply_parallel_for_simd_avx(float** A, float** B, float** C, int M, int N, int W, int num_threads) {
    omp_set_num_threads(num_threads);
    
    // Transpose B for better memory access
    float** Bt = create_matrix(W, N);
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < W; j++) {
            Bt[j][i] = B[i][j];
        }
    }
    
    // Parallelize over columns using parallel for (not tasks)
    #pragma omp parallel for
    for (int j = 0; j < W; j++) {
        const int SIMD_WIDTH = 8;
        
        for (int i = 0; i < M; i++) {
            __m256 sum_vec = _mm256_setzero_ps();
            int k;
            
            for (k = 0; k <= N - SIMD_WIDTH; k += SIMD_WIDTH) {
                __m256 a_vec = _mm256_load_ps(&A[i][k]);
                __m256 b_vec = _mm256_load_ps(&Bt[j][k]);
                sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
            }
            
            float sum = horizontal_sum_avx(sum_vec);
            
            for (; k < N; k++) {
                sum += A[i][k] * Bt[j][k];
            }
            
            C[i][j] = sum;
        }
    }
    
    free_matrix(Bt, W);
}

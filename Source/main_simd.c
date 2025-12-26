#include "matrix_mult_simd.h"
#include <string.h>
#include <ctype.h>

// Function to verify if two matrices are equal
int verify_matrices(float** C1, float** C2, int M, int W, float tolerance) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < W; j++) {
            if (fabs(C1[i][j] - C2[i][j]) > tolerance) {
                printf("Mismatch at [%d][%d]: %.6f vs %.6f\n", 
                       i, j, C1[i][j], C2[i][j]);
                return 0;
            }
        }
    }
    return 1;
}

// Function to clear input buffer
void clear_input_buffer() {
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}

// Function to get positive integer input
int get_positive_int(const char* prompt, int default_value) {
    char input[100];
    int value;
    
    while (1) {
        printf("%s (default: %d): ", prompt, default_value);
        if (fgets(input, sizeof(input), stdin) == NULL) {
            return default_value;
        }
        
        if (input[0] == '\n') {
            return default_value;
        }
        
        if (sscanf(input, "%d", &value) == 1 && value > 0) {
            return value;
        }
        
        printf("Invalid input! Please enter a positive integer.\n");
    }
}

// Run performance test with different implementations
void run_simd_test(int M, int N, int W, int use_random, int num_threads) {
    printf("\n=== SIMD Performance Test: M=%d, N=%d, W=%d, Threads=%d ===\n", M, N, W, num_threads);
    
    // Create matrices
    float** A = create_matrix(M, N);
    float** B = create_matrix(N, W);
    float** C_serial = create_matrix(M, W);
    float** C_tasks_avx = create_matrix(M, W);
    float** C_tasks_sse = create_matrix(M, W);
    float** C_for_avx = create_matrix(M, W);
    float** C_tasks_avx_opt = create_matrix(M, W);
    
    // Initialize matrices
    if (use_random) {
        srand(time(NULL));
        initialize_matrix_random(A, M, N);
        initialize_matrix_random(B, N, W);
    } else {
        initialize_matrix(A, M, N);
        initialize_matrix(B, N, W);
    }
    
    // Warm-up run
    matrix_multiply_serial(A, B, C_serial, M, N, W);
    
    // 1. Serial baseline
    double start = omp_get_wtime();
    matrix_multiply_serial(A, B, C_serial, M, N, W);
    double serial_time = omp_get_wtime() - start;
    
    // 2. AVX Tasks
    start = omp_get_wtime();
    matrix_multiply_parallel_tasks_simd_avx(A, B, C_tasks_avx, M, N, W, num_threads);
    double tasks_avx_time = omp_get_wtime() - start;
    
    // 3. SSE Tasks
    start = omp_get_wtime();
    matrix_multiply_parallel_tasks_simd_sse(A, B, C_tasks_sse, M, N, W, num_threads);
    double tasks_sse_time = omp_get_wtime() - start;
    
    // 4. Optimized AVX Tasks (with transposed B)
    start = omp_get_wtime();
    matrix_multiply_parallel_tasks_simd_avx_optimized(A, B, C_tasks_avx_opt, M, N, W, num_threads);
    double tasks_avx_opt_time = omp_get_wtime() - start;
    
    // 5. AVX Parallel For
    start = omp_get_wtime();
    matrix_multiply_parallel_for_simd_avx(A, B, C_for_avx, M, N, W, num_threads);
    double for_avx_time = omp_get_wtime() - start;
    
    // Verify results
    int avx_correct = verify_matrices(C_serial, C_tasks_avx, M, W, 0.001f);
    int sse_correct = verify_matrices(C_serial, C_tasks_sse, M, W, 0.001f);
    int avx_opt_correct = verify_matrices(C_serial, C_tasks_avx_opt, M, W, 0.001f);
    int for_avx_correct = verify_matrices(C_serial, C_for_avx, M, W, 0.001f);
    
    // Calculate speedups
    double speedup_avx = (tasks_avx_time > 0) ? serial_time / tasks_avx_time : 0;
    double speedup_sse = (tasks_sse_time > 0) ? serial_time / tasks_sse_time : 0;
    double speedup_avx_opt = (tasks_avx_opt_time > 0) ? serial_time / tasks_avx_opt_time : 0;
    double speedup_for_avx = (for_avx_time > 0) ? serial_time / for_avx_time : 0;
    
    // Print results
    printf("\nPerformance Results:\n");
    printf("-----------------------------------------------------\n");
    printf("Implementation               Time (s)     Speedup    Correct\n");
    printf("-----------------------------------------------------\n");
    printf("Serial                      %9.6f     1.00x      N/A\n", serial_time);
    printf("Parallel Tasks + AVX        %9.6f     %6.2fx      %s\n", 
           tasks_avx_time, speedup_avx, avx_correct ? "Yes" : "No");
    printf("Parallel Tasks + SSE        %9.6f     %6.2fx      %s\n", 
           tasks_sse_time, speedup_sse, sse_correct ? "Yes" : "No");
    printf("Parallel Tasks + AVX Opt    %9.6f     %6.2fx      %s\n", 
           tasks_avx_opt_time, speedup_avx_opt, avx_opt_correct ? "Yes" : "No");
    printf("Parallel For + AVX          %9.6f     %6.2fx      %s\n", 
           for_avx_time, speedup_for_avx, for_avx_correct ? "Yes" : "No");
    printf("-----------------------------------------------------\n");
    
    // Calculate efficiency
    double efficiency_avx = (speedup_avx / num_threads) * 100.0;
    double efficiency_avx_opt = (speedup_avx_opt / num_threads) * 100.0;
    printf("\nEfficiency (AVX Tasks): %.1f%%\n", efficiency_avx);
    printf("Efficiency (AVX Opt):   %.1f%%\n", efficiency_avx_opt);
    
    // Cleanup
    free_matrix(A, M);
    free_matrix(B, N);
    free_matrix(C_serial, M);
    free_matrix(C_tasks_avx, M);
    free_matrix(C_tasks_sse, M);
    free_matrix(C_tasks_avx_opt, M);
    free_matrix(C_for_avx, M);
}

// Run scaling test with SIMD
void run_scaling_test_simd(int base_threads) {
    printf("\n===================================================\n");
    printf("SIMD THREAD SCALING ANALYSIS (M=300, N=300, W=300)\n");
    printf("===================================================\n");
    
    printf("\nThreads | Serial Time | AVX Tasks Time | Speedup | Efficiency\n");
    printf("--------|-------------|----------------|---------|-----------\n");
    
    int max_threads = base_threads * 2;
    if (max_threads > 64) max_threads = 64;
    
    for (int threads = 1; threads <= max_threads; threads += (threads < 8 ? 1 : 2)) {
        int M = 300, N = 300, W = 300;
        
        float** A = create_matrix(M, N);
        float** B = create_matrix(N, W);
        float** C_serial = create_matrix(M, W);
        float** C_avx = create_matrix(M, W);
        
        initialize_matrix_random(A, M, N);
        initialize_matrix_random(B, N, W);
        
        // Warm-up
        matrix_multiply_serial(A, B, C_serial, M, N, W);
        
        double start = omp_get_wtime();
        matrix_multiply_serial(A, B, C_serial, M, N, W);
        double serial_time = omp_get_wtime() - start;
        
        start = omp_get_wtime();
        matrix_multiply_parallel_tasks_simd_avx_optimized(A, B, C_avx, M, N, W, threads);
        double avx_time = omp_get_wtime() - start;
        
        double speedup = (avx_time > 0) ? serial_time / avx_time : 0;
        double efficiency = (threads > 0) ? (speedup / threads) * 100.0 : 0;
        
        printf("%7d | %11.4f | %14.4f | %7.2f | %9.1f%%\n",
               threads, serial_time, avx_time, speedup, efficiency);
        
        free_matrix(A, M);
        free_matrix(B, N);
        free_matrix(C_serial, M);
        free_matrix(C_avx, M);
    }
}

// Interactive mode for SIMD testing
void interactive_simd_mode(int num_threads) {
    char choice;
    
    do {
        printf("\n===================================================\n");
        printf("SIMD INTERACTIVE MODE\n");
        printf("===================================================\n");
        
        int M = get_positive_int("Enter rows for matrix A (M)", 256);
        int N = get_positive_int("Enter columns for matrix A / rows for matrix B (N)", 256);
        int W = get_positive_int("Enter columns for matrix B (W)", 256);
        
        int threads = get_positive_int("Enter number of threads", num_threads);
        
        printf("\nInitialize matrices with:\n");
        printf("  1) Sequential values (1, 2, 3...)\n");
        printf("  2) Random values\n");
        printf("Your choice (1 or 2, default: 2): ");
        
        char init_choice[10];
        fgets(init_choice, sizeof(init_choice), stdin);
        int use_random = (init_choice[0] == '1') ? 0 : 1;
        
        run_simd_test(M, N, W, use_random, threads);
        
        printf("\nRun another SIMD test? (y/n): ");
        fgets(&choice, sizeof(choice), stdin);
        
    } while (choice == 'y' || choice == 'Y');
}

int main() {
    printf("===================================================\n");
    printf("SIMD-OPTIMIZED PARALLEL MATRIX MULTIPLICATION\n");
    printf("===================================================\n");
    
    // Check SIMD support
    check_simd_support();
    
    // Get thread count
    int max_threads = omp_get_max_threads();
    printf("\nMaximum threads available: %d\n", max_threads);
    
    int num_threads = get_positive_int("Enter number of threads to use for tests", max_threads);
    omp_set_num_threads(num_threads);
    
    // Run comprehensive test suite
    printf("\n===================================================\n");
    printf("RUNNING SIMD TEST SUITE\n");
    printf("===================================================\n");
    
    printf("\n1. Small test (verification, M=8, N=8, W=8):\n");
    run_simd_test(8, 8, 8, 0, num_threads);
    
    printf("\n2. Medium test (M=128, N=128, W=128):\n");
    run_simd_test(128, 128, 128, 1, num_threads);
    
    printf("\n3. Large test (M=512, N=512, W=512):\n");
    run_simd_test(512, 512, 512, 1, num_threads);
    
    printf("\n4. Tall matrix test (M=1024, N=64, W=128):\n");
    run_simd_test(1024, 64, 128, 1, num_threads);
    
    printf("\n5. Wide matrix test (M=64, N=128, W=1024):\n");
    run_simd_test(64, 128, 1024, 1, num_threads);
    
    // Run scaling analysis
    run_scaling_test_simd(num_threads);
    
    // Interactive mode
    interactive_simd_mode(num_threads);
    
    printf("\n===================================================\n");
    printf("SIMD TESTING COMPLETED\n");
    printf("===================================================\n");
    
    return 0;
}

# **SIMD-Optimized Parallel Matrix Multiplication with OpenMP**

### **Project Summary**

This C program implements a **high-performance matrix multiplication benchmark** combining **OpenMP parallelism** with **SIMD vectorization** using AVX and SSE instructions. The system compares multiple optimization strategies, measuring performance improvements from both parallelization and vectorization techniques across varying matrix dimensions and thread counts.

---

### **Core Features**

* **Five Implementation Strategies**:
  * **Serial Baseline**: Traditional triple-nested loop implementation
  * **Parallel Tasks + AVX**: OpenMP tasks with AVX-256 (8 floats per vector)
  * **Parallel Tasks + SSE**: OpenMP tasks with SSE (4 floats per vector)
  * **Parallel Tasks + AVX Optimized**: Pre-transposed B matrix for contiguous memory access
  * **Parallel For + AVX**: OpenMP parallel for with AVX vectorization

* **Advanced SIMD Features**:
  * **AVX-256 Vectorization**: Processes 8 floats simultaneously using fused multiply-add
  * **SSE Vectorization**: Processes 4 floats simultaneously
  * **Memory Alignment**: 32-byte aligned allocations for optimal SIMD performance
  * **Horizontal Vector Reduction**: Efficient summing of vector elements

* **Comprehensive Benchmarking**:
  * Measures execution times for all implementations
  * Calculates speedup relative to serial baseline
  * Computes parallel efficiency metrics
  * Verifies correctness across all implementations

* **Multi-dimensional Testing**:
  * Small matrices for verification (8×8×8)
  * Medium/large square matrices (128/512)
  * Tall and wide rectangular matrices
  * Thread scaling analysis (1 to 128 threads)

* **Interactive Testing Mode**: User-defined dimensions, thread counts, and initialization methods

---

### **Key Methods and Algorithms**

#### **1. Memory Management with Alignment**
```c
// 32-byte aligned allocation for AVX
float** matrix = (float**)aligned_alloc(32, rows * sizeof(float*));
matrix[i] = (float*)aligned_alloc(32, cols * sizeof(float));
```

#### **2. Serial Baseline (Reference)**
```c
for (int i = 0; i < M; i++) {
    for (int j = 0; j < W; j++) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[i][k] * B[k][j];  // Standard triple loop
        }
        C[i][j] = sum;
    }
}
```

#### **3. Parallel Tasks with AVX Vectorization**
```c
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
                    
                    // Process 8 elements at once
                    for (k = 0; k <= N - SIMD_WIDTH; k += SIMD_WIDTH) {
                        __m256 a_vec = _mm256_load_ps(&A[i][k]);
                        __m256 b_vec = _mm256_load_ps(&Bt[j][k]);  // Contiguous after transpose
                        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);  // FMA instruction
                    }
                    
                    // Horizontal reduction and remainder processing
                    float sum = horizontal_sum_avx(sum_vec);
                    for (; k < N; k++) sum += A[i][k] * Bt[j][k];
                    
                    C[i][j] = sum;
                }
            }
        }
    }
}
```

#### **4. Optimized Matrix Transposition**
```c
// Transpose B for contiguous memory access in SIMD loads
float** Bt = create_matrix(W, N);
#pragma omp parallel for collapse(2)
for (int i = 0; i < N; i++) {
    for (int j = 0; j < W; j++) {
        Bt[j][i] = B[i][j];  // Column-major to row-major
    }
}
```

#### **5. Efficient Horizontal Vector Reduction**
```c
static inline float horizontal_sum_avx(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);      // Add high and low
    __m128 shuf = _mm_movehdup_ps(vlow); // Duplicate odd elements
    __m128 sums = _mm_add_ps(vlow, shuf);// Add even and odd
    shuf = _mm_movehl_ps(shuf, sums);    // Move high to low
    sums = _mm_add_ss(sums, shuf);       // Add remaining
    return _mm_cvtss_f32(sums);          // Convert to float
}
```

#### **6. Performance Metrics**
* **Timing**: `omp_get_wtime()` for high-precision measurements
* **Speedup**: `serial_time / parallel_simd_time`
* **Efficiency**: `(speedup / thread_count) × 100%`
* **Verification**: Tolerance-based comparison (0.001)

---

### **Skills Demonstrated**

* **SIMD Programming**: AVX-256 and SSE intrinsics for data-level parallelism
* **Vectorization Techniques**: Fused multiply-add, horizontal reduction, aligned memory access
* **OpenMP Parallelism**: Task-based and loop-based parallelization strategies
* **Memory Optimization**: Matrix transposition for contiguous access patterns
* **Performance Analysis**: Multi-factorial benchmarking (threads × SIMD × algorithms)
* **Numerical Computing**: Floating-point arithmetic with error analysis
* **System Programming**: CPU feature detection, aligned memory management

---

### **File Overview**

| File | Description |
| :--- | :--- |
| **main_simd.c** | Main program with test suites, interactive mode, and performance analysis |
| **matrix_mult_simd.h** | Header with function declarations and SIMD intrinsics includes |
| **matrix_mult_simd.c** | Implementation of all matrix operations with SIMD optimizations |
| **Makefile_simd** | Build configuration with SIMD compiler flags and optimizations |

---

### **Hardware Requirements**

* **CPU Support**:
  * **AVX**: Intel Sandy Bridge (2011+) or AMD Bulldozer (2011+)
  * **AVX2/FMA**: Intel Haswell (2013+) or AMD Excavator (2015+)
  * **SSE4.2**: Most CPUs from 2008 onward
* **Memory**: 32-byte alignment support for AVX instructions
* **Compiler**: GCC with OpenMP and SIMD support

---

### **How to Compile and Run**

#### **1. Compilation**
```bash
# Standard build with AVX optimizations (auto-detects CPU features)
make -f Makefile_simd

# Debug build with symbols
make -f Makefile_simd debug

# SSE-only version (for older CPUs)
make -f Makefile_simd sse

# Clean build artifacts
make -f Makefile_simd clean
```

#### **2. Execution**
```bash
./matrix_mult_simd
# or
make -f Makefile_simd run
```

#### **3. Program Flow**
1. **SIMD Detection**: Checks available CPU vector extensions
2. **Thread Configuration**: User specifies thread count (defaults to CPU cores)
3. **Comprehensive Test Suite**:
   - Small verification test (8×8×8)
   - Medium random matrices (128×128×128)
   - Large random matrices (512×512×512)
   - Tall matrices (1024×64×128)
   - Wide matrices (64×128×1024)
4. **Scaling Analysis**: Tests 1 to 2×core_count threads on 300×300×300 matrices
5. **Interactive Mode**: Custom tests with user-defined parameters

#### **4. Interactive Mode Options**
- Matrix dimensions (M×N×W)
- Thread count (1 to available cores)
- Initialization: sequential (1,2,3...) or random values
- Multiple test iterations

#### **5. Output Interpretation**
```
=== SIMD Performance Test: M=512, N=512, W=512, Threads=8 ===

Implementation               Time (s)     Speedup    Correct
-------------------------------------------------------------
Serial                      2.456789     1.00x      N/A
Parallel Tasks + AVX        0.198765    12.36x      Yes
Parallel Tasks + SSE        0.345678     7.11x      Yes
Parallel Tasks + AVX Opt    0.154321    15.92x      Yes
Parallel For + AVX          0.167890    14.64x      Yes

Efficiency (AVX Tasks): 154.5%
Efficiency (AVX Opt):   199.0%
```

- **Speedup > 1**: Parallel faster than serial
- **Efficiency > 100%**: Indicates benefits from both parallelization AND vectorization
- **"Correct: Yes"**: All implementations produce identical results within tolerance
- **AVX Opt vs AVX**: Shows benefit of memory access optimization

---

### **Performance Characteristics**

#### **Optimal Use Cases**
- **Large Square Matrices**: Best parallel efficiency (512×512+)
- **Memory-bound Problems**: AVX optimization reduces memory bandwidth pressure
- **CPU-intensive Workloads**: High core count + SIMD provides maximum throughput

#### **Key Optimizations**
1. **Memory Access Pattern**: Transposing B enables contiguous SIMD loads
2. **Vectorization**: 8× speedup from AVX alone (theoretical maximum)
3. **Parallelization**: Near-linear scaling for large matrices
4. **Fused Operations**: FMA instructions combine multiply and add

#### **Expected Performance**
- **AVX + Parallelization**: 10-20× speedup over serial (depending on matrix size)
- **SSE vs AVX**: AVX typically 1.5-2× faster than SSE
- **Optimized vs Basic**: 20-30% improvement from memory access optimization
- **Super-linear Speedup**: Possible due to better cache utilization

#### **Limitations**
- **Memory Alignment**: Required for optimal SIMD performance
- **CPU Dependency**: Requires modern CPUs with AVX/AVX2 support
- **Matrix Size**: Smaller matrices (<64×64) may not benefit from parallelism
- **Remainder Handling**: Non-multiples of 8 require scalar processing

---

### **Technical Details**

#### **SIMD Intrinsics Used**
- `_mm256_load_ps()`: Load 8 floats with 32-byte alignment
- `_mm256_fmadd_ps()`: Fused multiply-add (8 operations in one instruction)
- `_mm256_setzero_ps()`: Initialize vector to zeros
- `_mm256_extractf128_ps()`: Extract 128-bit lane from 256-bit vector

#### **Compiler Flags**
- `-march=native`: Optimize for current CPU architecture
- `-mavx -mfma`: Enable AVX and FMA instruction sets
- `-fopenmp`: Enable OpenMP parallelization
- `-O3`: Maximum optimization level

#### **Memory Considerations**
- **Working Set Size**: ~3 × M × N × sizeof(float) bytes
- **Alignment**: All allocations 32-byte aligned for AVX
- **Cache Effects**: Transposition improves spatial locality

#ifndef PMOO_ALGORITHMS_UTILS_CUH
#define PMOO_ALGORITHMS_UTILS_CUH

#include <stdio.h>

#define TOLERANCE_EPSILON 1.0E-6

#if _OPENMP >= 200805 // OpenMP 3.0 or newer
#define OMP_FOR_INDEX unsigned
#else // Older OpenMP implementations do not support unsigned indexes
#define OMP_FOR_INDEX int
#endif

#define UNIFORM_DOUBLE_TO_RANGE(x, min, max) x * (max - min) + min

#define CUDA_CALL(ans) { cuda_call((ans), __FILE__, __LINE__); }
inline void cuda_call(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        printf("Error at %s:%d\n", file, line);
        printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
        if (abort) exit(code);
    }
}

template <unsigned SIZE>
struct array_with_size {
    unsigned size;              // Stores how many elements are present in the array
    unsigned elements[SIZE];    // Array elements
};

__host__ __device__ bool pareto_dominance(const double *point1, const double *point2, const unsigned d_dim);

// helper function for SBX
__host__ __device__ double sbx_betaq(double beta, double crossover_index, double rnd);

#endif
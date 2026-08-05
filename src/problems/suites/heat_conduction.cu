/**
 * Heat Conduction Multi-Objective Optimization Problem
 * VERSION 5.1 - GPU GLOBAL MEMORY (FIXED)
 *
 * Fixes from v5.0:
 * 1. Fixed thread_id bounds checking to prevent buffer overflow
 * 2. Fixed NULL pointer checks for device buffers
 * 3. Fixed function pointer compatibility
 * 4. Added proper error handling
 *
 * Author: Mohamed Habdallah
 * Date: January 2026
 */

#include "problems/suites/heat_conduction.cuh"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <cuda_runtime.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef GRID_SIZE_X
#define GRID_SIZE_X 64
#endif
#ifndef GRID_SIZE_Y
#define GRID_SIZE_Y 64
#endif
#ifndef TIME_STEPS
#define TIME_STEPS 200
#endif
#ifndef DOMAIN_LENGTH
#define DOMAIN_LENGTH 1.0
#endif
#ifndef AMBIENT_TEMP
#define AMBIENT_TEMP 20.0
#endif
#ifndef THERMAL_DIFFUSIVITY
#define THERMAL_DIFFUSIVITY 0.01
#endif
#ifndef DT
#define DT 0.0005
#endif
#ifndef MAX_SOURCE_INTENSITY
#define MAX_SOURCE_INTENSITY 100.0
#endif
#ifndef TARGET_TEMP
#define TARGET_TEMP 25.0
#endif
#ifndef CALIBRATION_SAMPLES
#define CALIBRATION_SAMPLES 2000
#endif
#ifndef PARETO_FRONT_SIZE
#define PARETO_FRONT_SIZE 100
#endif

// Population size for pre-allocating GPU buffers
#ifndef MAX_POPULATION_SIZE
#define MAX_POPULATION_SIZE 1024
#endif

#define MAX_SOURCES 20
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

// ============================================================================
// GPU MEMORY - GLOBAL ALLOCATION
// ============================================================================

// GPU memory for kernel-based solver (host-side)
static double *d_T_current = NULL;
static double *d_T_next = NULL;
static double *d_sources = NULL;
static double *d_results = NULL;
static int gpu_memory_initialized = 0;

// GPU memory for device-side solver (per-thread buffers in global memory)
static double *d_thread_T_buf0 = NULL;
static double *d_thread_T_buf1 = NULL;
static int gpu_thread_buffers_initialized = 0;

// Device pointers accessible from device code
__device__ double *dev_thread_T_buf0 = NULL;
__device__ double *dev_thread_T_buf1 = NULL;
__device__ int dev_buffers_ready = 0;  // FIX: Flag to check if buffers are initialized

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while(0)

// ============================================================================
// CUDA KERNELS FOR HOST-SIDE SOLVER
// ============================================================================

__global__ void kernel_init_temperature(double *T, int size, double ambient) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) T[idx] = ambient;
}

__global__ void kernel_apply_sources(double *T, const double *sources, int num_sources, 
                                      int GX, int GY, double ambient, double max_intensity) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < num_sources) {
        double x_norm = fmin(1.0, fmax(0.0, sources[s * 3]));
        double y_norm = fmin(1.0, fmax(0.0, sources[s * 3 + 1]));
        double intensity = fmin(max_intensity, fmax(0.0, sources[s * 3 + 2]));
        int x = (int)(x_norm * (GX - 1)); 
        int y = (int)(y_norm * (GY - 1));
        x = max(1, min(GX - 2, x)); 
        y = max(1, min(GY - 2, y));
        T[y * GX + x] = ambient + intensity;
    }
}

__global__ void kernel_heat_step(const double *T_current, double *T_next, 
                                  int GX, int GY, double r, double ambient) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < GX - 1 && j < GY - 1) {
        int idx = j * GX + i;
        double laplacian = T_current[idx - 1] + T_current[idx + 1] + 
                          T_current[idx - GX] + T_current[idx + GX] - 
                          4.0 * T_current[idx];
        T_next[idx] = T_current[idx] + r * laplacian;
    }
}

__global__ void kernel_apply_boundary(double *T, int GX, int GY, double ambient) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < GX) { 
        T[idx] = ambient; 
        T[(GY - 1) * GX + idx] = ambient; 
    }
    if (idx < GY) { 
        T[idx * GX] = ambient; 
        T[idx * GX + GX - 1] = ambient; 
    }
}

__global__ void kernel_compute_stats(const double *T, double *results, 
                                      int GX, int GY, double avg_for_variance) {
    __shared__ double s_max[256], s_sum[256], s_sum_sq[256];
    int tid = threadIdx.x;
    int interior_width = GX - 2, interior_height = GY - 2;
    int total_interior = interior_width * interior_height;
    double local_max = -1e30, local_sum = 0.0, local_sum_sq = 0.0;
    
    for (int k = blockIdx.x * blockDim.x + tid; k < total_interior; k += blockDim.x * gridDim.x) {
        int ii = k % interior_width + 1;
        int jj = k / interior_width + 1;
        double val = T[jj * GX + ii];
        if (val > local_max) local_max = val;
        local_sum += val;
        if (avg_for_variance > 0) { 
            double diff = val - avg_for_variance; 
            local_sum_sq += diff * diff; 
        }
    }
    
    s_max[tid] = local_max; 
    s_sum[tid] = local_sum; 
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_max[tid + stride] > s_max[tid]) s_max[tid] = s_max[tid + stride];
            s_sum[tid] += s_sum[tid + stride]; 
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicMax((unsigned long long*)&results[0], __double_as_longlong(s_max[0]));
        atomicAdd(&results[1], s_sum[0]); 
        atomicAdd(&results[2], s_sum_sq[0]);
    }
}

// ============================================================================
// GPU MEMORY MANAGEMENT
// ============================================================================

__host__ void init_gpu_memory() {
    if (gpu_memory_initialized) return;
    
    int grid_size = GRID_SIZE_X * GRID_SIZE_Y;
    
    CUDA_CHECK(cudaMalloc(&d_T_current, grid_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_T_next, grid_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sources, MAX_SOURCES * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_results, 4 * sizeof(double)));
    
    gpu_memory_initialized = 1;
}

__host__ void free_gpu_memory() {
    if (!gpu_memory_initialized) return;
    
    if (d_T_current) cudaFree(d_T_current);
    if (d_T_next) cudaFree(d_T_next);
    if (d_sources) cudaFree(d_sources);
    if (d_results) cudaFree(d_results);
    
    d_T_current = d_T_next = d_sources = d_results = NULL;
    gpu_memory_initialized = 0;
}

// ============================================================================
// GPU THREAD BUFFERS IN GLOBAL MEMORY (FIXED v5.1)
// ============================================================================

__host__ void init_gpu_thread_buffers() {
    if (gpu_thread_buffers_initialized) return;
    
    size_t grid_size = GRID_SIZE_X * GRID_SIZE_Y;
    size_t total_size = MAX_POPULATION_SIZE * grid_size * sizeof(double);
    
    cudaError_t err1 = cudaMalloc(&d_thread_T_buf0, total_size);
    cudaError_t err2 = cudaMalloc(&d_thread_T_buf1, total_size);
    
    if (err1 != cudaSuccess || err2 != cudaSuccess) {
        fprintf(stderr, "Failed to allocate GPU thread buffers: %s / %s\n",
                cudaGetErrorString(err1), cudaGetErrorString(err2));
        if (d_thread_T_buf0) cudaFree(d_thread_T_buf0);
        if (d_thread_T_buf1) cudaFree(d_thread_T_buf1);
        d_thread_T_buf0 = d_thread_T_buf1 = NULL;
        return;
    }
    
    // Copy pointers to device symbols
    CUDA_CHECK(cudaMemcpyToSymbol(dev_thread_T_buf0, &d_thread_T_buf0, sizeof(double*)));
    CUDA_CHECK(cudaMemcpyToSymbol(dev_thread_T_buf1, &d_thread_T_buf1, sizeof(double*)));
    
    // FIX: Set the ready flag
    int ready = 1;
    CUDA_CHECK(cudaMemcpyToSymbol(dev_buffers_ready, &ready, sizeof(int)));
    
    gpu_thread_buffers_initialized = 1;
}

__host__ void free_gpu_thread_buffers() {
    if (!gpu_thread_buffers_initialized) return;
    
    // Reset the ready flag first
    int ready = 0;
    cudaMemcpyToSymbol(dev_buffers_ready, &ready, sizeof(int));
    
    if (d_thread_T_buf0) cudaFree(d_thread_T_buf0);
    if (d_thread_T_buf1) cudaFree(d_thread_T_buf1);
    
    d_thread_T_buf0 = d_thread_T_buf1 = NULL;
    gpu_thread_buffers_initialized = 0;
}

// ============================================================================
// GPU KERNEL-BASED SOLVER (for host-side GPU calls)
// ============================================================================

__host__ void solve_heat_equation_2d_gpu_kernels(
    const double *heat_sources, 
    const unsigned num_sources, 
    double &max_temp, 
    double &avg_temp, 
    double &temp_variance
) {
    const int GX = GRID_SIZE_X;
    const int GY = GRID_SIZE_Y;
    const int grid_size = GX * GY;
    const int interior_count = (GX - 2) * (GY - 2);
    
    const double dx = DOMAIN_LENGTH / (GX - 1);
    double r = THERMAL_DIFFUSIVITY * DT / (dx * dx);
    
    int sub_steps = 1;
    if (r > 0.24) { 
        sub_steps = (int)ceil(r / 0.24); 
        if (sub_steps > 100) sub_steps = 100; 
        r = THERMAL_DIFFUSIVITY * DT / (dx * dx) / sub_steps; 
    }
    
    init_gpu_memory();
    
    dim3 block_init(256);
    dim3 grid_init((grid_size + 255) / 256);
    dim3 block_heat(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_heat((GX - 2 + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, 
                   (GY - 2 + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 block_boundary(256);
    dim3 grid_boundary((max(GX, GY) + 255) / 256);
    
    kernel_init_temperature<<<grid_init, block_init>>>(d_T_current, grid_size, AMBIENT_TEMP);
    kernel_init_temperature<<<grid_init, block_init>>>(d_T_next, grid_size, AMBIENT_TEMP);
    
    unsigned actual_sources = (num_sources < MAX_SOURCES) ? num_sources : MAX_SOURCES;
    if (actual_sources > 0) {
        CUDA_CHECK(cudaMemcpy(d_sources, heat_sources, 
                              actual_sources * 3 * sizeof(double), cudaMemcpyHostToDevice));
    }
    
    const unsigned total_steps = TIME_STEPS * sub_steps;
    
    for (unsigned t = 0; t < total_steps; ++t) {
        if (actual_sources > 0) {
            kernel_apply_sources<<<1, actual_sources>>>(
                d_T_current, d_sources, actual_sources, GX, GY, AMBIENT_TEMP, MAX_SOURCE_INTENSITY);
        }
        kernel_heat_step<<<grid_heat, block_heat>>>(d_T_current, d_T_next, GX, GY, r, AMBIENT_TEMP);
        kernel_apply_boundary<<<grid_boundary, block_boundary>>>(d_T_next, GX, GY, AMBIENT_TEMP);
        
        double *tmp = d_T_current; 
        d_T_current = d_T_next; 
        d_T_next = tmp;
    }
    
    double h_results[4] = {-1e30, 0.0, 0.0, 0.0};
    CUDA_CHECK(cudaMemcpy(d_results, h_results, 4 * sizeof(double), cudaMemcpyHostToDevice));
    kernel_compute_stats<<<32, 256>>>(d_T_current, d_results, GX, GY, 0.0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_results, d_results, 4 * sizeof(double), cudaMemcpyDeviceToHost));
    
    max_temp = h_results[0];
    avg_temp = h_results[1] / interior_count;
    
    h_results[2] = 0.0;
    CUDA_CHECK(cudaMemcpy(d_results, h_results, 4 * sizeof(double), cudaMemcpyHostToDevice));
    kernel_compute_stats<<<32, 256>>>(d_T_current, d_results, GX, GY, avg_temp);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_results, d_results, 4 * sizeof(double), cudaMemcpyDeviceToHost));
    
    temp_variance = h_results[2] / interior_count;
}

// ============================================================================
// CPU SOLVER (Thread-Local Buffers)
// ============================================================================

__host__ void solve_heat_equation_2d_cpu_serial(
    const double *heat_sources, 
    const unsigned num_sources, 
    double &max_temp, 
    double &avg_temp, 
    double &temp_variance
) {
    const int GX = GRID_SIZE_X;
    const int GY = GRID_SIZE_Y;
    const int grid_size = GX * GY;
    const int interior_count = (GX - 2) * (GY - 2);
    
    const double dx = DOMAIN_LENGTH / (GX - 1);
    double r = THERMAL_DIFFUSIVITY * DT / (dx * dx);
    
    int sub_steps = 1;
    if (r > 0.24) { 
        sub_steps = (int)ceil(r / 0.24); 
        if (sub_steps > 100) sub_steps = 100; 
        r = THERMAL_DIFFUSIVITY * DT / (dx * dx) / sub_steps; 
    }
    
    double *T_buf0 = (double*)malloc(grid_size * sizeof(double));
    double *T_buf1 = (double*)malloc(grid_size * sizeof(double));
    
    if (!T_buf0 || !T_buf1) {
        max_temp = 0.0; avg_temp = 0.0; temp_variance = 0.0;
        if (T_buf0) free(T_buf0);
        if (T_buf1) free(T_buf1);
        return;
    }
    
    for (int i = 0; i < grid_size; ++i) { 
        T_buf0[i] = AMBIENT_TEMP; 
        T_buf1[i] = AMBIENT_TEMP; 
    }
    
    int source_pos[MAX_SOURCES]; 
    double source_power[MAX_SOURCES];
    unsigned actual_sources = (num_sources < MAX_SOURCES) ? num_sources : MAX_SOURCES;
    
    for (unsigned s = 0; s < actual_sources; ++s) {
        double x_norm = fmax(0.0, fmin(1.0, heat_sources[s*3]));
        double y_norm = fmax(0.0, fmin(1.0, heat_sources[s*3 + 1]));
        int x = (int)(x_norm * (GX - 1));
        int y = (int)(y_norm * (GY - 1));
        x = (int)fmax(1, fmin(GX - 2, x));
        y = (int)fmax(1, fmin(GY - 2, y));
        source_pos[s] = y * GX + x;
        source_power[s] = fmax(0.0, fmin(MAX_SOURCE_INTENSITY, heat_sources[s*3 + 2]));
    }
    
    const unsigned total_steps = TIME_STEPS * sub_steps;
    
    for (unsigned t = 0; t < total_steps; ++t) {
        double *T_read  = (t % 2 == 0) ? T_buf0 : T_buf1;
        double *T_write = (t % 2 == 0) ? T_buf1 : T_buf0;
        
        for (unsigned s = 0; s < actual_sources; ++s) {
            T_read[source_pos[s]] = AMBIENT_TEMP + source_power[s];
        }
        
        for (int j = 1; j < GY - 1; ++j) {
            for (int i = 1; i < GX - 1; ++i) {
                int idx = j * GX + i;
                double laplacian = T_read[idx - 1] + T_read[idx + 1] + 
                                   T_read[idx - GX] + T_read[idx + GX] - 
                                   4.0 * T_read[idx];
                T_write[idx] = T_read[idx] + r * laplacian;
            }
        }
        
        for (int i = 0; i < GX; ++i) { 
            T_write[i] = AMBIENT_TEMP; 
            T_write[(GY-1) * GX + i] = AMBIENT_TEMP; 
        }
        for (int j = 0; j < GY; ++j) { 
            T_write[j * GX] = AMBIENT_TEMP; 
            T_write[j * GX + GX - 1] = AMBIENT_TEMP; 
        }
    }
    
    double *T_final = (total_steps % 2 == 0) ? T_buf0 : T_buf1;
    
    double local_max = AMBIENT_TEMP;
    double local_sum = 0.0;
    
    for (int j = 1; j < GY - 1; ++j) {
        for (int i = 1; i < GX - 1; ++i) {
            int idx = j * GX + i;
            double T = T_final[idx];
            if (T > local_max) local_max = T;
            local_sum += T;
        }
    }
    
    max_temp = local_max;
    avg_temp = local_sum / interior_count;
    
    double var_sum = 0.0;
    for (int j = 1; j < GY - 1; ++j) {
        for (int i = 1; i < GX - 1; ++i) {
            int idx = j * GX + i;
            double diff = T_final[idx] - avg_temp;
            var_sum += diff * diff;
        }
    }
    temp_variance = var_sum / interior_count;
    
    free(T_buf0);
    free(T_buf1);
}

// ============================================================================
// GPU DEVICE SOLVER v5.1 - USING GLOBAL MEMORY (FIXED)
// ============================================================================

__device__ void solve_heat_equation_2d_device_global_memory(
    const double *heat_sources, 
    const unsigned num_sources,
    const unsigned thread_id,
    double &max_temp, 
    double &avg_temp, 
    double &temp_variance
) {
    const int GX = GRID_SIZE_X;
    const int GY = GRID_SIZE_Y;
    const int grid_size = GX * GY;
    const int interior_count = (GX - 2) * (GY - 2);
    
    // FIX: Check if buffers are ready and thread_id is valid
    if (!dev_buffers_ready || thread_id >= MAX_POPULATION_SIZE) {
        // Fallback: return default values
        max_temp = AMBIENT_TEMP;
        avg_temp = AMBIENT_TEMP;
        temp_variance = 0.0;
        return;
    }
    
    // FIX: Check for NULL pointers
    if (dev_thread_T_buf0 == NULL || dev_thread_T_buf1 == NULL) {
        max_temp = AMBIENT_TEMP;
        avg_temp = AMBIENT_TEMP;
        temp_variance = 0.0;
        return;
    }
    
    const double dx = DOMAIN_LENGTH / (GX - 1);
    double r = THERMAL_DIFFUSIVITY * DT / (dx * dx);
    
    int sub_steps = 1;
    if (r > 0.24) { 
        sub_steps = (int)ceil(r / 0.24); 
        if (sub_steps > 100) sub_steps = 100; 
        r = THERMAL_DIFFUSIVITY * DT / (dx * dx) / sub_steps; 
    }
    
    // Get this thread's buffers from global memory
    double *T_buf0 = dev_thread_T_buf0 + thread_id * grid_size;
    double *T_buf1 = dev_thread_T_buf1 + thread_id * grid_size;
    
    // Initialize buffers
    for (int i = 0; i < grid_size; ++i) { 
        T_buf0[i] = AMBIENT_TEMP; 
        T_buf1[i] = AMBIENT_TEMP; 
    }
    
    // Parse heat sources
    int source_pos[MAX_SOURCES]; 
    double source_power[MAX_SOURCES];
    unsigned actual_sources = (num_sources < MAX_SOURCES) ? num_sources : MAX_SOURCES;
    
    for (unsigned s = 0; s < actual_sources; ++s) {
        double x_norm = fmin(1.0, fmax(0.0, heat_sources[s*3]));
        double y_norm = fmin(1.0, fmax(0.0, heat_sources[s*3 + 1]));
        int x = (int)(x_norm * (GX - 1));
        int y = (int)(y_norm * (GY - 1));
        x = max(1, min(GX - 2, x));
        y = max(1, min(GY - 2, y));
        source_pos[s] = y * GX + x;
        source_power[s] = fmin(MAX_SOURCE_INTENSITY, fmax(0.0, heat_sources[s*3 + 2]));
    }
    
    const unsigned total_steps = TIME_STEPS * sub_steps;
    
    // Time integration with buffer swapping
    for (unsigned t = 0; t < total_steps; ++t) {
        double *T_read  = (t % 2 == 0) ? T_buf0 : T_buf1;
        double *T_write = (t % 2 == 0) ? T_buf1 : T_buf0;
        
        // Apply heat sources
        for (unsigned s = 0; s < actual_sources; ++s) {
            T_read[source_pos[s]] = AMBIENT_TEMP + source_power[s];
        }
        
        // Heat equation update
        for (int j = 1; j < GY - 1; ++j) {
            for (int i = 1; i < GX - 1; ++i) {
                int idx = j * GX + i;
                double laplacian = T_read[idx - 1] + T_read[idx + 1] + 
                                   T_read[idx - GX] + T_read[idx + GX] - 
                                   4.0 * T_read[idx];
                T_write[idx] = T_read[idx] + r * laplacian;
            }
        }
        
        // Apply boundary conditions
        for (int i = 0; i < GX; ++i) { 
            T_write[i] = AMBIENT_TEMP; 
            T_write[(GY-1) * GX + i] = AMBIENT_TEMP; 
        }
        for (int j = 0; j < GY; ++j) { 
            T_write[j * GX] = AMBIENT_TEMP; 
            T_write[j * GX + GX - 1] = AMBIENT_TEMP; 
        }
    }
    
    // Determine which buffer has final result
    double *T_final = (total_steps % 2 == 0) ? T_buf0 : T_buf1;
    
    // Compute statistics
    max_temp = AMBIENT_TEMP;
    avg_temp = 0.0;
    
    for (int j = 1; j < GY - 1; ++j) {
        for (int i = 1; i < GX - 1; ++i) {
            int idx = j * GX + i;
            double T = T_final[idx];
            if (T > max_temp) max_temp = T;
            avg_temp += T;
        }
    }
    avg_temp /= interior_count;
    
    // Compute variance
    temp_variance = 0.0;
    for (int j = 1; j < GY - 1; ++j) {
        for (int i = 1; i < GX - 1; ++i) {
            int idx = j * GX + i;
            double diff = T_final[idx] - avg_temp;
            temp_variance += diff * diff;
        }
    }
    temp_variance /= interior_count;
}

// ============================================================================
// FITNESS FUNCTIONS (FIXED v5.1)
// ============================================================================

// Device fitness using GLOBAL MEMORY
__device__ void heat_conduction_fitness_device_impl(
    const double *decision_variables, 
    double *fitness_values, 
    const unsigned d_dim, 
    const unsigned f_dim
) {
    for (unsigned i = 0; i < f_dim; ++i) fitness_values[i] = 1e6;
    
    if (d_dim % 3 != 0 || d_dim == 0) return;
    unsigned num_sources = d_dim / 3;
    
    if (num_sources > MAX_SOURCES) { 
        fitness_values[0] = 10.0; 
        fitness_values[1] = 15.0; 
        if (f_dim > 2) fitness_values[2] = 100.0; 
        return; 
    }
    
    // FIX: Calculate thread_id properly
    unsigned thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // FIX: Bounds check
    if (thread_id >= MAX_POPULATION_SIZE) {
        // Use modulo to wrap around (safe fallback)
        thread_id = thread_id % MAX_POPULATION_SIZE;
    }
    
    double max_temp, avg_temp, temp_variance;
    
    solve_heat_equation_2d_device_global_memory(
        decision_variables, num_sources, thread_id,
        max_temp, avg_temp, temp_variance
    );
    
    fitness_values[0] = fabs(avg_temp - TARGET_TEMP);
    fitness_values[1] = sqrt(temp_variance);
    
    if (f_dim > 2) {
        double total_intensity = 0.0;
        for (unsigned s = 0; s < num_sources; ++s) {
            total_intensity += fmin(MAX_SOURCE_INTENSITY, fmax(0.0, decision_variables[s*3 + 2]));
        }
        fitness_values[2] = (total_intensity / (num_sources * MAX_SOURCE_INTENSITY)) * 100.0;
    }
}

// Host fitness (CPU)
__host__ void heat_conduction_fitness_host(
    const double *decision_variables, 
    double *fitness_values, 
    const unsigned d_dim, 
    const unsigned f_dim
) {
    for (unsigned i = 0; i < f_dim; ++i) fitness_values[i] = 1e6;
    
    if (d_dim % 3 != 0 || d_dim == 0) return;
    unsigned num_sources = d_dim / 3;
    
    if (num_sources > MAX_SOURCES) { 
        fitness_values[0] = 10.0; 
        fitness_values[1] = 15.0; 
        if (f_dim > 2) fitness_values[2] = 100.0; 
        return; 
    }
    
    double max_temp, avg_temp, temp_variance;
    solve_heat_equation_2d_cpu_serial(decision_variables, num_sources, max_temp, avg_temp, temp_variance);
    
    fitness_values[0] = fabs(avg_temp - TARGET_TEMP);
    fitness_values[1] = sqrt(temp_variance);
    
    if (f_dim > 2) {
        double total_intensity = 0.0;
        for (unsigned s = 0; s < num_sources; ++s) {
            total_intensity += fmax(0.0, fmin(MAX_SOURCE_INTENSITY, decision_variables[s*3 + 2]));
        }
        fitness_values[2] = (total_intensity / (num_sources * MAX_SOURCE_INTENSITY)) * 100.0;
    }
}

// FIX: Main fitness function with SAME signature (4 parameters)
__host__ __device__ void heat_conduction_fitness(
    const double *decision_variables, 
    double *fitness_values, 
    const unsigned d_dim, 
    const unsigned f_dim
) {
#ifdef __CUDA_ARCH__
    heat_conduction_fitness_device_impl(decision_variables, fitness_values, d_dim, f_dim);
#else
    heat_conduction_fitness_host(decision_variables, fitness_values, d_dim, f_dim);
#endif
}

// FIX: Function pointer now matches the signature
__device__ void (*heat_conduction_fitness_dptr)(const double*, double*, const unsigned, const unsigned) = heat_conduction_fitness;

// ============================================================================
// CALIBRATION
// ============================================================================

typedef struct { double f1, f2, f3; int dominated; } Sample;
static unsigned int calibration_seed = 42;

__host__ double calibration_rand() { 
    calibration_seed = calibration_seed * 1103515245 + 12345; 
    return (double)((calibration_seed / 65536) % 32768) / 32768.0; 
}

__host__ int dominates(Sample *a, Sample *b) {
    int dom_flag = 1, strictly_better = 0;
    if (a->f1 > b->f1) dom_flag = 0; 
    if (a->f2 > b->f2) dom_flag = 0; 
    if (a->f3 > b->f3) dom_flag = 0;
    if (a->f1 < b->f1) strictly_better = 1; 
    if (a->f2 < b->f2) strictly_better = 1; 
    if (a->f3 < b->f3) strictly_better = 1;
    return dom_flag && strictly_better;
}

__host__ void calibrate_with_random_sampling(
    unsigned num_sources, 
    unsigned f_dim, 
    double **pareto_front, 
    unsigned front_size, 
    double *ref_point
) {
    printf("  Calibrating with %d random samples...\n", CALIBRATION_SAMPLES);
    
    if (num_sources == 0) { 
        ref_point[0] = 10.0; ref_point[1] = 15.0; 
        if (f_dim > 2) ref_point[2] = 110.0; 
        for (unsigned i = 0; i < front_size; i++) { 
            pareto_front[i][0] = 5.0; pareto_front[i][1] = 10.0; 
            if (f_dim > 2) pareto_front[i][2] = 50.0; 
        } 
        return; 
    }
    
    Sample *samples = (Sample*)malloc(CALIBRATION_SAMPLES * sizeof(Sample));
    double *vars = (double*)malloc(num_sources * 3 * sizeof(double));
    
    if (!samples || !vars) { 
        ref_point[0] = 10.0; ref_point[1] = 15.0; 
        if (f_dim > 2) ref_point[2] = 110.0; 
        if (samples) free(samples); 
        if (vars) free(vars); 
        return; 
    }
    
    calibration_seed = 42;
    
    for (int i = 0; i < CALIBRATION_SAMPLES; i++) {
        for (unsigned s = 0; s < num_sources; s++) { 
            vars[s*3] = calibration_rand(); 
            vars[s*3 + 1] = calibration_rand(); 
            vars[s*3 + 2] = calibration_rand() * MAX_SOURCE_INTENSITY; 
        }
        
        double max_temp, avg_temp, variance;
        solve_heat_equation_2d_cpu_serial(vars, num_sources, max_temp, avg_temp, variance);
        
        samples[i].f1 = fabs(avg_temp - TARGET_TEMP); 
        samples[i].f2 = sqrt(variance);
        
        double total_intensity = 0; 
        for (unsigned s = 0; s < num_sources; s++) total_intensity += vars[s*3 + 2];
        samples[i].f3 = (total_intensity / (num_sources * MAX_SOURCE_INTENSITY)) * 100.0; 
        samples[i].dominated = 0;
    }
    
    int pareto_count = 0;
    for (int i = 0; i < CALIBRATION_SAMPLES; i++) { 
        for (int j = 0; j < CALIBRATION_SAMPLES; j++) { 
            if (i == j) continue; 
            if (dominates(&samples[j], &samples[i])) { 
                samples[i].dominated = 1; 
                break; 
            } 
        } 
        if (!samples[i].dominated) pareto_count++; 
    }
    
    printf("  Found %d non-dominated solutions\n", pareto_count);
    
    if (pareto_count == 0) { 
        ref_point[0] = 10.0; ref_point[1] = 15.0; 
        if (f_dim > 2) ref_point[2] = 110.0; 
        for (unsigned i = 0; i < front_size; i++) { 
            pareto_front[i][0] = 5.0; pareto_front[i][1] = 10.0; 
            if (f_dim > 2) pareto_front[i][2] = 50.0; 
        } 
        free(samples); free(vars); 
        return; 
    }
    
    double f1_max = -DBL_MAX, f2_max = -DBL_MAX;
    for (int i = 0; i < CALIBRATION_SAMPLES; i++) { 
        if (!samples[i].dominated) { 
            if (samples[i].f1 > f1_max) f1_max = samples[i].f1; 
            if (samples[i].f2 > f2_max) f2_max = samples[i].f2; 
        } 
    }
    
    int *pareto_indices = (int*)malloc(pareto_count * sizeof(int));
    if (!pareto_indices) { 
        ref_point[0] = f1_max * 1.2; ref_point[1] = f2_max * 1.2; 
        if (f_dim > 2) ref_point[2] = 110.0; 
        free(samples); free(vars); 
        return; 
    }
    
    int idx = 0; 
    for (int i = 0; i < CALIBRATION_SAMPLES && idx < pareto_count; i++) { 
        if (!samples[i].dominated) pareto_indices[idx++] = i; 
    }
    
    for (int i = 0; i < pareto_count - 1; i++) { 
        for (int j = i + 1; j < pareto_count; j++) { 
            if (samples[pareto_indices[j]].f3 < samples[pareto_indices[i]].f3) { 
                int tmp = pareto_indices[i]; 
                pareto_indices[i] = pareto_indices[j]; 
                pareto_indices[j] = tmp; 
            } 
        } 
    }
    
    for (unsigned i = 0; i < front_size; i++) { 
        int src_idx = 0; 
        if (pareto_count > 1 && front_size > 1) 
            src_idx = (int)((double)i / (front_size - 1) * (pareto_count - 1)); 
        if (src_idx >= pareto_count) src_idx = pareto_count - 1; 
        int sample_idx = pareto_indices[src_idx]; 
        pareto_front[i][0] = samples[sample_idx].f1; 
        pareto_front[i][1] = samples[sample_idx].f2; 
        if (f_dim > 2) pareto_front[i][2] = samples[sample_idx].f3; 
    }
    
    ref_point[0] = f1_max * 1.2; 
    ref_point[1] = f2_max * 1.2; 
    if (f_dim > 2) ref_point[2] = 110.0;
    if (ref_point[0] < 1.0) ref_point[0] = f1_max + 1.0; 
    if (ref_point[1] < 1.0) ref_point[1] = f2_max + 1.0;
    
    printf("  Reference point: [%.2f, %.2f, %.2f]\n", ref_point[0], ref_point[1], f_dim > 2 ? ref_point[2] : 0.0);
    
    free(samples); 
    free(vars); 
    free(pareto_indices);
}

// ============================================================================
// PARETO FRONT AND REFERENCE POINT
// ============================================================================

static double **g_pareto_front = NULL; 
static double *g_ref_point = NULL; 
static int g_calibrated = 0;

__host__ double** heat_conduction_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size) {
    double **f = (double**)malloc(size * sizeof(double*)); 
    if (!f) return NULL;
    
    double *_f = (double*)malloc(size * f_dim * sizeof(double)); 
    if (!_f) { free(f); return NULL; }
    
    for (unsigned i = 0u; i < size; ++i) f[i] = &_f[i * f_dim];
    
    if (!g_calibrated) {
        unsigned num_sources = d_dim / 3; 
        if (num_sources > 7) num_sources = 7;
        
        g_pareto_front = (double**)malloc(size * sizeof(double*)); 
        double *g_pf_data = (double*)malloc(size * f_dim * sizeof(double));
        
        if (!g_pareto_front || !g_pf_data) { 
            if (g_pareto_front) { free(g_pareto_front); g_pareto_front = NULL; } 
            if (g_pf_data) free(g_pf_data); 
            for (unsigned i = 0; i < size; i++) { 
                f[i][0] = 5.0; f[i][1] = 10.0; 
                if (f_dim > 2) f[i][2] = 50.0; 
            } 
            return f; 
        }
        
        for (unsigned i = 0; i < size; i++) g_pareto_front[i] = &g_pf_data[i * f_dim];
        
        g_ref_point = (double*)malloc(f_dim * sizeof(double));
        if (!g_ref_point) { 
            free(g_pareto_front); free(g_pf_data); 
            g_pareto_front = NULL; 
            for (unsigned i = 0; i < size; i++) { 
                f[i][0] = 5.0; f[i][1] = 10.0; 
                if (f_dim > 2) f[i][2] = 50.0; 
            } 
            return f; 
        }
        
        calibrate_with_random_sampling(num_sources, f_dim, g_pareto_front, size, g_ref_point);
        g_calibrated = 1;
    }
    
    for (unsigned i = 0; i < size; i++) {
        for (unsigned j = 0; j < f_dim; j++) {
            f[i][j] = g_pareto_front[i][j];
        }
    }
    
    return f;
}

__host__ double* heat_conduction_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds) {
    double *ref = (double*)malloc(f_dim * sizeof(double));
    if (!ref) return NULL;
    
    if (!g_calibrated) {
        heat_conduction_pareto_front(d_dim, f_dim, PARETO_FRONT_SIZE);
    }
    
    if (g_calibrated && g_ref_point) {
        for (unsigned i = 0; i < f_dim; i++) ref[i] = g_ref_point[i];
    } else {
        ref[0] = 10.0; ref[1] = 15.0;
        if (f_dim > 2) ref[2] = 110.0;
    }
    
    return ref;
}

// ============================================================================
// PROBLEM INITIALIZATION
// ============================================================================

__host__ Problem* generate_heat_conduction_problem(unsigned problem_id, unsigned d_dim, unsigned f_dim) {
    if (problem_id != 1u || d_dim == 0u || d_dim % 3u != 0u || f_dim != 3u) {
        fprintf(stderr,
                "Heat conduction requires problem_id=1, d_dim divisible by 3, and f_dim=3.\n");
        return NULL;
    }
    
    Problem *problem = (Problem*)malloc(sizeof(Problem));
    if (!problem) return NULL;
    
    problem->suite = HEAT_CONDUCTION;
    problem->problem_id = problem_id; 
    problem->d_dim = d_dim; 
    problem->f_dim = f_dim; 
    problem->i_dim = 0u;
    
    const unsigned adjusted_d_dim = d_dim;
    const unsigned num_sources = adjusted_d_dim / 3u;
    
    double dx = DOMAIN_LENGTH / (GRID_SIZE_X - 1); 
    double r = THERMAL_DIFFUSIVITY * DT / (dx * dx);
    
    // Initialize GPU memory
    init_gpu_memory();
    init_gpu_thread_buffers();
    
    // Calculate memory usage
    size_t grid_size = GRID_SIZE_X * GRID_SIZE_Y;
    size_t global_mem_per_thread = grid_size * 2 * sizeof(double);
    size_t total_global_mem = MAX_POPULATION_SIZE * global_mem_per_thread;
    
    printf("\n==================================================================\n");
    printf("     Heat Conduction Problem (v5.1 - Global Memory FIXED)        \n");
    printf("==================================================================\n");
    printf("  Sources: %-3u              Objectives: %-3u\n", num_sources, f_dim);
    printf("  Grid: %3dx%-3d             Time steps: %-5d\n", GRID_SIZE_X, GRID_SIZE_Y, TIME_STEPS);
    printf("  Target: %.1fC             Ambient: %.1fC\n", TARGET_TEMP, AMBIENT_TEMP);
    printf("  Fourier number: %.4f    (stable if <= 0.25)\n", r);
    printf("  Max Population: %d\n", MAX_POPULATION_SIZE);
    printf("  GPU Memory: %s\n", gpu_memory_initialized ? "OK" : "Failed");
    printf("  GPU Thread Buffers: %s (%.2f MB)\n", 
           gpu_thread_buffers_initialized ? "OK" : "Failed",
           total_global_mem / (1024.0 * 1024.0));
    #ifdef _OPENMP
    printf("  OpenMP: Available (%d threads)\n", omp_get_max_threads());
    #else
    printf("  OpenMP: Not available\n");
    #endif
    printf("==================================================================\n");
    
    problem->upper_bounds = (double*)malloc(sizeof(double) * d_dim); 
    problem->lower_bounds = (double*)malloc(sizeof(double) * d_dim);
    
    if (!problem->upper_bounds || !problem->lower_bounds) { 
        if (problem->upper_bounds) free(problem->upper_bounds); 
        if (problem->lower_bounds) free(problem->lower_bounds); 
        free(problem); 
        return NULL; 
    }
    
    for (unsigned dim = 0u; dim < d_dim; ++dim) { 
        if (dim % 3 == 2) { 
            problem->lower_bounds[dim] = 0.0; 
            problem->upper_bounds[dim] = MAX_SOURCE_INTENSITY; 
        } else { 
            problem->lower_bounds[dim] = 0.0; 
            problem->upper_bounds[dim] = 1.0; 
        } 
    }
    
    problem->front_size = PARETO_FRONT_SIZE; 
    problem->hypervolume_N = (unsigned)pow(100u, f_dim); 
    problem->fitness = heat_conduction_fitness;
    
    // Copy GPU fitness function pointer
    cudaError_t err = cudaMemcpyFromSymbol(&problem->gpu_fitness, heat_conduction_fitness_dptr, sizeof(void(*)()));
    if (err != cudaSuccess) problem->gpu_fitness = NULL;
    
    // Reset calibration and generate front/reference point
    g_calibrated = 0;
    problem->front = heat_conduction_pareto_front(adjusted_d_dim, f_dim, problem->front_size);
    problem->ref_point = heat_conduction_reference_point(adjusted_d_dim, f_dim, problem->upper_bounds, problem->lower_bounds);
    
    printf("\n");
    return problem;
}

// ============================================================================
// CLEANUP
// ============================================================================

__host__ void heat_conduction_cleanup() {
    free_gpu_memory();
    free_gpu_thread_buffers();
    
    if (g_pareto_front) {
        if (g_pareto_front[0]) free(g_pareto_front[0]);
        free(g_pareto_front);
        g_pareto_front = NULL;
    }
    if (g_ref_point) {
        free(g_ref_point);
        g_ref_point = NULL;
    }
    g_calibrated = 0;
}

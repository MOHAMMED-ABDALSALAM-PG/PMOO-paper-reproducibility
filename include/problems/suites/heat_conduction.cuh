#ifndef PMOO_PROBLEMS_SUITES_HEAT_CONDUCTION_CUH
#define PMOO_PROBLEMS_SUITES_HEAT_CONDUCTION_CUH

#include "../problem.cuh"
#include <cuda_runtime.h>

// ============================================================================
// Heat Conduction Parameters - VERSION 5.1 (CUDA global-memory kernels)
// 
// This version uses proper CUDA kernel parallelization for GPU acceleration.
// The grid size can now be larger because we use global memory instead of
// stack arrays.
//
// Stability condition: r = α * Δt / Δx² ≤ 0.25 (for 2D explicit FTCS)
// Reference: Smith, G.D. (1985) Numerical Solution of PDEs
// ============================================================================

#define GRID_SIZE_X 75              // Spatial grid width
#define GRID_SIZE_Y 75              // Spatial grid height; 73x73 = 5,329 interior points
#define TIME_STEPS 4000             // Temporal iterations
#define DOMAIN_LENGTH 1.0           // Physical domain size (meters)
#define AMBIENT_TEMP 20.0           // Boundary temperature (Celsius)
#define TARGET_TEMP 25.0            // Target average temperature
#define MAX_SOURCE_INTENSITY 100.0  // Max heat source power
#define THERMAL_DIFFUSIVITY 0.01    // α = k/(ρ*cp) in m²/s
#define DT 0.0005                   // Time step (seconds)
#define CALIBRATION_SAMPLES 2000    // Samples for Pareto front estimation
#define PARETO_FRONT_SIZE 100       // Size of reference front

// Stability check:
// dx = 1.0 / 74 = 0.01351 (approximately)
// r = 0.01 * 0.0005 / (1/74)^2 = 0.02738 (approximately), hence stable.

// ============================================================================
// Function Declarations
// ============================================================================

__host__ Problem* generate_heat_conduction_problem(unsigned problem_id,
                                                    unsigned d_dim,
                                                    unsigned f_dim);

__host__ __device__ void heat_conduction_fitness(const double *decision_variables,
                                                  double *fitness_values,
                                                  const unsigned d_dim,
                                                  const unsigned f_dim);

__host__ double** heat_conduction_pareto_front(const unsigned d_dim,
                                                const unsigned f_dim,
                                                const unsigned size);

__host__ double* heat_conduction_reference_point(const unsigned d_dim,
                                                  const unsigned f_dim,
                                                  const double *upper_bounds,
                                                  const double *lower_bounds);

__host__ void heat_conduction_cleanup();

#endif // PMOO_PROBLEMS_SUITES_HEAT_CONDUCTION_CUH

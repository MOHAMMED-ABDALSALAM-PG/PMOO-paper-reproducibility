#ifndef PMOO_PROBLEMS_SUITES_CUSTOM_CUH
#define PMOO_PROBLEMS_SUITES_CUSTOM_CUH

#include "../problem.cuh"

#include <cuda_runtime.h>

__host__ Problem* generate_custom_problem(unsigned problem_id, unsigned d_dim, unsigned f_dim);

__host__ __device__ void custom1_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim);

__host__ double** custom1_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size);

__host__ double* custom1_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds);

#endif
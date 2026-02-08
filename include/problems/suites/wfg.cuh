#ifndef PMOO_PROBLEMS_SUITES_WFG_CUH
#define PMOO_PROBLEMS_SUITES_WFG_CUH

#include "../problem.cuh"

#include <cuda_runtime.h>

__host__ Problem* generate_wfg_problem(unsigned problem_id, unsigned d_dim, unsigned f_dim);

__host__ __device__ void wfg1_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim);

__host__ __device__ void wfg2_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim);

__host__ __device__ void wfg3_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim);

__host__ __device__ void wfg4_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim);

__host__ __device__ void wfg5_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim);

__host__ __device__ void wfg6_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim);

__host__ __device__ void wfg7_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim);

__host__ __device__ void wfg8_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim);

__host__ __device__ void wfg9_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim);

__host__ double** wfg1_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size);

__host__ double** wfg2_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size);

__host__ double** wfg3_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size);

__host__ double** wfg4_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size);

__host__ double** wfg5_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size);

__host__ double** wfg6_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size);

__host__ double** wfg7_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size);

__host__ double** wfg8_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size);

__host__ double** wfg9_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size);

__host__ double* wfg1_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds);

__host__ double* wfg2_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds);

__host__ double* wfg3_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds);

__host__ double* wfg4_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds);

__host__ double* wfg5_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds);

__host__ double* wfg6_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds);

__host__ double* wfg7_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds);

__host__ double* wfg8_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds);

__host__ double* wfg9_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds);

#endif
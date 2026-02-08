#ifndef PMOO_PROBLEMS_SUITES_DTLZ_CUH
#define PMOO_PROBLEMS_SUITES_DTLZ_CUH

#include "../problem.cuh"

#include <cuda_runtime.h>

__host__ Problem* generate_dtlz_problem(unsigned problem_id, unsigned d_dim, unsigned f_dim);

__host__ __device__ void dtlz1_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim);

__host__ __device__ void dtlz2_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim);

__host__ __device__ void dtlz3_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim);

__host__ __device__ void dtlz4_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim);

__host__ __device__ void dtlz5_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim);

__host__ __device__ void dtlz6_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim);

__host__ __device__ void dtlz7_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim);

__host__ double** dtlz1_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size);

__host__ double** dtlz2_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size);

__host__ double** dtlz3_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size);

__host__ double** dtlz4_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size);

__host__ double** dtlz5_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size);

__host__ double** dtlz6_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size);

__host__ double** dtlz7_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size);

__host__ double* dtlz1_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds);

__host__ double* dtlz2_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds);

__host__ double* dtlz3_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds);

__host__ double* dtlz4_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds);

__host__ double* dtlz5_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds);

__host__ double* dtlz6_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds);

__host__ double* dtlz7_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds);

#endif
#ifndef PMOO_PROBLEMS_SUITES_ZDT_CUH
#define PMOO_PROBLEMS_SUITES_ZDT_CUH

#include "../problem.cuh"

#include <cuda_runtime.h>

__host__ Problem* generate_zdt_problem(unsigned problem_id, unsigned d_dim, unsigned f_dim);

__host__ __device__ void zdt1_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim);

__host__ __device__ void zdt2_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim);

__host__ __device__ void zdt3_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim);

__host__ __device__ void zdt4_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim);

__host__ __device__ void zdt5_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim);

__host__ __device__ void zdt6_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim);

__host__ double** zdt1_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size);

__host__ double** zdt2_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size);

__host__ double** zdt3_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size);

__host__ double** zdt4_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size);

__host__ double** zdt5_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size);

__host__ double** zdt6_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size);

__host__ double* zdt1_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds);

__host__ double* zdt2_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds);

__host__ double* zdt3_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds);

__host__ double* zdt4_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds);

__host__ double* zdt5_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds);

__host__ double* zdt6_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds);

#endif
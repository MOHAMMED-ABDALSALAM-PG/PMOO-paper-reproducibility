#include "problems/suites/wfg.cuh"

#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>

// we need to define device pointers this way, so we can get function address on gpu on the host side

__device__ void (*wfg1_fitness_dptr)(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) = wfg1_fitness;

__device__ void (*wfg2_fitness_dptr)(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) = wfg2_fitness;

__device__ void (*wfg3_fitness_dptr)(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) = wfg3_fitness;

__device__ void (*wfg4_fitness_dptr)(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) = wfg4_fitness;

__device__ void (*wfg5_fitness_dptr)(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) = wfg5_fitness;

__device__ void (*wfg6_fitness_dptr)(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) = wfg6_fitness;

__device__ void (*wfg7_fitness_dptr)(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) = wfg7_fitness;

__device__ void (*wfg8_fitness_dptr)(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) = wfg8_fitness;

__device__ void (*wfg9_fitness_dptr)(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) = wfg9_fitness;

__host__ Problem* generate_wfg_problem(unsigned problem_id, unsigned d_dim, unsigned f_dim) {
    if (problem_id == 0u || problem_id > TEST_SUITES_PROBLEM_COUNT[WFG]) return NULL;

    Problem* problem = (Problem*)malloc(sizeof(Problem));
    problem->suite = WFG;
    problem->problem_id = problem_id;
    problem->d_dim = d_dim;
    problem->f_dim = f_dim;
    problem->i_dim = 0u;
    problem->upper_bounds = (double*)malloc(sizeof(double) * d_dim);
    problem->lower_bounds = (double*)malloc(sizeof(double) * d_dim);
    for (unsigned dim = 0u; dim < d_dim; ++dim) {
        problem->upper_bounds[dim] = 2.0 * (dim + 1.0);
        problem->lower_bounds[dim] = 0.0;
    }
    problem->front_size = 0u; // TODO
    problem->hypervolume_N = 0u; // TODO
    problem->fitness = NULL;
    problem->gpu_fitness = NULL;
    problem->front = NULL;
    problem->ref_point = NULL;

    switch (problem_id) {
        case 1u:
            problem->fitness = wfg1_fitness;
            cudaMemcpyFromSymbol(&problem->gpu_fitness, wfg1_fitness_dptr, sizeof(Fitness));
            problem->front = wfg1_pareto_front(problem->d_dim, problem->f_dim, problem->front_size);
            problem->ref_point = wfg1_reference_point(problem->d_dim, problem->f_dim, problem->upper_bounds, problem->lower_bounds);
            break;
        case 2u:
            problem->fitness = wfg2_fitness;
            cudaMemcpyFromSymbol(&problem->gpu_fitness, wfg2_fitness_dptr, sizeof(Fitness));
            problem->front = wfg2_pareto_front(problem->d_dim, problem->f_dim, problem->front_size);
            problem->ref_point = wfg2_reference_point(problem->d_dim, problem->f_dim, problem->upper_bounds, problem->lower_bounds);
            break;
        case 3u:
            problem->fitness = wfg3_fitness;
            cudaMemcpyFromSymbol(&problem->gpu_fitness, wfg3_fitness_dptr, sizeof(Fitness));
            problem->front = wfg3_pareto_front(problem->d_dim, problem->f_dim, problem->front_size);
            problem->ref_point = wfg3_reference_point(problem->d_dim, problem->f_dim, problem->upper_bounds, problem->lower_bounds);
            break;
        case 4u:
            problem->fitness = wfg4_fitness;
            cudaMemcpyFromSymbol(&problem->gpu_fitness, wfg4_fitness_dptr, sizeof(Fitness));
            problem->front = wfg4_pareto_front(problem->d_dim, problem->f_dim, problem->front_size);
            problem->ref_point = wfg4_reference_point(problem->d_dim, problem->f_dim, problem->upper_bounds, problem->lower_bounds);
            break;
        case 5u:
            problem->fitness = wfg5_fitness;
            cudaMemcpyFromSymbol(&problem->gpu_fitness, wfg5_fitness_dptr, sizeof(Fitness));
            problem->front = wfg5_pareto_front(problem->d_dim, problem->f_dim, problem->front_size);
            problem->ref_point = wfg5_reference_point(problem->d_dim, problem->f_dim, problem->upper_bounds, problem->lower_bounds);
            break;
        case 6u:
            problem->fitness = wfg6_fitness;
            cudaMemcpyFromSymbol(&problem->gpu_fitness, wfg6_fitness_dptr, sizeof(Fitness));
            problem->front = wfg6_pareto_front(problem->d_dim, problem->f_dim, problem->front_size);
            problem->ref_point = wfg6_reference_point(problem->d_dim, problem->f_dim, problem->upper_bounds, problem->lower_bounds);
            break;
        case 7u:
            problem->fitness = wfg7_fitness;
            cudaMemcpyFromSymbol(&problem->gpu_fitness, wfg7_fitness_dptr, sizeof(Fitness));
            problem->front = wfg7_pareto_front(problem->d_dim, problem->f_dim, problem->front_size);
            problem->ref_point = wfg7_reference_point(problem->d_dim, problem->f_dim, problem->upper_bounds, problem->lower_bounds);
            break;
        case 8u:
            problem->fitness = wfg8_fitness;
            cudaMemcpyFromSymbol(&problem->gpu_fitness, wfg8_fitness_dptr, sizeof(Fitness));
            problem->front = wfg8_pareto_front(problem->d_dim, problem->f_dim, problem->front_size);
            problem->ref_point = wfg8_reference_point(problem->d_dim, problem->f_dim, problem->upper_bounds, problem->lower_bounds);
            break;
        case 9u:
            problem->fitness = wfg9_fitness;
            cudaMemcpyFromSymbol(&problem->gpu_fitness, wfg9_fitness_dptr, sizeof(Fitness));
            problem->front = wfg9_pareto_front(problem->d_dim, problem->f_dim, problem->front_size);
            problem->ref_point = wfg9_reference_point(problem->d_dim, problem->f_dim, problem->upper_bounds, problem->lower_bounds);
            break;
    }

    return problem;
};

__host__ __device__ void wfg1_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) {
    return; // TODO
};

__host__ __device__ void wfg2_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) {
    return; // TODO
};

__host__ __device__ void wfg3_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) {
    return; // TODO
};

__host__ __device__ void wfg4_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) {
    return; // TODO
};

__host__ __device__ void wfg5_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) {
    return; // TODO
};

__host__ __device__ void wfg6_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) {
    return; // TODO
};

__host__ __device__ void wfg7_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) {
    return; // TODO
};

__host__ __device__ void wfg8_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) {
    return; // TODO
};

__host__ __device__ void wfg9_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) {
    return; // TODO
};

__host__ double** wfg1_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size) {
    return NULL; // TODO
};

__host__ double** wfg2_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size) {
    return NULL; // TODO
};

__host__ double** wfg3_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size) {
    return NULL; // TODO
};

__host__ double** wfg4_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size) {
    return NULL; // TODO
};

__host__ double** wfg5_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size) {
    return NULL; // TODO
};

__host__ double** wfg6_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size) {
    return NULL; // TODO
};

__host__ double** wfg7_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size) {
    return NULL; // TODO
};

__host__ double** wfg8_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size) {
    return NULL; // TODO
};

__host__ double** wfg9_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size) {
    return NULL; // TODO
};

__host__ double* wfg1_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds) {
    return NULL; // TODO
};

__host__ double* wfg2_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds) {
    return NULL; // TODO
};

__host__ double* wfg3_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds) {
    return NULL; // TODO
};

__host__ double* wfg4_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds) {
    return NULL; // TODO
};

__host__ double* wfg5_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds) {
    return NULL; // TODO
};

__host__ double* wfg6_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds) {
    return NULL; // TODO
};

__host__ double* wfg7_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds) {
    return NULL; // TODO
};

__host__ double* wfg8_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds) {
    return NULL; // TODO
};

__host__ double* wfg9_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds) {
    return NULL; // TODO
};

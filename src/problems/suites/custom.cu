#include "problems/suites/custom.cuh"

#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>

// we need to define device pointers this way, so we can get function address on gpu on the host side

__device__ void (*custom1_fitness_dptr)(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) = custom1_fitness;

__host__ Problem* generate_custom_problem(unsigned problem_id, unsigned d_dim, unsigned f_dim) {
    if (problem_id == 0u || problem_id > TEST_SUITES_PROBLEM_COUNT[CUSTOM]) return NULL;

    Problem* problem = (Problem*)malloc(sizeof(Problem));
    problem->suite = CUSTOM;
    problem->problem_id = problem_id;
    problem->d_dim = d_dim;
    problem->f_dim = f_dim;
    problem->i_dim = 0u; ///////////////// INTEGER DECISION VARIABLES NOT SUPPORTED
    problem->upper_bounds = (double*)malloc(sizeof(double) * d_dim);
    problem->lower_bounds = (double*)malloc(sizeof(double) * d_dim);
    ///////////////// LOWER AND UPPER BOUNDS FOR DECISION PROBLEMS
    for (unsigned dim = 0u; dim < d_dim; ++dim) {
        problem->upper_bounds[dim] = 1.0;
        problem->lower_bounds[dim] = 0.0;
    }
    /////////////////
    problem->front_size = 100u;
    problem->hypervolume_N = (unsigned)pow(100u, f_dim);
    problem->fitness = NULL;
    problem->gpu_fitness = NULL;
    problem->front = NULL;
    problem->ref_point = NULL;

    switch (problem_id) {
        case 1u:
            ///////////////// SET LOWER AND UPPER BOUNDS FOR DECISION PROBLEMS PER PROBLEM
            problem->fitness = custom1_fitness;
            cudaMemcpyFromSymbol(&problem->gpu_fitness, custom1_fitness_dptr, sizeof(Fitness));
            problem->front = custom1_pareto_front(problem->d_dim, problem->f_dim, problem->front_size);
            problem->ref_point = custom1_reference_point(problem->d_dim, problem->f_dim, problem->upper_bounds, problem->lower_bounds);
            break;
    }

    return problem;
};

__host__ __device__ void custom1_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) {
    ///////////////// FITNESS FUNCTION IMPLEMENTATION
    double g = 0.0;
    for (unsigned i = 1u; i < d_dim; ++i)
        g += decision_variables[i];
    g = 1.0 + (9.0 * g) / (d_dim - 1.0);

    fitness_values[0] = decision_variables[0];
    fitness_values[1] = g * (1.0 - sqrt(decision_variables[0] / g));
    /////////////////
};

__host__ double** custom1_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size) {
    double *x = (double*)malloc(d_dim * sizeof(double));
    for (unsigned dim = 0u; dim < d_dim; ++dim)
        x[dim] = 0.0;

    double **f = (double**)malloc(size * sizeof(double*));
    double *_f = (double*)malloc(size * f_dim * sizeof(double));
    for (unsigned i = 0u; i < size; ++i)
        f[i] = &_f[i * f_dim];
    
    double step = 1.0 / (size - 1.0);
    for (unsigned i = 0u; i < size; ++i) {
        x[0] = i * step;
        ///////////////// CALCULATING PARETO FRONT FOR GD AND IGD
        custom1_fitness(x, f[i], d_dim, f_dim);
        /////////////////
    }

    free(x);
    return f;
};

__host__ double* custom1_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds) {
    double *ref = (double*)malloc(f_dim * sizeof(double));
    ///////////////// REFERENCE POINT FOR HYPERVOLUME
    for (unsigned i = 0u; i < f_dim; ++i) ref[i] = 1.0 * REF_POINT_MULTIPLIER;
    /////////////////
    return ref;
};

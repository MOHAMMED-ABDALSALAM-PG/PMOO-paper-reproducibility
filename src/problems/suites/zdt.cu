#include "problems/suites/zdt.cuh"

#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>

// we need to define device pointers this way, so we can get function address on gpu on the host side

__device__ void (*zdt1_fitness_dptr)(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) = zdt1_fitness;

__device__ void (*zdt2_fitness_dptr)(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) = zdt2_fitness;

__device__ void (*zdt3_fitness_dptr)(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) = zdt3_fitness;

__device__ void (*zdt4_fitness_dptr)(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) = zdt4_fitness;

__device__ void (*zdt5_fitness_dptr)(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) = zdt5_fitness;

__device__ void (*zdt6_fitness_dptr)(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) = zdt6_fitness;

__host__ Problem* generate_zdt_problem(unsigned problem_id, unsigned d_dim, unsigned f_dim) {
    if (problem_id == 0u || problem_id > TEST_SUITES_PROBLEM_COUNT[ZDT]) return NULL;

    Problem* problem = (Problem*)malloc(sizeof(Problem));
    problem->suite = ZDT;
    problem->problem_id = problem_id;
    problem->d_dim = d_dim;
    problem->f_dim = f_dim;
    problem->i_dim = 0u;
    problem->upper_bounds = (double*)malloc(sizeof(double) * d_dim);
    problem->lower_bounds = (double*)malloc(sizeof(double) * d_dim);
    for (unsigned dim = 0u; dim < d_dim; ++dim) {
        problem->upper_bounds[dim] = 1.0;
        problem->lower_bounds[dim] = 0.0;
    }
    problem->front_size = 100u; // NOTE: keep it a multiply of 5 due to ZDT3
    problem->hypervolume_N = (unsigned)pow(100u, f_dim);
    problem->fitness = NULL;
    problem->gpu_fitness = NULL;
    problem->front = NULL;
    problem->ref_point = NULL;

    // we are returning object containing metadata, but without fitness and pareto front
    if (f_dim != 2u) return problem; // NOTE: ZDT test suite only utilizes F_DIM = 2
    if (problem_id == 5u) return problem; // TODO implement ZDT5

    switch (problem_id) {
        case 1u:
            problem->fitness = zdt1_fitness;
            cudaMemcpyFromSymbol(&problem->gpu_fitness, zdt1_fitness_dptr, sizeof(Fitness));
            problem->front = zdt1_pareto_front(problem->d_dim, problem->f_dim, problem->front_size);
            problem->ref_point = zdt1_reference_point(problem->d_dim, problem->f_dim, problem->upper_bounds, problem->lower_bounds);
            break;
        case 2u:
            problem->fitness = zdt2_fitness;
            cudaMemcpyFromSymbol(&problem->gpu_fitness, zdt2_fitness_dptr, sizeof(Fitness));
            problem->front = zdt2_pareto_front(problem->d_dim, problem->f_dim, problem->front_size);
            problem->ref_point = zdt2_reference_point(problem->d_dim, problem->f_dim, problem->upper_bounds, problem->lower_bounds);
            break;
        case 3u:
            problem->fitness = zdt3_fitness;
            cudaMemcpyFromSymbol(&problem->gpu_fitness, zdt3_fitness_dptr, sizeof(Fitness));
            problem->front = zdt3_pareto_front(problem->d_dim, problem->f_dim, problem->front_size);
            problem->ref_point = zdt3_reference_point(problem->d_dim, problem->f_dim, problem->upper_bounds, problem->lower_bounds);
            break;
        case 4u:
            for (unsigned dim = 1u; dim < d_dim; ++dim) { // every dimension except the first
                problem->lower_bounds[dim] = -5.0;
                problem->upper_bounds[dim] = 5.0;
            }
            problem->fitness = zdt4_fitness;
            cudaMemcpyFromSymbol(&problem->gpu_fitness, zdt4_fitness_dptr, sizeof(Fitness));
            problem->front = zdt4_pareto_front(problem->d_dim, problem->f_dim, problem->front_size);
            problem->ref_point = zdt4_reference_point(problem->d_dim, problem->f_dim, problem->upper_bounds, problem->lower_bounds);
            break;
        case 5u:
            problem->i_dim = problem->d_dim; // ZDT5 is an integer problem
            problem->fitness = NULL; // TODO implement ZDT5
            cudaMemcpyFromSymbol(&problem->gpu_fitness, zdt5_fitness_dptr, sizeof(Fitness));
            problem->front = zdt5_pareto_front(problem->d_dim, problem->f_dim, problem->front_size);
            problem->ref_point = zdt5_reference_point(problem->d_dim, problem->f_dim, problem->upper_bounds, problem->lower_bounds);
            break;
        case 6u:
            problem->fitness = zdt6_fitness;
            cudaMemcpyFromSymbol(&problem->gpu_fitness, zdt6_fitness_dptr, sizeof(Fitness));
            problem->front = zdt6_pareto_front(problem->d_dim, problem->f_dim, problem->front_size);
            problem->ref_point = zdt6_reference_point(problem->d_dim, problem->f_dim, problem->upper_bounds, problem->lower_bounds);
            break;
    }

    return problem;
};

__host__ __device__ void zdt1_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) {
    double g = 0.0;
    for (unsigned i = 1u; i < d_dim; ++i)
        g += decision_variables[i];
    g = 1.0 + (9.0 * g) / (d_dim - 1.0);

    fitness_values[0] = decision_variables[0];
    fitness_values[1] = g * (1.0 - sqrt(decision_variables[0] / g));
};

__host__ __device__ void zdt2_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) {
    double g = 0.0;
    for (unsigned i = 1u; i < d_dim; ++i)
        g += decision_variables[i];
    g = 1.0 + (9.0 * g) / (d_dim - 1.0);

    fitness_values[0] = decision_variables[0];
    fitness_values[1] = g * (1.0 - pow(decision_variables[0] / g, 2.0));
};

__host__ __device__ void zdt3_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) {
    double g = 0.0;
    for (unsigned i = 1u; i < d_dim; ++i)
        g += decision_variables[i];
    g = 1.0 + (9.0 * g) / (d_dim - 1.0);

    fitness_values[0] = decision_variables[0];
    fitness_values[1] = g * (1.0 - sqrt(decision_variables[0] / g) - (decision_variables[0] / g * sin(10.0 * M_PI * decision_variables[0])));
};

__host__ __device__ void zdt4_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) {
    double g = 0.0;
    g = 1.0 + 10.0 * (d_dim - 1.0);
    for (unsigned i = 1u; i < d_dim; ++i)
        g += pow(decision_variables[i], 2.0) - (10.0 * cos(4.0 * M_PI * decision_variables[i]));

    fitness_values[0] = decision_variables[0];
    fitness_values[1] = g * (1.0 - sqrt(decision_variables[0] / g));
};

__host__ __device__ void zdt5_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) {
    return; // TODO
};

__host__ __device__ void zdt6_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) {
    double g = 0.0;
    for (unsigned i = 1u; i < d_dim; ++i)
        g += decision_variables[i];
    g = 1.0 + (9.0 * pow(g / (d_dim - 1u), 0.25));

    fitness_values[0] = 1.0 - exp(-4.0 * decision_variables[0]) * pow(sin(6 * M_PI * decision_variables[0]), 6.0);
    fitness_values[1] = g * (1.0 - (fitness_values[0] / g) * (fitness_values[0] / g));
};

__host__ double** zdt1_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size) {
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
        zdt1_fitness(x, f[i], d_dim, f_dim);
    }

    free(x);
    return f;
};

__host__ double** zdt2_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size) {
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
        zdt2_fitness(x, f[i], d_dim, f_dim);
    }

    free(x);
    return f;
};

__host__ double** zdt3_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size) {
    const unsigned RANGES_COUNT = 5u;
    const double RANGES[RANGES_COUNT][2u] = {
        { 0.0, 0.0830 },
        { 0.1822, 0.2577 },
        { 0.4093, 0.4538 },
        { 0.6183, 0.6525 },
        { 0.8233, 0.8518 }
    };

    double *x = (double*)malloc(d_dim * sizeof(double));
    for (unsigned dim = 0u; dim < d_dim; ++dim)
        x[dim] = 0.0;

    double **f = (double**)malloc(size * sizeof(double*));
    double *_f = (double*)malloc(size * f_dim * sizeof(double));
    for (unsigned i = 0u; i < size; ++i)
        f[i] = &_f[i * f_dim];

    unsigned range_size = size / RANGES_COUNT;
    for (unsigned r = 0u; r < RANGES_COUNT; ++r) {
        double step = (RANGES[r][1] - RANGES[r][0]) / (range_size - 1.0);
        for (unsigned i = 0u; i < range_size; ++i) {
            x[0] = RANGES[r][0] + i * step;
            zdt3_fitness(x, f[r * range_size + i], d_dim, f_dim);
        }
    }

    free(x);
    return f;
};

__host__ double** zdt4_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size) {
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
        zdt4_fitness(x, f[i], d_dim, f_dim);
    }

    free(x);
    return f;
};

__host__ double** zdt5_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size) {
    return NULL; // TODO
};

/*
 * NOTE:
 * ZDT6 is designed to test algorithm ability to maintain diversity.
 * Evenly spaced points in decision space will not result in evenly spaced pareto front.
 * That is why we are dividing by 4 during point generation.
 */
__host__ double** zdt6_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size) {
    double *x = (double*)malloc(d_dim * sizeof(double));
    for (unsigned dim = 0u; dim < d_dim; ++dim)
        x[dim] = 0.0;

    double **f = (double**)malloc(size * sizeof(double*));
    double *_f = (double*)malloc(size * f_dim * sizeof(double));
    for (unsigned i = 0u; i < size; ++i)
        f[i] = &_f[i * f_dim];
    
    double step = 1.0 / (size - 1.0);
    for (unsigned i = 0u; i < size; ++i) {
        x[0] = (i * step) / 4.0;
        zdt6_fitness(x, f[i], d_dim, f_dim);
    }

    free(x);
    return f;
};

__host__ double* zdt1_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds) {
    double *ref = (double*)malloc(f_dim * sizeof(double));
    for (unsigned i = 0u; i < f_dim; ++i) ref[i] = 1.0 * REF_POINT_MULTIPLIER;
    return ref;
};

__host__ double* zdt2_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds) {
    double *ref = (double*)malloc(f_dim * sizeof(double));
    for (unsigned i = 0u; i < f_dim; ++i) ref[i] = 1.0 * REF_POINT_MULTIPLIER;
    return ref;
};

__host__ double* zdt3_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds) {
    double *ref = (double*)malloc(f_dim * sizeof(double));
    for (unsigned i = 0u; i < f_dim; ++i) ref[i] = 1.0 * REF_POINT_MULTIPLIER;
    return ref;
};

__host__ double* zdt4_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds) {
    double *ref = (double*)malloc(f_dim * sizeof(double));
    for (unsigned i = 0u; i < f_dim; ++i) ref[i] = 1.0 * REF_POINT_MULTIPLIER;
    return ref;
};

__host__ double* zdt5_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds) {
    return NULL; // TODO
};

__host__ double* zdt6_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds) {
    double *ref = (double*)malloc(f_dim * sizeof(double));
    for (unsigned i = 0u; i < f_dim; ++i) ref[i] = 1.0 * REF_POINT_MULTIPLIER;
    return ref;
};

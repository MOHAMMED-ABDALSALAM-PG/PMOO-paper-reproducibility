#include "problems/suites/dtlz.cuh"

#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>

// we need to define device pointers this way, so we can get function address on gpu on the host side

__device__ void (*dtlz1_fitness_dptr)(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) = dtlz1_fitness;

__device__ void (*dtlz2_fitness_dptr)(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) = dtlz2_fitness;

__device__ void (*dtlz3_fitness_dptr)(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) = dtlz3_fitness;

__device__ void (*dtlz4_fitness_dptr)(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) = dtlz4_fitness;

__device__ void (*dtlz5_fitness_dptr)(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) = dtlz5_fitness;

__device__ void (*dtlz6_fitness_dptr)(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) = dtlz6_fitness;

__device__ void (*dtlz7_fitness_dptr)(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) = dtlz7_fitness;

// custom function to choose front size based on f_dim
__host__ inline double choose_front_size_per_dimension(const unsigned f_dim) {
    return f_dim > 4u ? 6.0 : 10.0;
};

// NOTE: d_dim has to be larger than f_dim (d_dim > f_dim)
__host__ Problem* generate_dtlz_problem(unsigned problem_id, unsigned d_dim, unsigned f_dim) {
    if (problem_id == 0u || problem_id > TEST_SUITES_PROBLEM_COUNT[DTLZ]) return NULL;

    Problem* problem = (Problem*)malloc(sizeof(Problem));
    problem->suite = DTLZ;
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
    /*
     * NOTE:
     * Our implementation of DTLZ pareto front is not evenly distributed,
     * but all points should be lying on the actual front. Take that into
     * consideration when evaluating GD and IGD metrics for these problems.
     */
    problem->front_size = pow(choose_front_size_per_dimension(f_dim), f_dim - 1.0);
    problem->hypervolume_N = min((unsigned)pow(25u, f_dim), 10000000u);
    problem->fitness = NULL;
    problem->gpu_fitness = NULL;
    problem->front = NULL;
    problem->ref_point = NULL;

    switch (problem_id) {
        case 1u:
            problem->fitness = dtlz1_fitness;
            cudaMemcpyFromSymbol(&problem->gpu_fitness, dtlz1_fitness_dptr, sizeof(Fitness));
            problem->front = dtlz1_pareto_front(problem->d_dim, problem->f_dim, problem->front_size);
            problem->ref_point = dtlz1_reference_point(problem->d_dim, problem->f_dim, problem->upper_bounds, problem->lower_bounds);
            break;
        case 2u:
            problem->fitness = dtlz2_fitness;
            cudaMemcpyFromSymbol(&problem->gpu_fitness, dtlz2_fitness_dptr, sizeof(Fitness));
            problem->front = dtlz2_pareto_front(problem->d_dim, problem->f_dim, problem->front_size);
            problem->ref_point = dtlz2_reference_point(problem->d_dim, problem->f_dim, problem->upper_bounds, problem->lower_bounds);
            break;
        case 3u:
            problem->fitness = dtlz3_fitness;
            cudaMemcpyFromSymbol(&problem->gpu_fitness, dtlz3_fitness_dptr, sizeof(Fitness));
            problem->front = dtlz3_pareto_front(problem->d_dim, problem->f_dim, problem->front_size);
            problem->ref_point = dtlz3_reference_point(problem->d_dim, problem->f_dim, problem->upper_bounds, problem->lower_bounds);
            break;
        case 4u:
            problem->fitness = dtlz4_fitness;
            cudaMemcpyFromSymbol(&problem->gpu_fitness, dtlz4_fitness_dptr, sizeof(Fitness));
            problem->front = dtlz4_pareto_front(problem->d_dim, problem->f_dim, problem->front_size);
            problem->ref_point = dtlz4_reference_point(problem->d_dim, problem->f_dim, problem->upper_bounds, problem->lower_bounds);
            break;
        case 5u:
            problem->fitness = dtlz5_fitness;
            cudaMemcpyFromSymbol(&problem->gpu_fitness, dtlz5_fitness_dptr, sizeof(Fitness));
            problem->front = dtlz5_pareto_front(problem->d_dim, problem->f_dim, problem->front_size);
            problem->ref_point = dtlz5_reference_point(problem->d_dim, problem->f_dim, problem->upper_bounds, problem->lower_bounds);
            break;
        case 6u:
            problem->fitness = dtlz6_fitness;
            cudaMemcpyFromSymbol(&problem->gpu_fitness, dtlz6_fitness_dptr, sizeof(Fitness));
            problem->front = dtlz6_pareto_front(problem->d_dim, problem->f_dim, problem->front_size);
            problem->ref_point = dtlz6_reference_point(problem->d_dim, problem->f_dim, problem->upper_bounds, problem->lower_bounds);
            break;
        case 7u:
            problem->fitness = dtlz7_fitness;
            cudaMemcpyFromSymbol(&problem->gpu_fitness, dtlz7_fitness_dptr, sizeof(Fitness));
            problem->front = dtlz7_pareto_front(problem->d_dim, problem->f_dim, problem->front_size);
            problem->ref_point = dtlz7_reference_point(problem->d_dim, problem->f_dim, problem->upper_bounds, problem->lower_bounds);
            break;
    }

    return problem;
};

/*
 * NOTE:
 * G function for DTLZ1 and DTLZ3 has "100.0 *" in the front.
 * We omitted it, because it makes fitness values and metric values enormous,
 * while it seems it doesnt matter at all.
 */
__host__ __device__ double g_function_dtlz13(const double *decision_variables, const unsigned d_dim, const unsigned f_dim) {
    double g = 0.0;

    for (unsigned i = f_dim - 1u; i < d_dim; ++i)
        g += pow(decision_variables[i] - 0.5, 2.0) - cos(20.0 * M_PI * (decision_variables[i] - 0.5));
    g = (g + (double)(d_dim - f_dim + 1u));

    return g;
};

__host__ __device__ double g_function_dtlz245(const double *decision_variables, const unsigned d_dim, const unsigned f_dim) {
    double g = 0.0;

    for (unsigned i = f_dim - 1u; i < d_dim; ++i)
        g += pow(decision_variables[i] - 0.5, 2.0);
    
    return g;
};

__host__ __device__ double g_function_dtlz6(const double *decision_variables, const unsigned d_dim, const unsigned f_dim) {
    double g = 0.0;

    for (unsigned i = f_dim - 1u; i < d_dim; ++i)
        g += pow(decision_variables[i], 0.1);

    return g;
};

__host__ __device__ double g_function_dtlz7(const double *decision_variables, const unsigned d_dim, const unsigned f_dim) {
    double g = 0.0;

    for (unsigned i = f_dim - 1u; i < d_dim; ++i)
        g += decision_variables[i];
    g = 1.0 + 9.0 / (double)(d_dim - f_dim + 1u) * g;
    
    return g;
};

/*
 * NOTE:
 * DTLZ problems divide decision variables into 2 parts: x_ and x_M.
 * These 2 ranges are dependent on f_dim and d_dim. x_M is used to calculate g functions.
 * This division applies to all DTLZ problems.
 * 
 * They are calculated as follows:
 * x_  = [0, f_dim - 1u)
 * x_M = [f_dim - 1u, d_dim)
 */
__host__ __device__ void dtlz1_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) {
    double g = g_function_dtlz13(decision_variables, d_dim, f_dim);

    for (unsigned i = 0u; i < f_dim; ++i) {
        fitness_values[i] = 0.5 * (1.0 + g);
        for (unsigned j = 0u; j < f_dim - 1u - i; ++j)
            fitness_values[i] *= decision_variables[j];
        if (i > 0u)
            fitness_values[i] *= 1.0 - decision_variables[f_dim - 1u - i];
    }
};

__host__ __device__ void dtlz2_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) {
    double g = g_function_dtlz245(decision_variables, d_dim, f_dim);
    
    for (unsigned i = 0u; i < f_dim; ++i) {
        fitness_values[i] = 1.0 + g;
        for (unsigned j = 0u; j < f_dim - 1u - i; ++j)
            fitness_values[i] *= cos(decision_variables[j] * M_PI_2);
        if (i > 0u)
            fitness_values[i] *= sin(decision_variables[f_dim - 1u - i] * M_PI_2);
    }
};

__host__ __device__ void dtlz3_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) {
    double g = g_function_dtlz13(decision_variables, d_dim, f_dim);

    for (unsigned i = 0u; i < f_dim; ++i) {
        fitness_values[i] = 1.0 + g;
        for (unsigned j = 0u; j < f_dim - 1u - i; ++j)
            fitness_values[i] *= cos(decision_variables[j] * M_PI_2);
        if (i > 0u)
            fitness_values[i] *= sin(decision_variables[f_dim - 1u - i] * M_PI_2);
    }
};

__host__ __device__ void dtlz4_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) {
    double g = g_function_dtlz245(decision_variables, d_dim, f_dim);
    
    for (unsigned i = 0u; i < f_dim; ++i) {
        fitness_values[i] = 1.0 + g;
        for (unsigned j = 0u; j < f_dim - 1u - i; ++j)
            fitness_values[i] *= cos(pow(decision_variables[j], 100.0) * M_PI_2);
        if (i > 0u)
            fitness_values[i] *= sin(pow(decision_variables[f_dim - 1u - i], 100.0) * M_PI_2);
    }
};

__host__ __device__ void dtlz5_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) {
    double g = g_function_dtlz245(decision_variables, d_dim, f_dim);
    double theta = 1.0 / (2.0 * (1.0 + g));
    
    for (unsigned i = 0u; i < f_dim; ++i) {
        fitness_values[i] = 1.0 + g;
        for (unsigned j = 0u; j < f_dim - 1u - i; ++j)
            if (j == 0) // NOTE: original paper states that theta shouldnt be applied on the first decision variable
                fitness_values[i] *= cos(decision_variables[j] * M_PI_2);
            else
                fitness_values[i] *= cos(theta * (2.0 * g * decision_variables[j] + 1.0) * M_PI_2);
        if (i > 0u)
            if (i == f_dim - 1u) // NOTE: last fitness value also shouldnt have theta applied
                fitness_values[i] *= sin(decision_variables[f_dim - 1u - i] * M_PI_2);
            else
                fitness_values[i] *= sin(theta * (2.0 * g * decision_variables[f_dim - 1u - i] + 1.0) * M_PI_2);
    }
};

__host__ __device__ void dtlz6_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) {
    double g = g_function_dtlz6(decision_variables, d_dim, f_dim);
    double theta = 1.0 / (2.0 * (1.0 + g));
    
    for (unsigned i = 0u; i < f_dim; ++i) {
        fitness_values[i] = 1.0 + g;
        for (unsigned j = 0u; j < f_dim - 1u - i; ++j)
            if (j == 0) // NOTE: original paper states that theta shouldnt be applied on the first decision variable
                fitness_values[i] *= cos(decision_variables[j] * M_PI_2);
            else
                fitness_values[i] *= cos(theta * (2.0 * g * decision_variables[j] + 1.0) * M_PI_2);
        if (i > 0u)
            if (i == f_dim - 1u) // NOTE: last fitness value also shouldnt have theta applied
                fitness_values[i] *= sin(decision_variables[f_dim - 1u - i] * M_PI_2);
            else
                fitness_values[i] *= sin(theta * (2.0 * g * decision_variables[f_dim - 1u - i] + 1.0) * M_PI_2);
    }
};

__host__ __device__ void dtlz7_fitness(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim) {
    double g = g_function_dtlz7(decision_variables, d_dim, f_dim);

    for (unsigned i = 0u; i < f_dim - 1u; ++i)
        fitness_values[i] = decision_variables[i];

    double h = f_dim;
    for (unsigned i = 0u; i < f_dim - 1u; ++i)
        h -= (fitness_values[i] / (1.0 + g)) * (1.0 + sin(3.0 * M_PI * fitness_values[i]));
    fitness_values[f_dim - 1u] = (1.0 + g) * h;
};

__host__ double** dtlz1_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size) {
    unsigned front_size_per_dimension = round(pow(size, 1.0 / (f_dim - 1.0)));

    unsigned *pows = (unsigned*)malloc((f_dim - 1u) * sizeof(unsigned));
    for (unsigned dim = 0u; dim < f_dim - 1u; ++dim)
        pows[dim] = (unsigned)floor(pow(front_size_per_dimension, dim));

    double *x = (double*)malloc(d_dim * sizeof(double));
    for (unsigned dim = 0u; dim < d_dim; ++dim)
        x[dim] = 0.5;
    
    double **f = (double**)malloc(size * sizeof(double*));
    double *_f = (double*)malloc(size * f_dim * sizeof(double));
    for (unsigned i = 0u; i < size; ++i)
        f[i] = &_f[i * f_dim];
    
    double step = 1.0 / (front_size_per_dimension - 1.0);
    for (unsigned i = 0u; i < size; ++i) {
        for (unsigned dim = 0u; dim < f_dim - 1u; ++dim)
            x[dim] = ((i / pows[dim]) % front_size_per_dimension) * step; // generate evenly spaced points
        dtlz1_fitness(x, f[i], d_dim, f_dim);
    }

    free(pows);
    free(x);
    return f;
};

__host__ double** dtlz2_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size) {
    unsigned front_size_per_dimension = round(pow(size, 1.0 / (f_dim - 1.0)));
    
    unsigned *pows = (unsigned*)malloc((f_dim - 1u) * sizeof(unsigned));
    for (unsigned dim = 0u; dim < f_dim - 1u; ++dim)
        pows[dim] = (unsigned)floor(pow(front_size_per_dimension, dim));

    double *x = (double*)malloc(d_dim * sizeof(double));
    for (unsigned dim = 0u; dim < d_dim; ++dim)
        x[dim] = 0.5;
    
    double **f = (double**)malloc(size * sizeof(double*));
    double *_f = (double*)malloc(size * f_dim * sizeof(double));
    for (unsigned i = 0u; i < size; ++i)
        f[i] = &_f[i * f_dim];
    
    double step = 1.0 / (front_size_per_dimension - 1.0);
    for (unsigned i = 0u; i < size; ++i) {
        for (unsigned dim = 0u; dim < f_dim - 1u; ++dim)
            x[dim] = ((i / pows[dim]) % front_size_per_dimension) * step; // generate evenly spaced points
        dtlz2_fitness(x, f[i], d_dim, f_dim);
    }

    free(pows);
    free(x);
    return f;
};

__host__ double** dtlz3_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size) {
    unsigned front_size_per_dimension = round(pow(size, 1.0 / (f_dim - 1.0)));
    
    unsigned *pows = (unsigned*)malloc((f_dim - 1u) * sizeof(unsigned));
    for (unsigned dim = 0u; dim < f_dim - 1u; ++dim)
        pows[dim] = (unsigned)floor(pow(front_size_per_dimension, dim));

    double *x = (double*)malloc(d_dim * sizeof(double));
    for (unsigned dim = 0u; dim < d_dim; ++dim)
        x[dim] = 0.5;
    
    double **f = (double**)malloc(size * sizeof(double*));
    double *_f = (double*)malloc(size * f_dim * sizeof(double));
    for (unsigned i = 0u; i < size; ++i)
        f[i] = &_f[i * f_dim];
    
    double step = 1.0 / (front_size_per_dimension - 1.0);
    for (unsigned i = 0u; i < size; ++i) {
        for (unsigned dim = 0u; dim < f_dim - 1u; ++dim)
            x[dim] = ((i / pows[dim]) % front_size_per_dimension) * step; // generate evenly spaced points
        dtlz3_fitness(x, f[i], d_dim, f_dim);
    }

    free(pows);
    free(x);
    return f;
};

/*
 * NOTE:
 * For DTLZ4 we are calculating 100th root for each decision variable in x_,
 * because DTLZ4 raises x_ to 100th power. We need to do this to ensure
 * even distribution of pareto front. We are calculating 100th root using power of 1/100.
 */
__host__ double** dtlz4_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size) {
    unsigned front_size_per_dimension = round(pow(size, 1.0 / (f_dim - 1.0)));
    
    unsigned *pows = (unsigned*)malloc((f_dim - 1u) * sizeof(unsigned));
    for (unsigned dim = 0u; dim < f_dim - 1u; ++dim)
        pows[dim] = (unsigned)floor(pow(front_size_per_dimension, dim));

    double *x = (double*)malloc(d_dim * sizeof(double));
    for (unsigned dim = 0u; dim < d_dim; ++dim)
        x[dim] = 0.5;
    
    double **f = (double**)malloc(size * sizeof(double*));
    double *_f = (double*)malloc(size * f_dim * sizeof(double));
    for (unsigned i = 0u; i < size; ++i)
        f[i] = &_f[i * f_dim];
    
    double step = 1.0 / (front_size_per_dimension - 1.0);
    for (unsigned i = 0u; i < size; ++i) {
        for (unsigned dim = 0u; dim < f_dim - 1u; ++dim)
            x[dim] = pow(((i / pows[dim]) % front_size_per_dimension) * step, 0.01); // generate evenly spaced points
        dtlz4_fitness(x, f[i], d_dim, f_dim);
    }

    free(pows);
    free(x);
    return f;
};

__host__ double** dtlz5_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size) {
    unsigned front_size_per_dimension = round(pow(size, 1.0 / (f_dim - 1.0)));
    
    double *x = (double*)malloc(d_dim * sizeof(double));
    for (unsigned dim = 0u; dim < d_dim; ++dim)
        x[dim] = 0.5;

    double **f = (double**)malloc(size * sizeof(double*));
    double *_f = (double*)malloc(size * f_dim * sizeof(double));
    for (unsigned i = 0u; i < size; ++i)
        f[i] = &_f[i * f_dim];
    
    double step = 1.0 / (size - 1.0);
    for (unsigned i = 0u; i < size; ++i) {
        x[0] = i * step;
        dtlz5_fitness(x, f[i], d_dim, f_dim);
    }

    free(x);
    return f;
};

__host__ double** dtlz6_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size) {
    unsigned front_size_per_dimension = round(pow(size, 1.0 / (f_dim - 1.0)));
    
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
        dtlz6_fitness(x, f[i], d_dim, f_dim);
    }

    free(x);
    return f;
};

/*
 * NOTE:
 * DTLZ7 pareto front is made of 4 seperate patches in the fitness space.
 * Because of that we need to split generated decision variables
 * into these 4 patches to properly generate pareto front.
 */
__host__ double** dtlz7_pareto_front(const unsigned d_dim, const unsigned f_dim, const unsigned size) {
    unsigned front_size_per_dimension = round(pow(size, 1.0 / (f_dim - 1.0)));
    
    unsigned *pows = (unsigned*)malloc((f_dim - 1u) * sizeof(unsigned));
    for (unsigned dim = 0u; dim < f_dim - 1u; ++dim)
        pows[dim] = (unsigned)floor(pow(front_size_per_dimension, dim));

    double *x = (double*)malloc(d_dim * sizeof(double));
    for (unsigned dim = 0u; dim < d_dim; ++dim)
        x[dim] = 0.0;
    
    double **f = (double**)malloc(size * sizeof(double*));
    double *_f = (double*)malloc(size * f_dim * sizeof(double));
    for (unsigned i = 0u; i < size; ++i)
        f[i] = &_f[i * f_dim];
    
    double step = 1.0 / (front_size_per_dimension - 1.0);
    for (unsigned i = 0u; i < size; ++i) {
        for (unsigned dim = 0u; dim < f_dim - 1u; ++dim) {
            x[dim] = ((i / pows[dim]) % front_size_per_dimension) * step; // generate evenly spaced points
            if (x[dim] < 0.5) {
                x[dim] = x[dim] * 0.5;
            } else {
                x[dim] = (x[dim] - 0.5) * 0.5 + 0.6;
            }
        }
        dtlz7_fitness(x, f[i], d_dim, f_dim);
    }

    free(pows);
    free(x);
    return f;
};

/*
 * NOTE:
 * Reference points for DTLZ are calculated by upper bound analysis
 * of original formulas for each problem. Then each each value is being
 * multiplied by 1.2 to make sure that hypervolume will contain all points in population.
 */
__host__ double* dtlz1_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds) {
    double *ref = (double*)malloc(f_dim * sizeof(double));
    double max_fitness_value = 0.5 * (1.0 + g_function_dtlz13(upper_bounds, d_dim, f_dim));
    for (unsigned i = 0u; i < f_dim; ++i) ref[i] = max_fitness_value * REF_POINT_MULTIPLIER;
    return ref;
};

__host__ double* dtlz2_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds) {
    double *ref = (double*)malloc(f_dim * sizeof(double));
    double max_fitness_value = 1.0 + g_function_dtlz245(upper_bounds, d_dim, f_dim);
    for (unsigned i = 0u; i < f_dim; ++i) ref[i] = max_fitness_value * REF_POINT_MULTIPLIER;
    return ref;
};

__host__ double* dtlz3_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds) {
    double *ref = (double*)malloc(f_dim * sizeof(double));
    double max_fitness_value = 1.0 + g_function_dtlz13(upper_bounds, d_dim, f_dim);
    for (unsigned i = 0u; i < f_dim; ++i) ref[i] = max_fitness_value * REF_POINT_MULTIPLIER;
    return ref;
};

__host__ double* dtlz4_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds) {
    double *ref = (double*)malloc(f_dim * sizeof(double));
    double max_fitness_value = 1.0 + g_function_dtlz245(upper_bounds, d_dim, f_dim);
    for (unsigned i = 0u; i < f_dim; ++i) ref[i] = max_fitness_value * REF_POINT_MULTIPLIER;
    return ref;
};

__host__ double* dtlz5_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds) {
    double *ref = (double*)malloc(f_dim * sizeof(double));
    double max_fitness_value = 1.0 + g_function_dtlz245(upper_bounds, d_dim, f_dim);
    for (unsigned i = 0u; i < f_dim; ++i) ref[i] = max_fitness_value * REF_POINT_MULTIPLIER;
    return ref;
};

__host__ double* dtlz6_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds) {
    double *ref = (double*)malloc(f_dim * sizeof(double));
    double max_fitness_value = 1.0 + g_function_dtlz6(upper_bounds, d_dim, f_dim);
    for (unsigned i = 0u; i < f_dim; ++i) ref[i] = max_fitness_value * REF_POINT_MULTIPLIER;
    return ref;
};

/*
 * NOTE:
 * Final dimension of a reference point
 * probably can have a smaller value, but we
 * were not sure how to better calculate upper bound for it.
 */
__host__ double* dtlz7_reference_point(const unsigned d_dim, const unsigned f_dim, const double *upper_bounds, const double *lower_bounds) {
    double *ref = (double*)malloc(f_dim * sizeof(double));
    for (unsigned i = 0u; i < f_dim - 1u; ++i) ref[i] = upper_bounds[i] * REF_POINT_MULTIPLIER;
    ref[f_dim - 1u] = (1.0 + g_function_dtlz7(upper_bounds, d_dim, f_dim)) * f_dim * REF_POINT_MULTIPLIER;
    return ref;
};

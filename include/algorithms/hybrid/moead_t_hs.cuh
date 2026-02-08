#ifndef PMOO_HYBRID_MOEAD_T_HS_CUH
#define PMOO_HYBRID_MOEAD_T_HS_CUH

#include "../algorithm.cuh"
#include "../utils.cuh"

#include <random> // mt19937, random_device and distributions
#include <algorithm> // sort

#include <stdlib.h>
#include <string.h> // memcpy, memset
#include <math.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio> 
#include <cassert>

#ifndef TOLERANCE_EPSILON
#define TOLERANCE_EPSILON 1e-8
#endif

#define MOEAD_NEIGHBORHOOD_SIZE 20 // Size of neighborhood for each subproblem
#define INF 1e30

// Initialize CURAND states
template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
__global__ void init_curand_hs_hybrid_states(
    curandState* states,
    unsigned long long seed) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < POP_SIZE) {
        curand_init(seed, id, 0, &states[id]);
    }
}

__device__ double halton(unsigned index, unsigned base) {
    double f = 1.0;
    double r = 0.0;
    while (index > 0) {
        f /= base;
        r += f * (index % base);
        index /= base;
    }
    return r;
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
__global__ void generate_weight_vectors_halton_kernel(
    double* __restrict__ device_weight_vectors)
{
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= POP_SIZE) return;

    // Halton primes for bases (up to F_DIM-1 dimensions)
    constexpr unsigned HALTON_PRIMES[10] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};

    // Generate Halton coords in (0,1)
    double h[F_DIM - 1];
    for (unsigned d = 0; d < F_DIM - 1; ++d) {
        h[d] = halton(tid + 1, HALTON_PRIMES[d]);  // Halton uses 1-based index
    }

    // Build array: [0, h[0..F_DIM-2], 1]
    double s[F_DIM + 1];
    s[0] = 0.0;
    for (unsigned d = 0; d < F_DIM - 1; ++d)
        s[d + 1] = h[d];
    s[F_DIM] = 1.0;

    // Insertion sort on registers
    for (unsigned i = 1; i <= F_DIM; ++i) {
        double key = s[i];
        int j = i - 1;
        while (j >= 0 && s[j] > key) {
            s[j + 1] = s[j];
            j--;
        }
        s[j + 1] = key;
    }

    // Compute weights as differences
    for (unsigned d = 0; d < F_DIM; ++d) {
        device_weight_vectors[tid * F_DIM + d] = s[d + 1] - s[d];
    }
}



template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
__global__ void build_neighborhood_hs_hybrid_kernel(
    unsigned* device_neighborhood,
    const double* __restrict__ device_weight_vectors) {

    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= POP_SIZE) return;

    constexpr unsigned K = MOEAD_NEIGHBORHOOD_SIZE;
    double query[F_DIM];

    #pragma unroll
    for (unsigned m = 0; m < F_DIM; ++m) {
        query[m] = device_weight_vectors[i * F_DIM + m];
    }

    // Top-K buffers in registers
    double top_dists[K];
    unsigned top_idxs[K];

    // Initialize to worst possible values
    #pragma unroll
    for (unsigned k = 0; k < K; ++k) {
        top_dists[k] = INF;
        top_idxs[k] = i;  // fallback: self index
    }

    // Brute-force scan
    for (unsigned j = 0; j < POP_SIZE; ++j) {
        if (j == i) continue;

        double sumsq = 0.0;
        #pragma unroll
        for (unsigned m = 0; m < F_DIM; ++m) {
            double diff = query[m] - device_weight_vectors[j * F_DIM + m];
            sumsq += diff * diff;
        }

        // Check if this one is better than the worst in top-K
        if (sumsq < top_dists[K - 1]) {
            // Insert in sorted order
            unsigned pos = K - 1;
            while (pos > 0 && sumsq < top_dists[pos - 1]) {
                top_dists[pos] = top_dists[pos - 1];
                top_idxs[pos] = top_idxs[pos - 1];
                --pos;
            }
            top_dists[pos] = sumsq;
            top_idxs[pos] = j;
        }
    }

    // Write top-K indices to global memory
    #pragma unroll
    for (unsigned k = 0; k < K; ++k) {
        device_neighborhood[i * K + k] = top_idxs[k];
    }
}



template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
struct moead_t_hs_hybrid {
    // algorithm settings
    unsigned gen;           // Number of generations to run in a single evolve call
    double p_crossover;     // Crossover probability
    double eta_crossover;   // Crossover index (determines how similar are children to parents)
    double p_mutation;      // Mutation probability
    double eta_mutation;    // Mutation index (determines how big of a change the mutated value will be)
    // algorithm and problem metadata
    unsigned pop_size;          // How many individuals are in population
    unsigned d_dim;             // How many decision variables does the problem have
    unsigned f_dim;             // How many objectives does the problem have
    unsigned i_dim;             // How many integer dimensions does the problem have (integer variables are stored at the end of deicision vector)
    double upper_bounds[D_DIM]; // Upper bounds for decision variables values based on problem constraints
    double lower_bounds[D_DIM]; // Lower bounds for decision variables values based on problem constraints

    void (*fitness)(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim); // fitness evaluation function
    double decision_variables[POP_SIZE][D_DIM];                            // Population decision variables
    double fitness_values[POP_SIZE][F_DIM];                               // Population fitness values
    double ideal_point[F_DIM]; 
    double offspring_vars[POP_SIZE][D_DIM];
    double offspring_fitness[POP_SIZE][F_DIM];
    double weight_vectors[POP_SIZE][F_DIM];                           // Decomposition weight vectors
    unsigned neighborhood[POP_SIZE][MOEAD_NEIGHBORHOOD_SIZE];        // Neighborhood indices

    // MOEA/D CUDA-specific  
    double* device_weight_vectors;
    unsigned* device_neighborhood;
    curandState* deviceStates;
    
    dim3 THREADS;
    dim3 BLOCKS;
    std::mt19937 random[POP_SIZE];
    unsigned num_threads;
    omp_lock_t lock[POP_SIZE];
};


template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
// Problem metadata should be loaded first before initializing algorithm
void moead_t_hs_hybrid_initialize(struct moead_t_hs_hybrid<POP_SIZE, D_DIM, F_DIM> *moead) {

    omp_set_num_threads(moead->num_threads);
    // initialize random number generators
    for (unsigned i = 0; i < POP_SIZE; ++i) {
        moead->random[i].seed(std::random_device{}());
        omp_init_lock(&moead->lock[i]);
    }


    // Initialize decision variables (random within bounds)
    // Continuous variables
    for (unsigned dim = 0; dim < moead->d_dim - moead->i_dim; ++dim) {
        std::uniform_real_distribution<double> uniform(moead->lower_bounds[dim], moead->upper_bounds[dim]);
        for (unsigned i = 0; i < moead->pop_size; ++i) {
            moead->decision_variables[i][dim] = uniform(moead->random[i]);
        }
    }
    // Integer variables (stored as double)
    for (unsigned dim = moead->d_dim - moead->i_dim; dim < moead->d_dim; ++dim) {
        std::uniform_real_distribution<double> uniform(moead->lower_bounds[dim], moead->upper_bounds[dim]);
        for (unsigned i = 0; i < moead->pop_size; ++i) {
            moead->decision_variables[i][dim] = uniform(moead->random[i]);
        }
    }
    // Evaluate initial population
    #pragma omp parallel for
    for (unsigned i = 0; i < moead->pop_size; ++i) {
        moead->fitness(moead->decision_variables[i], moead->fitness_values[i],
                       moead->d_dim, moead->f_dim);
    }
    // Initialize ideal point (minimum of objectives)
    for (unsigned j = 0; j < moead->f_dim; ++j) {
        moead->ideal_point[j] = moead->fitness_values[0][j];
    }
    
    for (unsigned i = 1; i < moead->pop_size; ++i) {
        for (unsigned j = 0; j < moead->f_dim; ++j) {
            if (moead->fitness_values[i][j] < moead->ideal_point[j]) {
                moead->ideal_point[j] = moead->fitness_values[i][j];
            }
        }
    }
    // init curand states
    unsigned threads = moead->THREADS.x;
    unsigned blocks  = (moead->pop_size + threads - 1) / threads;
    CUDA_CALL(cudaMalloc(&moead->deviceStates, POP_SIZE * sizeof(curandState)));
    CUDA_CALL(cudaMalloc(&moead->device_neighborhood, POP_SIZE * MOEAD_NEIGHBORHOOD_SIZE * sizeof(unsigned)));
    CUDA_CALL(cudaMemset(moead->device_neighborhood, 0, POP_SIZE * MOEAD_NEIGHBORHOOD_SIZE * sizeof(unsigned)));
    CUDA_CALL(cudaMalloc(&moead->device_weight_vectors, POP_SIZE * F_DIM * sizeof(double)));
    CUDA_CALL(cudaMemset(moead->device_weight_vectors, 0, POP_SIZE * F_DIM * sizeof(double)));

    init_curand_hybrid_states<POP_SIZE,D_DIM,F_DIM><<<blocks, threads>>>(
        moead->deviceStates,
        (unsigned long long)std::random_device{}()
    );

    CUDA_CALL(cudaDeviceSynchronize());
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
void moead_t_hs_hybrid_evolve(struct moead_t_hs_hybrid<POP_SIZE, D_DIM, F_DIM> *moead) {
    // 1D launch parameters
    auto tpb = moead->THREADS.x;
    auto bl  = moead->BLOCKS.x;

    generate_weight_vectors_halton_kernel<POP_SIZE, D_DIM, F_DIM><<<bl, tpb>>>(
        moead->device_weight_vectors);
    CUDA_CALL(cudaDeviceSynchronize());


    build_neighborhood_hs_hybrid_kernel<POP_SIZE,D_DIM,F_DIM><<<bl,tpb>>>(
        moead->device_neighborhood,
        moead->device_weight_vectors
    );
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(
        moead->neighborhood, moead->device_neighborhood, 
        POP_SIZE * MOEAD_NEIGHBORHOOD_SIZE * sizeof(unsigned), 
        cudaMemcpyDeviceToHost)
    );


    for (unsigned gen = 0; gen < moead->gen; ++gen) {
        // Loop through each subproblem i
        #pragma omp parallel for
        for (OMP_FOR_INDEX i = 0; i < POP_SIZE; ++i) {
            std::mt19937 &rng = moead->random[i];
            double *offspring_vars = moead->offspring_vars[i];
            double *offspring_fitness = moead->offspring_fitness[i];
            // Reproduction – choose two random neighbors of i for parents
            unsigned neighbor_size = (MOEAD_NEIGHBORHOOD_SIZE < POP_SIZE ? MOEAD_NEIGHBORHOOD_SIZE : POP_SIZE);
            std::uniform_int_distribution<unsigned> dist(0, neighbor_size - 1);

            unsigned index1 = moead->neighborhood[i][dist(rng)];
            unsigned index2 = moead->neighborhood[i][dist(rng)];
            // Ensure distinct parents
            while (index2 == index1 && neighbor_size > 1) {
                
                index2 = moead->neighborhood[i][dist(rng)];
            }
            double *parent1 = moead->decision_variables[index1];
            double *parent2 = moead->decision_variables[index2];
            
           // Perform crossover and mutation to produce an offspring solution
            moead_t_hs_hybrid_simulated_binary_crossover(moead, parent1, parent2, offspring_vars, i);
            moead_t_hs_hybrid_mutation(moead, offspring_vars, i);
            // Evaluate the offspring's objectives
            moead->fitness(offspring_vars, offspring_fitness, D_DIM, F_DIM);
            
            // Update ideal point if offspring is better
            for (unsigned j = 0; j < F_DIM; ++j) {
                
                if (offspring_fitness[j] < moead->ideal_point[j]) {
                    #pragma omp atomic write
                    moead->ideal_point[j] = offspring_fitness[j];
                }
            }

            // Update neighboring solutions
            for (unsigned k = 0; k < neighbor_size; ++k) {
                unsigned j = moead->neighborhood[i][k];
                // Calculate Tchebycheff scalar value for current solution j and for offspring
                double f_current = moead_t_hs_hybrid_calc_tchebycheff(moead, j, moead->fitness_values[j]);
                double f_offspring = moead_t_hs_hybrid_calc_tchebycheff(moead, j, offspring_fitness);
                if (f_offspring < f_current - TOLERANCE_EPSILON) {
                    omp_set_lock(&moead->lock[j]);
                    // Replace solution j with the offspring
                    memcpy(moead->decision_variables[j], offspring_vars, D_DIM * sizeof(double));
                    memcpy(moead->fitness_values[j], offspring_fitness, F_DIM * sizeof(double));
                    omp_unset_lock(&moead->lock[j]);
                }
            }
        }
    }
}

// Calculate the Tchebycheff scalarizing value for a given subproblem and objective vector
template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
double moead_t_hs_hybrid_calc_tchebycheff(struct moead_t_hs_hybrid<POP_SIZE, D_DIM, F_DIM> *moead, unsigned subproblem_index, const double *obj_values) {
    double max_val = -1e30;
    const double *lambda = moead->weight_vectors[subproblem_index];
    for (unsigned j = 0; j < F_DIM; ++j) {
        // Distance from ideal point on objective j
        double diff = obj_values[j] - moead->ideal_point[j];
        if (diff < 0.0) diff = 0.0; // (should not happen if ideal_point[j] is min seen, but just in case)

        // Weighted deviation
        double weighted;
        if (lambda[j] < 1e-6)
            weighted = 1e-4 * diff;
        else
            weighted = lambda[j] * diff;

        if (weighted > max_val)
            max_val = weighted;
    }
    return max_val;
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
void moead_t_hs_hybrid_simulated_binary_crossover(struct moead_t_hs_hybrid<POP_SIZE, D_DIM, F_DIM> *moead,
                                             const double *parent1,
                                             const double *parent2,
                                             double *offspring,
                                             unsigned index) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    if (uniform(moead->random[index]) < moead->p_crossover) {
        for (unsigned dim = 0; dim < moead->d_dim - moead->i_dim; ++dim) {
            double y1 = parent1[dim];
            double y2 = parent2[dim];
            if (fabs(y1 - y2) < TOLERANCE_EPSILON) {
                offspring[dim] = y1;
                continue;
            }

            if (y1 > y2) std::swap(y1, y2);

            double lb = moead->lower_bounds[dim];
            double ub = moead->upper_bounds[dim];
            double randu = uniform(moead->random[index]);
            double beta = 1.0 + (2.0 * (y1 - lb) / (y2 - y1));
            double alpha = 2.0 - pow(beta, -(moead->eta_crossover + 1.0));
            double betaq;

            if (randu <= (1.0 / alpha)) {
                betaq = pow(randu * alpha, 1.0 / (moead->eta_crossover + 1.0));
            } else {
                betaq = pow(1.0 / (2.0 - randu * alpha), 1.0 / (moead->eta_crossover + 1.0));
            }

            double c = 0.5 * ((y1 + y2) - betaq * (y2 - y1));
            if (c < lb) c = lb;
            if (c > ub) c = ub;

            offspring[dim] = c;
        }

        // Integer variable part: two-point crossover for i_dim (if any)
        if (moead->i_dim > 0) {
            std::uniform_int_distribution<unsigned> i_dim_dist(moead->d_dim - moead->i_dim, moead->d_dim - 1);
            unsigned cut1 = i_dim_dist(moead->random[index]);
            unsigned cut2 = i_dim_dist(moead->random[index]);
            if (cut1 > cut2) std::swap(cut1, cut2);
            for (unsigned dim = moead->d_dim - moead->i_dim; dim < moead->d_dim; ++dim) {
                if (dim >= cut1 && dim <= cut2) {
                    offspring[dim] = parent2[dim];
                } else {
                    offspring[dim] = parent1[dim];
                }
            }
        }
    }else
    {
        for (unsigned dim = 0; dim < moead->d_dim; ++dim) {
            offspring[dim] = (uniform(moead->random[index]) < 0.5)
                ? parent1[dim]
                : parent2[dim];
        }
    }
}

// Polynomial mutation for a single offspring solution
template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
static void moead_t_hs_hybrid_mutation(struct moead_t_hs_hybrid<POP_SIZE, D_DIM, F_DIM> *moead, double *individual, unsigned index) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (unsigned j = 0; j < moead->d_dim; ++j) {
        if (uniform(moead->random[index]) <= moead->p_mutation) {
            double lb = moead->lower_bounds[j];
            double ub = moead->upper_bounds[j];
            if (ub - lb < TOLERANCE_EPSILON) {
                continue; // no range
            }
            double x = individual[j];
            double delta1 = (x - lb) / (ub - lb);
            double delta2 = (ub - x) / (ub - lb);
            double randu = uniform(moead->random[index]);
            double mut_pow = 1.0 / (moead->eta_mutation + 1.0);
            double deltaq;
            if (randu < 0.5) {
                double xy = 1.0 - delta1;
                double val = 2.0 * randu + (1.0 - 2.0 * randu) * pow(xy, (moead->eta_mutation + 1.0));
                deltaq = pow(val, mut_pow) - 1.0;
            } else {
                double xy = 1.0 - delta2;
                double val = 2.0 * (1.0 - randu) + 2.0 * (randu - 0.5) * pow(xy, (moead->eta_mutation + 1.0));
                deltaq = 1.0 - pow(val, mut_pow);
            }
            // Update gene
            double mutated = x + deltaq * (ub - lb);
            // Ensure within [lb, ub]
            if (mutated < lb) mutated = lb;
            if (mutated > ub) mutated = ub;
            individual[j] = mutated;
        }
    }
}


// pmoo wrappers

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
inline void pmoo_moead_t_hs_hybrid_initialize(void *self, void *gpu_self) {
    moead_t_hs_hybrid_initialize((struct moead_t_hs_hybrid<POP_SIZE, D_DIM, F_DIM>*)self);
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
inline void pmoo_moead_t_hs_hybrid_load_parameters(void *self, Parameters* parameters) {
    struct moead_t_hs_hybrid<POP_SIZE, D_DIM, F_DIM> *moead = (struct moead_t_hs_hybrid<POP_SIZE, D_DIM, F_DIM>*)self;
    // make sure that metadata is set
    moead->pop_size = POP_SIZE;
    moead->d_dim = D_DIM;
    moead->f_dim = F_DIM;

    // initialize algorithm settings
    if (parameters) {
        moead->gen = parameters->gen;
        moead->p_crossover = parameters->p_crossover;
        moead->eta_crossover = parameters->eta_crossover;
        moead->p_mutation = parameters->p_mutation;
        moead->eta_mutation = parameters->eta_mutation;
    } else {
        moead->gen = 100u;
        moead->p_crossover = 0.95;
        moead->eta_crossover = 10.0;
        moead->p_mutation = 0.01;
        moead->eta_mutation = 50.0;
    }
    // set number of threads to use
    moead->num_threads = parameters->cpu_threads;
    unsigned threads_per_block;
    if (parameters) {
      threads_per_block = parameters->gpu_block_size>0u
                        ? parameters->gpu_block_size
                        : 256u;
    } else {
      threads_per_block = 256u;
    }

    int opt_bs = 0, min_grid = 0;
    cudaOccupancyMaxPotentialBlockSize(
    &min_grid,
    &opt_bs,
    (void*)build_neighborhood_hybrid_kernel<POP_SIZE,D_DIM,F_DIM>,
    /*dynamic_smem=*/0,
    threads_per_block
    );

    unsigned bs = std::min<unsigned>(threads_per_block, opt_bs);
    moead->THREADS.x = bs;
    moead->BLOCKS.x  = (POP_SIZE + bs - 1)/bs;

    //moead->THREADS.x = threads_per_block;
    moead->THREADS.y = 1u;
    moead->THREADS.z = 1u;
    //moead->BLOCKS.x  = (POP_SIZE + threads_per_block - 1u) / threads_per_block;
    moead->BLOCKS.y  = 1u;
    moead->BLOCKS.z  = 1u;
    
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
inline void pmoo_moead_t_hs_hybrid_load_problem(void *self, Problem* problem) {
    struct moead_t_hs_hybrid<POP_SIZE, D_DIM, F_DIM> *moead = (struct moead_t_hs_hybrid<POP_SIZE, D_DIM, F_DIM>*)self;
    moead->d_dim = problem->d_dim;
    moead->f_dim = problem->f_dim;
    moead->i_dim = problem->i_dim;
    memcpy(moead->lower_bounds, problem->lower_bounds, sizeof(double) * problem->d_dim);
    memcpy(moead->upper_bounds, problem->upper_bounds, sizeof(double) * problem->d_dim);
    
    moead->fitness = problem->fitness;
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
inline void pmoo_moead_t_hs_hybrid_evolve(void *self, void *gpu_self) {
    moead_t_hs_hybrid_evolve((struct moead_t_hs_hybrid<POP_SIZE, D_DIM, F_DIM>*)self);
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
inline void pmoo_moead_t_hs_hybrid_cleanup(void *self, void *gpu_self) {
    struct moead_t_hs_hybrid<POP_SIZE, D_DIM, F_DIM> *moead = (struct moead_t_hs_hybrid<POP_SIZE, D_DIM, F_DIM>*)self;

    CUDA_CALL(cudaFree(moead->device_neighborhood));
    CUDA_CALL(cudaFree(moead->device_weight_vectors));
    CUDA_CALL(cudaFree(moead->deviceStates));
    
    for (unsigned i = 0; i < POP_SIZE; ++i) omp_destroy_lock(&moead->lock[i]);
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
inline double** pmoo_moead_t_hs_hybrid_decision_variables(void *self) {
    struct moead_t_hs_hybrid<POP_SIZE, D_DIM, F_DIM> *moead = (struct moead_t_hs_hybrid<POP_SIZE, D_DIM, F_DIM>*)self;
    double **population = (double**)malloc(POP_SIZE * sizeof(double*));
    for (unsigned i = 0u; i < POP_SIZE; ++i)
        population[i] = moead->decision_variables[i];
    
    return population;
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
inline double** pmoo_moead_t_hs_hybrid_fitness_values(void *self) {
    struct moead_t_hs_hybrid<POP_SIZE, D_DIM, F_DIM> *moead = (struct moead_t_hs_hybrid<POP_SIZE, D_DIM, F_DIM>*)self;
    double **population = (double**)malloc(POP_SIZE * sizeof(double*));
    for (unsigned i = 0u; i < POP_SIZE; ++i)
        population[i] = moead->fitness_values[i];
    
    return population;
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
Algorithm* pmoo_moead_t_hs_hybrid_generate() {
    Algorithm *algorithm = (Algorithm*)malloc(sizeof(Algorithm));

    algorithm->name = MOEAD_T_HS;
    algorithm->variant = HYBRID;
    algorithm->size = sizeof(struct moead_t_hs_hybrid<POP_SIZE, D_DIM, F_DIM>);
    algorithm->self = malloc(algorithm->size);
    algorithm->gpu_self = NULL;
    algorithm->population_size = POP_SIZE;
    algorithm->decision_variables = pmoo_moead_t_hs_hybrid_decision_variables<POP_SIZE, D_DIM, F_DIM>(algorithm->self);
    algorithm->fitness_values = pmoo_moead_t_hs_hybrid_fitness_values<POP_SIZE, D_DIM, F_DIM>(algorithm->self);
    algorithm->load_problem = pmoo_moead_t_hs_hybrid_load_problem<POP_SIZE, D_DIM, F_DIM>;
    algorithm->load_parameters = pmoo_moead_t_hs_hybrid_load_parameters<POP_SIZE, D_DIM, F_DIM>;
    algorithm->initialize = pmoo_moead_t_hs_hybrid_initialize<POP_SIZE, D_DIM, F_DIM>;
    algorithm->evolve = pmoo_moead_t_hs_hybrid_evolve<POP_SIZE, D_DIM, F_DIM>;
    algorithm->cleanup = pmoo_moead_t_hs_hybrid_cleanup<POP_SIZE, D_DIM, F_DIM>;
    return algorithm;
}

#endif
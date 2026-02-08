#ifndef PMOO_HYBRID_SPEA2_CUH
#define PMOO_HYBRID_SPEA2_CUH

#include "../algorithm.cuh"
#include "../utils.cuh"

#include <random> // mt19937, random_device and distributions
#include <algorithm> // sort

#include <string.h> // memcpy, memset
#include <math.h>
#include <cuda_runtime.h>

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
struct spea2_hybrid_fs {
    double fitness_values[2u * POP_SIZE][F_DIM];
    unsigned dominations[2u * POP_SIZE][2u * POP_SIZE];
    unsigned domination_count[2u * POP_SIZE];
    double distance[2u * POP_SIZE][2u * POP_SIZE];
    double fitness_score[2u * POP_SIZE];
};

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
struct spea2_hybrid {
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
    // variables used for calculations
    double decision_variables[2u * POP_SIZE][D_DIM];                            // Population decision variables
    double fitness_values[2u * POP_SIZE][F_DIM];                                // Population fitness values
    double selected_decision_variables[POP_SIZE][D_DIM];                        // Used for selecting best N individuals
    double selected_fitness_values[POP_SIZE][F_DIM];                            // Used for selecting best N individuals
    unsigned shuffle[2u * POP_SIZE];                                            // Used to store random order for tournament selection
    double distance[2u * POP_SIZE][2u * POP_SIZE];                              // Used to store the distances between each individuals
    double fitness_score[2u * POP_SIZE];                                        // Used to store the sum of strength and density for each individual
    struct array_with_size<2u * POP_SIZE> fitness_score_order;                  // Used to store indices of individuals sorted by fitness score
    unsigned distance_order[2u * POP_SIZE][2u * POP_SIZE];                      // Used to store order of sorted distances for each individual
    struct array_with_size<2u * POP_SIZE> selected_fitness_score_order;         // Used to map to fitness_score_order in an improved environmental selection
    // rng
    std::mt19937 random[2u * POP_SIZE];
    // cuda
    struct spea2_hybrid_fs<POP_SIZE, D_DIM, F_DIM> *cuda_fitness_score;
    dim3 THREADS_IN_BLOCK_1D;
    dim3 BLOCKS_IN_GRID_1D;
    dim3 THREADS_IN_BLOCK_2D;
    dim3 BLOCKS_IN_GRID_2D;
    
    unsigned num_threads;
};

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
// Problem metadata should be loaded first before initializing algorithm
void spea2_hybrid_initialize(struct spea2_hybrid<POP_SIZE, D_DIM, F_DIM> *spea2) {
    omp_set_num_threads(spea2->num_threads);
    // initialize random number generators
    for (unsigned i = 0u; i < 2u * spea2->pop_size; ++i)
        spea2->random[i].seed(std::random_device{}());

    // allocate memory on gpu
    CUDA_CALL(cudaMalloc(&spea2->cuda_fitness_score, sizeof(struct spea2_hybrid_fs<POP_SIZE, D_DIM, F_DIM>)));

    // initialize shuffle arrays
    for (unsigned i = 0u; i < spea2->pop_size; ++i) spea2->shuffle[i] = i;
    for (unsigned i = 0u; i < spea2->pop_size; ++i) spea2->shuffle[spea2->pop_size + i] = i;

    // initialize continous decision variables
    for (unsigned dim = 0u; dim < spea2->d_dim - spea2->i_dim; ++dim) {
        std::uniform_real_distribution<double> uniform(spea2->lower_bounds[dim], spea2->upper_bounds[dim]);
        for (unsigned i = 0u; i < spea2->pop_size; ++i) {
            spea2->decision_variables[i][dim] = uniform(spea2->random[i]);
        }
    }

    // initialize integer decision variables
    for (unsigned dim = spea2->d_dim - spea2->i_dim; dim < spea2->d_dim; ++dim) {
        std::uniform_real_distribution<double> uniform(spea2->lower_bounds[dim], spea2->upper_bounds[dim]);
        for (unsigned i = 0u; i < spea2->pop_size; ++i) {
            spea2->decision_variables[i][dim] = uniform(spea2->random[i]);
        }
    }

    // calculate fitness for initialized population
    for (unsigned i = 0u; i < spea2->pop_size; ++i) {
        spea2->fitness(spea2->decision_variables[i], spea2->fitness_values[i], spea2->d_dim, spea2->f_dim);
    }
};

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
double spea2_hybrid_distance(struct spea2_hybrid<POP_SIZE, D_DIM, F_DIM> *spea2, const unsigned individual1, const unsigned individual2) {
    double distance = 0.0;
    for (unsigned dim = 0; dim < spea2->f_dim; ++dim)
        distance += pow(spea2->fitness_values[individual1][dim] - spea2->fitness_values[individual2][dim], 2.0);
    return sqrt(distance);
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
__global__ void spea2_hybrid_pareto_dominance(struct spea2_hybrid_fs<POP_SIZE, D_DIM, F_DIM> *spea2) {
    const unsigned global_id_x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned global_id_y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned dominating = 0u;
    if (pareto_dominance(spea2->fitness_values[global_id_x], spea2->fitness_values[global_id_y], F_DIM)) dominating = 1u;
    __syncthreads();
    spea2->dominations[global_id_x][global_id_y] = dominating;
    if (dominating) atomicAdd(&spea2->domination_count[global_id_x], 1u);
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
__global__ void spea2_hybrid_strength_score(struct spea2_hybrid_fs<POP_SIZE, D_DIM, F_DIM> *spea2) {
    const unsigned global_id_x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned global_id_y = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned dominating = spea2->dominations[global_id_x][global_id_y];
    if (dominating) atomicAdd(&spea2->fitness_score[global_id_y], (double)spea2->domination_count[global_id_x]);
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
__global__ void spea2_hybrid_calculate_distances(struct spea2_hybrid_fs<POP_SIZE, D_DIM, F_DIM> *spea2) {
    const unsigned global_id_x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned global_id_y = blockIdx.y * blockDim.y + threadIdx.y;

    double distance;
    if (global_id_x == global_id_y) distance = INFINITY;
    else {
        distance = 0.0;
        for (unsigned dim = 0; dim < F_DIM; ++dim)
            distance += pow(spea2->fitness_values[global_id_x][dim] - spea2->fitness_values[global_id_y][dim], 2.0);
        distance = sqrt(distance);
    }
    __syncthreads();
    spea2->distance[global_id_x][global_id_y] = distance;
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
__global__ void spea2_hybrid_bitonic_sort_distances(struct spea2_hybrid_fs<POP_SIZE, D_DIM, F_DIM> *spea2, int j, int k) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned ij = i ^ j;

    if (ij > i) {
        if ((i & k) == 0) {
            if (spea2->distance[row][i] > spea2->distance[row][ij]) {
                double temp = spea2->distance[row][i];
                spea2->distance[row][i] = spea2->distance[row][ij];
                spea2->distance[row][ij] = temp;
            }
        } else {
            if (spea2->distance[row][i] < spea2->distance[row][ij]) {
                double temp = spea2->distance[row][i];
                spea2->distance[row][i] = spea2->distance[row][ij];
                spea2->distance[row][ij] = temp;
            }
        }
    }
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
__global__ void spea2_hybrid_density_score(struct spea2_hybrid_fs<POP_SIZE, D_DIM, F_DIM> *spea2, const unsigned k) {
    const unsigned global_id = blockIdx.x * blockDim.x + threadIdx.x;

    double score = 1.0 / (spea2->distance[global_id][k] + 2.0);
    __syncthreads();
    spea2->fitness_score[global_id] += score;

}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
// calculates fitness from strength and density
void spea2_hybrid_fitness_score(struct spea2_hybrid<POP_SIZE, D_DIM, F_DIM> *spea2, const unsigned N) {
    const unsigned k = floor(sqrt(N)) - 1u; // -1 because its indexed from 0

    spea2->BLOCKS_IN_GRID_2D.x = N / spea2->THREADS_IN_BLOCK_2D.x;
    spea2->BLOCKS_IN_GRID_2D.y = N / spea2->THREADS_IN_BLOCK_2D.y;
    spea2->BLOCKS_IN_GRID_2D.z = 1u;

    spea2->BLOCKS_IN_GRID_1D.x = N / spea2->THREADS_IN_BLOCK_1D.x;
    spea2->BLOCKS_IN_GRID_1D.y = 1u;
    spea2->BLOCKS_IN_GRID_1D.z = 1u;

    CUDA_CALL(cudaMemset(spea2->cuda_fitness_score->domination_count, 0u, N * sizeof(unsigned)));
    CUDA_CALL(cudaMemset(spea2->cuda_fitness_score->fitness_score, 0.0, N * sizeof(double)));
    CUDA_CALL(cudaMemcpy(spea2->cuda_fitness_score->fitness_values, spea2->fitness_values, N * spea2->f_dim * sizeof(double), cudaMemcpyHostToDevice));
    spea2_hybrid_pareto_dominance<<<spea2->BLOCKS_IN_GRID_2D, spea2->THREADS_IN_BLOCK_2D>>>(spea2->cuda_fitness_score);
    spea2_hybrid_strength_score<<<spea2->BLOCKS_IN_GRID_2D, spea2->THREADS_IN_BLOCK_2D>>>(spea2->cuda_fitness_score);
    spea2_hybrid_calculate_distances<<<spea2->BLOCKS_IN_GRID_2D, spea2->THREADS_IN_BLOCK_2D>>>(spea2->cuda_fitness_score);
    for (unsigned int k = 2; k <= N; k <<= 1)
        for (unsigned int j = k >> 1; j > 0; j >>= 1)
            spea2_hybrid_bitonic_sort_distances<<<spea2->BLOCKS_IN_GRID_2D, spea2->THREADS_IN_BLOCK_2D>>>(spea2->cuda_fitness_score, j, k);
    spea2_hybrid_density_score<<<spea2->BLOCKS_IN_GRID_1D, spea2->THREADS_IN_BLOCK_1D>>>(spea2->cuda_fitness_score, k);
    CUDA_CALL(cudaMemcpy(spea2->fitness_score, spea2->cuda_fitness_score->fitness_score, N * sizeof(double), cudaMemcpyDeviceToHost));
};

// Implementation of std::shuffle
template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
void spea2_hybrid_shuffle_selection(struct spea2_hybrid<POP_SIZE, D_DIM, F_DIM> *spea2) {
    std::uniform_int_distribution<unsigned> dist(0u, spea2->pop_size - 1u);
    unsigned tmp, rnd;

    for (unsigned i = 0u; i < spea2->pop_size; ++i) {
        rnd = dist(spea2->random[i]);
        tmp = spea2->shuffle[i];

        spea2->shuffle[i] = spea2->shuffle[rnd];
        spea2->shuffle[rnd] = tmp;
    }

    for (unsigned i = 0u; i < spea2->pop_size; ++i) {
        rnd = dist(spea2->random[i]);
        tmp = spea2->shuffle[spea2->pop_size + i];

        spea2->shuffle[spea2->pop_size + i] = spea2->shuffle[spea2->pop_size + rnd];
        spea2->shuffle[spea2->pop_size + rnd] = tmp;
    }
};

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
unsigned spea2_hybrid_tournament_selection(struct spea2_hybrid<POP_SIZE, D_DIM, F_DIM> *spea2, const unsigned individual1, const unsigned individual2) {
    // select individual with lower fitness score
    if (spea2->fitness_score[individual1] < spea2->fitness_score[individual2]) return individual1;
    if (spea2->fitness_score[individual1] > spea2->fitness_score[individual2]) return individual2;
    // select randomly
    std::uniform_real_distribution<double> uniform(0., 1.);
    return ((uniform(spea2->random[individual1]) < 0.5) ? individual1 : individual2);
};

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
void spea2_hybrid_simulated_binary_crossover(struct spea2_hybrid<POP_SIZE, D_DIM, F_DIM> *spea2, const unsigned parent1_idx, const unsigned parent2_idx, const unsigned child1_idx, const unsigned child2_idx) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    
    // initialize children decision variables same as their parents
    memcpy(spea2->decision_variables[child1_idx], spea2->decision_variables[parent1_idx], spea2->d_dim * sizeof(double));
    memcpy(spea2->decision_variables[child2_idx], spea2->decision_variables[parent2_idx], spea2->d_dim * sizeof(double));

    if (uniform(spea2->random[parent1_idx]) < spea2->p_crossover) { // crossover might not happen with crossover probability
        // SBX - only for continous decision variables
        for (unsigned dim = 0u; dim < spea2->d_dim - spea2->i_dim; ++dim) { // for each decision variable in parents
            if (uniform(spea2->random[parent1_idx]) < 0.5 && fabs(spea2->decision_variables[parent1_idx][dim] - spea2->decision_variables[parent2_idx][dim]) > TOLERANCE_EPSILON) { // each variable has 50% chance to be crossovered
                double y1, y2, beta, betaq, c1, c2, rnd;

                // find smaller variable in parents
                if (spea2->decision_variables[parent1_idx][dim] < spea2->decision_variables[parent2_idx][dim]) {
                    y1 = spea2->decision_variables[parent1_idx][dim];
                    y2 = spea2->decision_variables[parent2_idx][dim];
                } else {
                    y1 = spea2->decision_variables[parent2_idx][dim];
                    y2 = spea2->decision_variables[parent1_idx][dim];
                }

                rnd = uniform(spea2->random[parent1_idx]);

                beta = 1.0 + (2.0 * (y1 - spea2->lower_bounds[dim]) / (y2 - y1));
                betaq = sbx_betaq(beta, spea2->eta_crossover, rnd);
                c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1));

                beta = 1.0 + (2.0 * (spea2->upper_bounds[dim] - y2) / (y2 - y1));
                betaq = sbx_betaq(beta, spea2->eta_crossover, rnd);
                c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1));

                // make sure that new variable values fit in problem bounds
                if (c1 < spea2->lower_bounds[dim]) c1 = spea2->lower_bounds[dim];
                if (c2 < spea2->lower_bounds[dim]) c2 = spea2->lower_bounds[dim];
                if (c1 > spea2->upper_bounds[dim]) c1 = spea2->upper_bounds[dim];
                if (c2 > spea2->upper_bounds[dim]) c2 = spea2->upper_bounds[dim];

                // assign new variable values randomly to children
                if (uniform(spea2->random[parent1_idx]) < 0.5) {
                    spea2->decision_variables[child1_idx][dim] = c1;
                    spea2->decision_variables[child2_idx][dim] = c2;
                } else {
                    spea2->decision_variables[child1_idx][dim] = c2;
                    spea2->decision_variables[child2_idx][dim] = c1;
                }
            }
        }

        // Two-points crossover - only for integer decision variables
        if (spea2->i_dim > 0u) {
            std::uniform_int_distribution<unsigned> i_dim_dist(spea2->d_dim - spea2->i_dim, spea2->d_dim - 1u);
            unsigned cut1 = i_dim_dist(spea2->random[parent1_idx]);
            unsigned cut2 = i_dim_dist(spea2->random[parent1_idx]);
            if (cut1 < cut2) {
                memcpy(&spea2->decision_variables[child1_idx][cut1], &spea2->decision_variables[parent2_idx][cut1], (cut2 - cut1 + 1u) * sizeof(double));
                memcpy(&spea2->decision_variables[child2_idx][cut1], &spea2->decision_variables[parent1_idx][cut1], (cut2 - cut1 + 1u) * sizeof(double));
            } else {
                memcpy(&spea2->decision_variables[child1_idx][cut2], &spea2->decision_variables[parent2_idx][cut2], (cut1 - cut2 + 1u) * sizeof(double));
                memcpy(&spea2->decision_variables[child2_idx][cut2], &spea2->decision_variables[parent1_idx][cut2], (cut1 - cut2 + 1u) * sizeof(double));
            }
        }
    }
};

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
void spea2_hybrid_polynomial_mutation(struct spea2_hybrid<POP_SIZE, D_DIM, F_DIM> *spea2, const unsigned individual) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    // polynomial mutation - only for continous decision variables
    double deltad, delta1, delta2, rnd, mutation_exp, xy, val, deltaq;
    for (unsigned dim = 0u; dim < spea2->d_dim - spea2->i_dim; ++dim) {
        if (uniform(spea2->random[individual]) < spea2->p_mutation) { // mutation might not occur
            deltad = spea2->upper_bounds[dim] - spea2->lower_bounds[dim];
            delta1 = (spea2->decision_variables[individual][dim] - spea2->lower_bounds[dim]) / (deltad);
            delta2 = (spea2->upper_bounds[dim] - spea2->decision_variables[individual][dim]) / (deltad);
            mutation_exp = 1. / (spea2->eta_mutation + 1.);

            rnd = uniform(spea2->random[individual]);
            if (rnd < 0.5) {
                xy = 1.0 - delta1;
                val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (pow(xy, (spea2->eta_mutation + 1.0)));
                deltaq = pow(val, mutation_exp) - 1.0;
            } else {
                xy = 1.0 - delta2;
                val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (pow(xy, (spea2->eta_mutation + 1.0)));
                deltaq = 1.0 - (pow(val, mutation_exp));
            }
            spea2->decision_variables[individual][dim] += deltaq * deltad;

            // make sure that new variable value fits in problem bounds
            if (spea2->decision_variables[individual][dim] < spea2->lower_bounds[dim]) spea2->decision_variables[individual][dim] = spea2->lower_bounds[dim];
            if (spea2->decision_variables[individual][dim] > spea2->upper_bounds[dim]) spea2->decision_variables[individual][dim] = spea2->upper_bounds[dim];
        }
    }

    // integer mutation - only for integer decision variables
    for (unsigned dim = spea2->d_dim - spea2->i_dim; dim < spea2->d_dim; ++dim) {
        std::uniform_int_distribution<int> dim_dist((int)spea2->lower_bounds[dim], (int)spea2->upper_bounds[dim]);
        spea2->decision_variables[individual][dim] = (double)dim_dist(spea2->random[individual]);
    }
};

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
void spea2_hybrid_environmental_selection(struct spea2_hybrid<POP_SIZE, D_DIM, F_DIM> *spea2) {
    const unsigned N = 2u * spea2->pop_size;

    // calculate fitness score same as previously but with 2N population
    spea2_hybrid_fitness_score(spea2, N);

    // sort individuals based on fitness score in an ascending order
    spea2->fitness_score_order.size = N;
    for (unsigned i = 0; i < N; ++i) spea2->fitness_score_order.elements[i] = i;
    std::sort(spea2->fitness_score_order.elements, spea2->fitness_score_order.elements + N, [spea2](unsigned idx1, unsigned idx2) {
        return spea2->fitness_score[idx1] < spea2->fitness_score[idx2];
    });

    // find how many individuals have fitness score below 1 (via binary search)
    unsigned l = 0u, r = N - 1u, m;
    while (l <= r) {
        m = l + (r - l) / 2u;
        if (spea2->fitness_score[spea2->fitness_score_order.elements[m]] < 1.0) l = m + 1u;
        else r = m - 1u;
    }
    // l contains index (in fitness_score_order) of first individual with fitness score over 1

    if (l <= spea2->pop_size) { // if there are exactly pop_size individuals with fitness score < 1 or less
        spea2->fitness_score_order.size = spea2->pop_size; // set pop_size individuals as selected

        // copy over selected individuals decision and fitness values to a buffer
        for (unsigned i = 0u; i < spea2->fitness_score_order.size; ++i)
            memcpy(spea2->selected_decision_variables[i], spea2->decision_variables[spea2->fitness_score_order.elements[i]], sizeof(double) * spea2->d_dim);
        for (unsigned i = 0u; i < spea2->fitness_score_order.size; ++i)
            memcpy(spea2->selected_fitness_values[i], spea2->fitness_values[spea2->fitness_score_order.elements[i]], sizeof(double) * spea2->f_dim);
    } else { // if there are more individuals with fitness score < 1 than there are available spaces
        spea2->fitness_score_order.size = l; // we need to truncate selected population from l to pop_size

        // create mapping to fitness_score_order in selected_fitness_score_order
        spea2->selected_fitness_score_order.size = spea2->fitness_score_order.size;
        for (unsigned i = 0; i < spea2->selected_fitness_score_order.size; ++i) spea2->selected_fitness_score_order.elements[i] = i;

        // calculate distances between each individual IN FITNESS_SCORE_ORDER only (same as in fitness_score procedure)
        for (unsigned i = 0u; i < spea2->fitness_score_order.size; ++i) spea2->distance[i][i] = INFINITY; // infinity on diagonal

        #pragma omp parallel for
        for (OMP_FOR_INDEX idx = 0; idx < (spea2->fitness_score_order.size * (spea2->fitness_score_order.size - 1)) / 2; ++idx) {
            OMP_FOR_INDEX i = 0;
            OMP_FOR_INDEX temp = idx;
            while (temp >= spea2->fitness_score_order.size - i - 1) {
                temp -= spea2->fitness_score_order.size - i - 1;
                ++i;
            }
            OMP_FOR_INDEX j = i + 1 + temp;

            spea2->distance[i][j] = spea2_hybrid_distance(spea2, spea2->fitness_score_order.elements[i], spea2->fitness_score_order.elements[j]);
            spea2->distance[j][i] = spea2->distance[i][j]; // the matrix is symmetric
        }

        // create order of distances in an ascending order for each individual
        for (unsigned i = 0u; i < spea2->fitness_score_order.size; ++i)
            for (unsigned j = 0u; j < spea2->fitness_score_order.size; ++j)
                spea2->distance_order[i][j] = j;

        #pragma omp parallel for
        for (OMP_FOR_INDEX i = 0u; i < spea2->fitness_score_order.size; ++i) {
            std::sort(spea2->distance_order[i], spea2->distance_order[i] + spea2->fitness_score_order.size, [spea2, i](unsigned idx1, unsigned idx2) {
                return spea2->distance[i][idx1] < spea2->distance[i][idx2];
            });
        }

        // truncate individuals with lowest distances
        while (spea2->selected_fitness_score_order.size > spea2->pop_size) {
            // find individual with lowest distance to be removed
            unsigned truncate = 0u;
            for (unsigned i = 1u; i < spea2->selected_fitness_score_order.size; ++i) {
                unsigned truncated = spea2->selected_fitness_score_order.elements[truncate];
                unsigned tested = spea2->selected_fitness_score_order.elements[i];
                int truncated_dim = -1, tested_dim = -1; // start from -1 for the do while loop to work

                if (spea2->distance[truncated][tested] < TOLERANCE_EPSILON || spea2->distance[tested][truncated] < TOLERANCE_EPSILON) continue; // if individuals are equal (distance = 0.0) just skip it doesnt matter which

                do {
                    ++truncated_dim;
                    ++tested_dim;

                    while (spea2->distance[truncated][spea2->distance_order[truncated][truncated_dim]] == INFINITY) ++truncated_dim; // find first distance that isnt INFINITY (didnt come from already truncated individual)
                    while (spea2->distance[tested][spea2->distance_order[tested][tested_dim]] == INFINITY) ++tested_dim; // find first distance that isnt INFINITY (didnt come from already truncated individual)

                    if (spea2->distance[tested][spea2->distance_order[tested][tested_dim]] < spea2->distance[truncated][spea2->distance_order[truncated][truncated_dim]]) truncate = i; // select the one with lower distance for removal
                } while (spea2->distance[tested][spea2->distance_order[tested][tested_dim]] == spea2->distance[truncated][spea2->distance_order[truncated][truncated_dim]]);
            }

            // set INFINITY for distances in truncated column (so they cannot be included in next iteration)
            for (unsigned i = 0u; i < spea2->fitness_score_order.size; ++i) spea2->distance[i][spea2->selected_fitness_score_order.elements[truncate]] = INFINITY;

            // truncate the individual with lowest distance
            --spea2->selected_fitness_score_order.size;
            spea2->selected_fitness_score_order.elements[truncate] = spea2->selected_fitness_score_order.elements[spea2->selected_fitness_score_order.size];
        }

        // copy over selected individuals decision and fitness values to a buffer
        for (unsigned i = 0u; i < spea2->selected_fitness_score_order.size; ++i)
            memcpy(spea2->selected_decision_variables[i], spea2->decision_variables[spea2->fitness_score_order.elements[spea2->selected_fitness_score_order.elements[i]]], sizeof(double) * spea2->d_dim);
        for (unsigned i = 0u; i < spea2->selected_fitness_score_order.size; ++i)
            memcpy(spea2->selected_fitness_values[i], spea2->fitness_values[spea2->fitness_score_order.elements[spea2->selected_fitness_score_order.elements[i]]], sizeof(double) * spea2->f_dim);
    }
};

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
void spea2_hybrid_evolve(struct spea2_hybrid<POP_SIZE, D_DIM, F_DIM> *spea2) {
    for (unsigned gen = 1u; gen <= spea2->gen; ++gen) {
        // prepare indexes for tournament selection
        spea2_hybrid_shuffle_selection(spea2);

        // calculate fitness score
        spea2_hybrid_fitness_score(spea2, spea2->pop_size);

        // selection, crossover and mutation (generate N children so that the population will reach 2N size)
        #pragma omp parallel for
        for (OMP_FOR_INDEX i = 0u; i < spea2->pop_size / 2u; ++i) {
            // calculate index to select parents from shuffle and store new children
            unsigned selection_idx = i * 4u;
            unsigned child1_idx = spea2->pop_size + i * 2u;
            unsigned child2_idx = child1_idx + 1u;
            // select two parents through tournament selection from 4 random individuals
            unsigned parent1_idx = spea2_hybrid_tournament_selection(spea2, spea2->shuffle[selection_idx], spea2->shuffle[selection_idx + 1]);
            unsigned parent2_idx = spea2_hybrid_tournament_selection(spea2, spea2->shuffle[selection_idx + 2], spea2->shuffle[selection_idx + 3]);
            // generate two new children by crossover of selected parents
            spea2_hybrid_simulated_binary_crossover(spea2, parent1_idx, parent2_idx, child1_idx, child2_idx);
            // run mutation on both of new children
            spea2_hybrid_polynomial_mutation(spea2, child1_idx);
            spea2_hybrid_polynomial_mutation(spea2, child2_idx);
            // calculate fitness vectors for new children
            spea2->fitness(spea2->decision_variables[child1_idx], spea2->fitness_values[child1_idx], spea2->d_dim, spea2->f_dim);
            spea2->fitness(spea2->decision_variables[child2_idx], spea2->fitness_values[child2_idx], spea2->d_dim, spea2->f_dim);
        }

        // select best N individuals from 2*N population
        spea2_hybrid_environmental_selection(spea2);

        // move selected individuals into the new population
        memcpy(spea2->decision_variables, spea2->selected_decision_variables, sizeof(double) * spea2->d_dim * spea2->pop_size);
        memcpy(spea2->fitness_values, spea2->selected_fitness_values, sizeof(double) * spea2->f_dim * spea2->pop_size);
    }
};

// pmoo wrappers

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
inline void pmoo_spea2_hybrid_load_problem(void *self, Problem* problem) {
    struct spea2_hybrid<POP_SIZE, D_DIM, F_DIM> *spea2 = (struct spea2_hybrid<POP_SIZE, D_DIM, F_DIM>*)self;
    spea2->d_dim = problem->d_dim;
    spea2->f_dim = problem->f_dim;
    spea2->i_dim = problem->i_dim;
    memcpy(spea2->lower_bounds, problem->lower_bounds, sizeof(double) * problem->d_dim);
    memcpy(spea2->upper_bounds, problem->upper_bounds, sizeof(double) * problem->d_dim);
    spea2->fitness = problem->fitness;
};

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
inline void pmoo_spea2_hybrid_load_parameters(void *self, Parameters* parameters) {
    struct spea2_hybrid<POP_SIZE, D_DIM, F_DIM> *spea2 = (struct spea2_hybrid<POP_SIZE, D_DIM, F_DIM>*)self;

    // make sure that metadata is set
    spea2->pop_size = POP_SIZE;
    spea2->d_dim = D_DIM;
    spea2->f_dim = F_DIM;

    // initialize algorithm settings
    if (parameters) {
        spea2->gen = parameters->gen;
        spea2->p_crossover = parameters->p_crossover;
        spea2->eta_crossover = parameters->eta_crossover;
        spea2->p_mutation = parameters->p_mutation;
        spea2->eta_mutation = parameters->eta_mutation;
    } else {
        spea2->gen = 100u;
        spea2->p_crossover = 0.95;
        spea2->eta_crossover = 10.0;
        spea2->p_mutation = 0.01;
        spea2->eta_mutation = 50.0;
    }

    // set cuda block sizes to call kernels
    spea2->THREADS_IN_BLOCK_2D.x = ceil(sqrt(parameters->gpu_block_size));
    spea2->THREADS_IN_BLOCK_2D.y = ceil(sqrt(parameters->gpu_block_size));
    spea2->THREADS_IN_BLOCK_2D.z = 1u;

    spea2->THREADS_IN_BLOCK_1D.x = parameters->gpu_block_size;
    spea2->THREADS_IN_BLOCK_1D.y = 1u;
    spea2->THREADS_IN_BLOCK_1D.z = 1u;

    
    // set number of threads to use
    spea2->num_threads = parameters->cpu_threads;
};

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
inline void pmoo_spea2_hybrid_initialize(void *self, void *gpu_self) {
    spea2_hybrid_initialize((struct spea2_hybrid<POP_SIZE, D_DIM, F_DIM>*)self);
};

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
inline void pmoo_spea2_hybrid_evolve(void *self, void *gpu_self) {
    spea2_hybrid_evolve((struct spea2_hybrid<POP_SIZE, D_DIM, F_DIM>*)self);
};

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
inline void pmoo_spea2_hybrid_cleanup(void *self, void *gpu_self) {
    struct spea2_hybrid<POP_SIZE, D_DIM, F_DIM> *spea2 = (struct spea2_hybrid<POP_SIZE, D_DIM, F_DIM>*)self;

    cudaFree(spea2->cuda_fitness_score);
};

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
inline double** pmoo_spea2_hybrid_decision_variables(void *self) {
    struct spea2_hybrid<POP_SIZE, D_DIM, F_DIM> *spea2 = (struct spea2_hybrid<POP_SIZE, D_DIM, F_DIM>*)self;

    double **population = (double**)malloc(POP_SIZE * sizeof(double*));
    for (unsigned i = 0u; i < POP_SIZE; ++i)
        population[i] = spea2->decision_variables[i];
    
    return population;
};

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
inline double** pmoo_spea2_hybrid_fitness_values(void *self) {
    struct spea2_hybrid<POP_SIZE, D_DIM, F_DIM> *spea2 = (struct spea2_hybrid<POP_SIZE, D_DIM, F_DIM>*)self;

    double **population = (double**)malloc(POP_SIZE * sizeof(double*));
    for (unsigned i = 0u; i < POP_SIZE; ++i)
        population[i] = spea2->fitness_values[i];
    
    return population;
};

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
Algorithm* pmoo_spea2_hybrid_generate() {
    Algorithm *algorithm = (Algorithm*)malloc(sizeof(Algorithm));

    algorithm->name = SPEA2;
    algorithm->variant = HYBRID;
    algorithm->size = sizeof(struct spea2_hybrid<POP_SIZE, D_DIM, F_DIM>);
    // NOTE: cudaMallocHost greatly reduces cudaMemcpy times, but will refuse to allocate large amounts of memory
    CUDA_CALL(cudaMallocHost(&algorithm->self, algorithm->size));
    algorithm->gpu_self = NULL;
    algorithm->population_size = POP_SIZE;
    algorithm->decision_variables = pmoo_spea2_hybrid_decision_variables<POP_SIZE, D_DIM, F_DIM>(algorithm->self);
    algorithm->fitness_values = pmoo_spea2_hybrid_fitness_values<POP_SIZE, D_DIM, F_DIM>(algorithm->self);
    algorithm->load_problem = pmoo_spea2_hybrid_load_problem<POP_SIZE, D_DIM, F_DIM>;
    algorithm->load_parameters = pmoo_spea2_hybrid_load_parameters<POP_SIZE, D_DIM, F_DIM>;
    algorithm->initialize = pmoo_spea2_hybrid_initialize<POP_SIZE, D_DIM, F_DIM>;
    algorithm->evolve = pmoo_spea2_hybrid_evolve<POP_SIZE, D_DIM, F_DIM>;
    algorithm->cleanup = pmoo_spea2_hybrid_cleanup<POP_SIZE, D_DIM, F_DIM>;

    return algorithm;
};

#endif
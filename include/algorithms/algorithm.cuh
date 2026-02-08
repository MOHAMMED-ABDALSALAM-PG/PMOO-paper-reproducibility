#ifndef PMOO_ALGORITHM_CUH
#define PMOO_ALGORITHM_CUH

#include "problems/problem.cuh"

enum AlgorithmEnum {
    NSGA2 = 0,
    SPEA2,
    MOEAD_T_D,
    MOEAD_T_DD,
    MOEAD_T_HS,
    ALGORITHMS_COUNT
};

const AlgorithmEnum ALGORITHMS[ALGORITHMS_COUNT] = { NSGA2, SPEA2, MOEAD_T_D, MOEAD_T_DD, MOEAD_T_HS, };
const char* const ALGORITHMS_NAMES[ALGORITHMS_COUNT] = { "NSGA2", "SPEA2", "MOEAD_T_D", "MOEAD_T_DD", "MOEAD_T_HS" };

enum VariantEnum {
    SERIAL = 0,
    CPU,
    GPU,
    HYBRID,
    VARIANTS_COUNT
};

const VariantEnum VARIANTS[VARIANTS_COUNT] = { SERIAL, CPU, GPU, HYBRID };
const char* const VARIANTS_NAMES[VARIANTS_COUNT] = { "SERIAL", "CPU", "GPU", "HYBRID" };

typedef struct algorithm_parameters {
    unsigned gen;                   // Number of generations to run in a single evolve call
    double p_crossover;             // Crossover probability
    double eta_crossover;           // Crossover index (determines how similar are children to parents)
    double p_mutation;              // Mutation probability
    double eta_mutation;            // Mutation index (determines how big of a change the mutated value will be)
    // Algorithm independent parameters
    unsigned cpu_threads;           // Number of OMP threads
    unsigned gpu_block_size;        // CUDA kernel block size
    unsigned long long curand_seed; // CUDA curand seed
} Parameters;

Parameters* default_parameters();

/*
 * NOTE:
 * We are storing both cpu and gpu pointers
 * and passing them to each function to make it easier
 * (even though its probably not an ideal architecture).
 * 
 * Functions should be called in particular order:
 * 1. LoadProblem - set dimensions, bounds and fitness function from Problem object
 * 2. LoadParameters - set algorithm parameters from Parameters object
 * 3. Initialize - initialize algorithm state
 * 4. Evolve - run algorithm with previously set values
 * 5. Cleanup - let algorithm cleanup some stuff after initialization (like free memory on the gpu)
 * 
 * For gpu variants, you have to manage gpu memory yourself:
 * 1) Call cudaMemcpy after LoadProblem and LoadParameters to copy algorithm from host to device
 * 2) Run Initialize and Evolve by passing both cpu and gpu pointers:
 *  - we need both cpu and gpu pointers because we need population size to declare amount of threads to call kernel with gpu pointer
 *  - remember to run cudaDeviceSynchronize to wait for kernel completion
 * 3) After Evolve call cudaMemcpy from device to host so you can access results and calculate metrics
 */
typedef void (*LoadProblem)(void *self, Problem* problem);
typedef void (*LoadParameters)(void *self, Parameters* parameters);
typedef void (*Initialize)(void *self, void *gpu_self);
typedef void (*Evolve)(void *self, void *gpu_self);
typedef void (*Cleanup)(void *self, void *gpu_self);

typedef struct algorithm {
    AlgorithmEnum name;
    VariantEnum variant;
    unsigned long size;
    void *self;
    void *gpu_self; // NOTE: Used for gpu implementations that are meant to be used with the entire algorithm struct on the gpu memory
    unsigned population_size;
    double **decision_variables;
    double **fitness_values;
    LoadProblem load_problem;
    LoadParameters load_parameters;
    Initialize initialize;
    Evolve evolve;
    Cleanup cleanup;
} Algorithm;

inline unsigned total_algorithm_count() {
    return ALGORITHMS_COUNT * VARIANTS_COUNT;
}

void free_algorithm(Algorithm *algorithm);

#endif
#include "algorithms/algorithm.cuh"

#include <cuda_runtime.h>

#include <stdlib.h>
#include <random>

Parameters* default_parameters() {
    Parameters* parameters = (Parameters*)malloc(sizeof(Parameters));

    parameters->gen = 100u;
    parameters->p_crossover = 0.90;// 0.95;
    parameters->eta_crossover = 10.0;
    parameters->p_mutation = 0.1;// 0.01;
    parameters->eta_mutation = 50.0;

    parameters->cpu_threads = 2u;
    parameters->gpu_block_size = 256u;
    parameters->curand_seed = std::random_device{}();

    return parameters;
}

void free_algorithm(Algorithm *algorithm) {
    if (algorithm->decision_variables)
        free(algorithm->decision_variables);

    if (algorithm->fitness_values)
        free(algorithm->fitness_values);

    if (algorithm->variant == GPU) {
        if(algorithm->self) cudaFreeHost(algorithm->self);
    } else if (algorithm->variant == HYBRID) {
        if(algorithm->self) cudaFreeHost(algorithm->self);
    } else {
        if (algorithm->self) free(algorithm->self);
    }

    if (algorithm->gpu_self)
        cudaFree(algorithm->gpu_self);

    free(algorithm);
}
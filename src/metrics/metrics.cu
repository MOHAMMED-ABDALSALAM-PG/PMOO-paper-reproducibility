#include "metrics/metrics.cuh"
#include "metrics/utils.cuh"
#include "algorithms/utils.cuh"

#include <random> // mt19937, random_device and distributions

#include <math.h>

double GD(double **front, double **population, const unsigned front_size, const unsigned population_size, const unsigned dim, double norm) {
    double distance = 0.0;

    #pragma omp parallel for schedule(static) reduction(+:distance)
    for (OMP_FOR_INDEX i = 0u; i < population_size; ++i) {
        double best = INFINITY;
        for (OMP_FOR_INDEX j = 0u; j < front_size; ++j) {
            double tmp = calculate_distance(population[i], front[j], dim, norm);
            if (tmp < best) best = tmp;
        }
        distance += best;
    }

    return distance / (double)population_size;
}

double IGD(double **front, double **population, const unsigned front_size, const unsigned population_size, const unsigned dim, double norm) {
    double distance = 0.0;

    #pragma omp parallel for schedule(static) reduction(+:distance)
    for (OMP_FOR_INDEX i = 0u; i < front_size; ++i) {
        double best = INFINITY;
        for (OMP_FOR_INDEX j = 0u; j < population_size; ++j) {
            double tmp = calculate_distance(front[i], population[j], dim, norm);
            if (tmp < best) best = tmp;
        }
        distance += pow(best, norm);
    }

    return pow(distance, 1.0 / norm) / (double)front_size;
}

// NOTE: Hypervolume implementation using monte carlo approach
double HV(double *ref_point, const unsigned N, double **population, const unsigned population_size, const unsigned dim) {
    double volume = 1.0;
    for (unsigned i = 0u; i < dim; ++i) volume *= ref_point[i]; // we are assuming that space is bounded by 0.0

    unsigned dominated = 0u;
    #pragma omp parallel reduction(+:dominated)
    {
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> *distributions = (std::uniform_real_distribution<double>*)malloc(dim * sizeof(std::uniform_real_distribution<double>));
        for (OMP_FOR_INDEX i = 0u; i < dim; ++i) distributions[i] = std::uniform_real_distribution<double>(0.0, ref_point[i]);
        
        double *point = (double*)malloc(dim * sizeof(double));
        #pragma omp for schedule(static)
        for (OMP_FOR_INDEX n = 0u; n < N; ++n) {
            for (OMP_FOR_INDEX i = 0u; i < dim; ++i) point[i] = distributions[i](rng); // randomize point inside volume bounds
            for (OMP_FOR_INDEX p = 0u; p < population_size; ++p) { // check if its dominated by any point in the population
                if (pareto_dominance(population[p], point, dim)) {
                    ++dominated;
                    break;
                }
            }
        }

        free(distributions);
        free(point);
    }

    return ((double)dominated / (double)N) * volume;
}
#include "algorithms/utils.cuh"

// checks whether point1 pareto dominates point2
__host__ __device__ bool pareto_dominance(const double *point1, const double *point2, const unsigned d_dim) {
    bool dominating = false;
    for (unsigned dim = 0u; dim < d_dim; ++dim) {
        if (point1[dim] > point2[dim]) {
            return false;
        } else if (point1[dim] < point2[dim]) {
            dominating = true;
        }
    }
    return dominating;
}

// helper function for SBX
__host__ __device__ double sbx_betaq(double beta, double crossover_index, double rnd) {
    double alpha = 2.0 - pow(beta, -(crossover_index + 1.0));
    if (rnd < 1.0 / alpha) {
        return pow(rnd * alpha, 1.0 / (crossover_index + 1.0));
    } else {
        return pow(1.0 / (2.0 - rnd * alpha), 1.0 / (crossover_index + 1.0));
    }
}
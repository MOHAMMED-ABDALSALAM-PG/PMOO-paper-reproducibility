#include "metrics/utils.cuh"

#include <math.h>
#include <stdio.h>

double calculate_distance(const double *A, const double *B, const unsigned dim, double norm) {
    double distance = 0.0;

    for (unsigned i = 0; i < dim; ++i)
        distance += pow(A[i] - B[i], norm);
    
    return pow(distance, 1.0 / norm);
}
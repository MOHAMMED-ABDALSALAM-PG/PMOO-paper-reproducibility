#ifndef PMOO_METRICS_CUH
#define PMOO_METRICS_CUH

double GD(double **front, double **population, const unsigned front_size, const unsigned population_size, const unsigned dim, double norm = 2.0);

double IGD(double **front, double **population, const unsigned front_size, const unsigned population_size, const unsigned dim, double norm = 2.0);

double HV(double *ref_point, const unsigned N, double **population, const unsigned population_size, const unsigned dim);

#endif
#ifndef PMOO_BENCHMARK_UTILS_CUH
#define PMOO_BENCHMARK_UTILS_CUH

#include <algorithms/algorithm.cuh>

#include <stdio.h>

typedef struct measurement {
    const char *algorithm;
    const char *variant;
    const char *problem_suite;
    unsigned problem_id;
    unsigned cpu_threads;
    unsigned gpu_block_size;
    double execution_time;
    double gd;
    double igd;
    double hv;
} Measurement;

void algorithm_print_to_console(const unsigned algorithm_id, const unsigned algorithm_total, const char* name, const char *variant);

void pre_run_print_to_console(const unsigned run_id, const unsigned run_total, const Measurement* measurement);

void post_run_print_to_console(const Measurement* measurement);

void write_measurement_header_to_csv(FILE *file);

void write_measurement_to_csv(FILE *file, Measurement* measurement);

void write_population_header_to_csv(FILE *file);

void write_population_to_csv(FILE *file, Algorithm* algorithm, unsigned d_dim, unsigned f_dim);

#endif
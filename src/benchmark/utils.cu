#include "benchmark/utils.cuh"

const char CSV_SEP[] = ",";
const char CSV_NL[] = "\n";

void algorithm_print_to_console(const unsigned algorithm_id, const unsigned algorithm_total, const char* name, const char *variant) {
    printf("[%3u/%-3u] %s (%s)\n", algorithm_id, algorithm_total, name, variant);
}

void pre_run_print_to_console(const unsigned run_id, const unsigned run_total, const Measurement* measurement) {
    printf("\t[%3u/%-3u] %6s%u (CPU-T: %-4u GPU-BS: %-4u)\t",
        run_id, run_total, measurement->problem_suite,
        measurement->problem_id, measurement->cpu_threads, measurement->gpu_block_size);
    fflush(stdout);
}

void post_run_print_to_console(const Measurement* measurement) {
    printf("t: %.5lfs\tGD: %.5lf\tIGD: %.5lf\tHV: %.5lf\n",
        measurement->execution_time, measurement->gd, measurement->igd, measurement->hv);
}

void write_measurement_header_to_csv(FILE *file) {
    fprintf(file, "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s",
        "algorithm", CSV_SEP,
        "variant", CSV_SEP,
        "problem", CSV_SEP,
        "cpu_threads", CSV_SEP,
        "gpu_block_size", CSV_SEP,
        "execution_time", CSV_SEP,
        "generational_distance", CSV_SEP,
        "inverted_generational_distance", CSV_SEP,
        "hypervolume", CSV_NL);
    fflush(file);
}

void write_measurement_to_csv(FILE *file, Measurement* measurement) {
    fprintf(file, "%s%s%s%s%s%u%s%u%s%u%s%lf%s%lf%s%lf%s%lf%s",
        measurement->algorithm, CSV_SEP,
        measurement->variant, CSV_SEP,
        measurement->problem_suite, measurement->problem_id, CSV_SEP,
        measurement->cpu_threads, CSV_SEP,
        measurement->gpu_block_size, CSV_SEP,
        measurement->execution_time, CSV_SEP,
        measurement->gd, CSV_SEP,
        measurement->igd, CSV_SEP,
        measurement->hv, CSV_NL);
    fflush(file);
}

void write_1d_vector_to_file(FILE *file, double* vector, unsigned dim) {
    fprintf(file, "[");
    for (unsigned i = 0u; i < dim; ++i) {
        fprintf(file, "%lf", vector[i]);
        if (i < dim - 1u) {
            fprintf(file, "%s", CSV_SEP);
        }
    }
    fprintf(file, "]");
}

void write_2d_vector_to_file(FILE *file, double** vector, unsigned dim1, unsigned dim2) {
    fprintf(file, "[");
    for (unsigned i = 0u; i < dim1; ++i) {
        write_1d_vector_to_file(file, vector[i], dim2);
        if (i < dim1 - 1u) {
            fprintf(file, "%s", CSV_SEP);
        }
    }
    fprintf(file, "]");
}

void write_population_header_to_csv(FILE *file) {
    fprintf(file, "%s%s%s%s",
        "decision_variables", CSV_SEP,
        "fitness_values", CSV_NL);
    fflush(file);
}

void write_population_to_csv(FILE *file, Algorithm* algorithm, unsigned d_dim, unsigned f_dim) {
    fprintf(file, "\"");
    write_2d_vector_to_file(file, algorithm->decision_variables, algorithm->population_size, d_dim);
    fprintf(file, "\"%s\"", CSV_SEP);
    write_2d_vector_to_file(file, algorithm->fitness_values, algorithm->population_size, f_dim);
    fprintf(file, "\"%s", CSV_NL);
    fflush(file);
}
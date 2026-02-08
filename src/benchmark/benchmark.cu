#include "benchmark/utils.cuh"
#include "problems/problem.cuh"
#include "metrics/metrics.cuh"
#include "algorithms/algorithm.cuh"

#include "algorithms/serial/nsga2.cuh"
#include "algorithms/cpu/nsga2.cuh"
#include "algorithms/gpu/nsga2.cuh"
#include "algorithms/hybrid/nsga2.cuh"

#include "algorithms/serial/spea2.cuh"
#include "algorithms/cpu/spea2.cuh"
#include "algorithms/gpu/spea2.cuh"
#include "algorithms/hybrid/spea2.cuh"

#include "algorithms/serial/moead_t_d.cuh"
#include "algorithms/cpu/moead_t_d.cuh"
#include "algorithms/gpu/moead_t_d.cuh"
#include "algorithms/hybrid/moead_t_d.cuh"

#include "algorithms/serial/moead_t_dd.cuh"
#include "algorithms/cpu/moead_t_dd.cuh"
#include "algorithms/gpu/moead_t_dd.cuh"
#include "algorithms/hybrid/moead_t_dd.cuh"

#include "algorithms/serial/moead_t_hs.cuh"
#include "algorithms/cpu/moead_t_hs.cuh"
#include "algorithms/gpu/moead_t_hs.cuh"
#include "algorithms/hybrid/moead_t_hs.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/*
 * NOTE:
 * generate_algorithm should be declared in algorithm.cuh and algorithm.cu but they cannot be due to:
 * a) use of templates
 * b) circular dependencies caused by overcoming template restrictions
 */
template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
Algorithm* generate_algorithm(AlgorithmEnum algorithm, VariantEnum variant) {
    switch (algorithm) {
        case NSGA2:
            switch (variant) {
                case SERIAL:
                    return pmoo_nsga2_serial_generate<POP_SIZE, D_DIM, F_DIM>();
                case CPU:
                    return pmoo_nsga2_cpu_generate<POP_SIZE, D_DIM, F_DIM>();
                case GPU:
                    return pmoo_nsga2_gpu_generate<POP_SIZE, D_DIM, F_DIM>();
                case HYBRID:
                    return pmoo_nsga2_hybrid_generate<POP_SIZE, D_DIM, F_DIM>();
                default:
                    return NULL;
            }
        case SPEA2:
            switch (variant) {
                case SERIAL:
                    return pmoo_spea2_serial_generate<POP_SIZE, D_DIM, F_DIM>();
                case CPU:
                    return pmoo_spea2_cpu_generate<POP_SIZE, D_DIM, F_DIM>();
                case GPU:
                    return pmoo_spea2_gpu_generate<POP_SIZE, D_DIM, F_DIM>();
                case HYBRID:
                    return pmoo_spea2_hybrid_generate<POP_SIZE, D_DIM, F_DIM>();
                default:
                    return NULL;
            }
        case MOEAD_T_D:
            switch (variant) {
                case SERIAL:
                    return pmoo_moead_t_d_serial_generate<POP_SIZE, D_DIM, F_DIM>();
                case CPU:
                    return pmoo_moead_t_d_cpu_generate<POP_SIZE, D_DIM, F_DIM>();
                case GPU:
                    return pmoo_moead_t_d_gpu_generate<POP_SIZE, D_DIM, F_DIM>();
                case HYBRID:
                    return pmoo_moead_t_d_hybrid_generate<POP_SIZE, D_DIM, F_DIM>();
                default:
                    return NULL;
            }
        case MOEAD_T_DD:
            switch (variant) {
                case SERIAL:
                    return pmoo_moead_t_dd_serial_generate<POP_SIZE, D_DIM, F_DIM>();
                case CPU:
                    return pmoo_moead_t_dd_cpu_generate<POP_SIZE, D_DIM, F_DIM>();
                case GPU:
                    return pmoo_moead_t_dd_gpu_generate<POP_SIZE, D_DIM, F_DIM>();
                case HYBRID:
                    return pmoo_moead_t_dd_hybrid_generate<POP_SIZE, D_DIM, F_DIM>();
                default:
                    return NULL;
            }
        case MOEAD_T_HS:
            switch (variant) {
                case SERIAL:
                    return pmoo_moead_t_hs_serial_generate<POP_SIZE, D_DIM, F_DIM>();
                case CPU:
                    return pmoo_moead_t_hs_cpu_generate<POP_SIZE, D_DIM, F_DIM>();
                case GPU:
                    return pmoo_moead_t_hs_gpu_generate<POP_SIZE, D_DIM, F_DIM>();
                case HYBRID:
                    return pmoo_moead_t_hs_hybrid_generate<POP_SIZE, D_DIM, F_DIM>();
                default:
                    return NULL;
            }
        default:
            return NULL;
    }
}

int main(int argc, char **argv) {
    /*
     * NOTE:
     * 1. Keep POP_SIZE a power of 2 to better fit CUDA kernel block sizes
     * (we didn't implement block size checks into kernels)
     * 2. Our code won't work for F_DIM >= 9 due to integer overflows and memory limitations
     * 3. Setting too big POP_SIZE, D_DIM and F_DIM in total won't work due to too high memory consumption.
     * Each algorithm memory consumption scales with all of these parameters, specially with POP_SIZE.
     */
    // Algorithm and problem parameters
    const unsigned POP_SIZE = 512u;
    const unsigned D_DIM = 10u;
    const unsigned F_DIM = 2u;

    // Test settings
    const unsigned TEST_PROBLEMS_COUNT = total_test_count();
    const unsigned ALGORITHM_COUNT = total_algorithm_count();
    const unsigned RUNS_PER_TEST_PROBLEM = 100u;
    const unsigned THREADS_TEST_COUNT = 4u;
    const unsigned THREADS[THREADS_TEST_COUNT] = {2u, 4u, 8u, 16u};
    const unsigned BLOCK_SIZES_TEST_COUNT = 1u;
    const unsigned BLOCK_SIZES[BLOCK_SIZES_TEST_COUNT] = {256u};

    FILE *test_results_file, *population_results_file;
    char test_results_filename[32u], population_results_filename[32u];
    double timer_start = 0.0, timer_end = 0.0;

    Measurement measurement;
    Parameters *parameters = default_parameters();
    Problem **problems = generate_all_problems(D_DIM, F_DIM), *problem;
    Algorithm *algorithm;
    unsigned algorithm_counter = 0u;

    sprintf(test_results_filename, "N%uM%un%u.csv", POP_SIZE, F_DIM, D_DIM);
    test_results_file = fopen(test_results_filename, "w");
    write_measurement_header_to_csv(test_results_file);

    sprintf(population_results_filename, "N%uM%un%u_pop.csv", POP_SIZE, F_DIM, D_DIM);
    population_results_file = fopen(population_results_filename, "w");
    write_population_header_to_csv(population_results_file);

    for (unsigned algorithm_enum = 0u; algorithm_enum < ALGORITHMS_COUNT; ++algorithm_enum) { // for every algorithm
        for (unsigned variant_enum = 0u; variant_enum < VARIANTS_COUNT; ++variant_enum) { // for every variant
            ++algorithm_counter;
            algorithm = generate_algorithm<POP_SIZE, D_DIM, F_DIM>((AlgorithmEnum)algorithm_enum, (VariantEnum)variant_enum);

            if (!algorithm) { // QOL skipping disabled algorithms
                printf("[%3u/%-3u] Skipping %s %s (pointer NULL)\n", algorithm_counter, ALGORITHM_COUNT, ALGORITHMS_NAMES[algorithm_enum], VARIANTS_NAMES[variant_enum]);
                continue;
            }

            measurement.algorithm = ALGORITHMS_NAMES[(unsigned)algorithm->name];
            measurement.variant = VARIANTS_NAMES[(unsigned)algorithm->variant];
            algorithm_print_to_console(algorithm_counter, ALGORITHM_COUNT, measurement.algorithm, measurement.variant);

            for (unsigned t = 1u; t <= TEST_PROBLEMS_COUNT; ++t) { // for every test problem
                problem = problems[t - 1u]; // select test problem from the list

                if (!problem->fitness) { // QOL skipping disabled problems
                    printf("\tSkipping %s%u (pointer NULL)\n", TEST_SUITES_NAMES[problem->suite], problem->problem_id);
                    continue;
                }

                measurement.problem_suite = TEST_SUITES_NAMES[(unsigned)problem->suite];
                measurement.problem_id = problem->problem_id;

                algorithm->load_problem(algorithm->self, problem); // load problem into algorithm

                for (unsigned tr = 1u; tr <= RUNS_PER_TEST_PROBLEM; ++tr) {
                    if (algorithm->variant == CPU) { // for openmp implementations
                        for (unsigned th = 1u; th <= THREADS_TEST_COUNT; ++th) { // test all thread numbers
                            measurement.cpu_threads = THREADS[th - 1u];
                            measurement.gpu_block_size = 0u;
                            pre_run_print_to_console((tr - 1u) * THREADS_TEST_COUNT + th, RUNS_PER_TEST_PROBLEM * THREADS_TEST_COUNT, &measurement);

                            parameters->cpu_threads = THREADS[th - 1u];
                            parameters->gpu_block_size = 0u;
                            algorithm->load_parameters(algorithm->self, parameters); // load parameters into algorithm
                            algorithm->initialize(algorithm->self, NULL); // initialize algorithm

                            timer_start = omp_get_wtime();
                            algorithm->evolve(algorithm->self, NULL);
                            timer_end = omp_get_wtime();

                            algorithm->cleanup(algorithm->self, NULL);

                            measurement.execution_time = timer_end - timer_start;
                            omp_set_num_threads(THREADS[THREADS_TEST_COUNT - 1u]);
                            measurement.gd = GD(problem->front, algorithm->fitness_values, problem->front_size, algorithm->population_size, problem->f_dim);
                            measurement.igd = IGD(problem->front, algorithm->fitness_values, problem->front_size, algorithm->population_size, problem->f_dim);
                            measurement.hv = HV(problem->ref_point, problem->hypervolume_N, algorithm->fitness_values, algorithm->population_size, problem->f_dim);
                            write_measurement_to_csv(test_results_file, &measurement);
                            write_population_to_csv(population_results_file, algorithm, problem->d_dim, problem->f_dim);
                            post_run_print_to_console(&measurement);
                        }
                    } else if (algorithm->variant == GPU) { // for cuda implementations
                        for (unsigned bs = 1u; bs <= BLOCK_SIZES_TEST_COUNT; ++bs) { // test all block sizes
                            measurement.cpu_threads = 0u;
                            measurement.gpu_block_size = BLOCK_SIZES[bs - 1u];
                            pre_run_print_to_console((tr - 1u) * BLOCK_SIZES_TEST_COUNT + bs, RUNS_PER_TEST_PROBLEM * BLOCK_SIZES_TEST_COUNT, &measurement);

                            parameters->cpu_threads = 0u;
                            parameters->gpu_block_size = BLOCK_SIZES[bs - 1u];
                            algorithm->load_parameters(algorithm->self, parameters); // load parameters into algorithm
                            algorithm->initialize(algorithm->self, algorithm->gpu_self); // initialize algorithm

                            timer_start = omp_get_wtime();
                            algorithm->evolve(algorithm->self, algorithm->gpu_self);
                            timer_end = omp_get_wtime();

                            algorithm->cleanup(algorithm->self, algorithm->gpu_self);

                            measurement.execution_time = timer_end - timer_start;
                            omp_set_num_threads(THREADS[THREADS_TEST_COUNT - 1u]);
                            measurement.gd = GD(problem->front, algorithm->fitness_values, problem->front_size, algorithm->population_size, problem->f_dim);
                            measurement.igd = IGD(problem->front, algorithm->fitness_values, problem->front_size, algorithm->population_size, problem->f_dim);
                            measurement.hv = HV(problem->ref_point, problem->hypervolume_N, algorithm->fitness_values, algorithm->population_size, problem->f_dim);
                            write_measurement_to_csv(test_results_file, &measurement);
                            write_population_to_csv(population_results_file, algorithm, problem->d_dim, problem->f_dim);
                            post_run_print_to_console(&measurement);
                        }
                    } else if (algorithm->variant == HYBRID) { // for cuda + OpenMP implementations
                        for (unsigned bs = 1u; bs <= BLOCK_SIZES_TEST_COUNT; ++bs) { // test all block sizes
                            for (unsigned th = 1u; th <= THREADS_TEST_COUNT; ++th) { // test all thread numbers
                                measurement.cpu_threads = THREADS[th - 1u];
                                measurement.gpu_block_size = BLOCK_SIZES[bs - 1u];
                                pre_run_print_to_console((tr - 1u) * (THREADS_TEST_COUNT * BLOCK_SIZES_TEST_COUNT) + (bs - 1u) * THREADS_TEST_COUNT + th, RUNS_PER_TEST_PROBLEM * (THREADS_TEST_COUNT * BLOCK_SIZES_TEST_COUNT), &measurement);

                                parameters->cpu_threads = THREADS[th - 1u];
                                parameters->gpu_block_size = BLOCK_SIZES[bs - 1u];
                                algorithm->load_parameters(algorithm->self, parameters); // load parameters into algorithm
                                algorithm->initialize(algorithm->self, algorithm->gpu_self); // initialize algorithm

                                timer_start = omp_get_wtime();
                                algorithm->evolve(algorithm->self, algorithm->gpu_self);
                                timer_end = omp_get_wtime();

                                algorithm->cleanup(algorithm->self, algorithm->gpu_self);

                                measurement.execution_time = timer_end - timer_start;
                                omp_set_num_threads(THREADS[THREADS_TEST_COUNT - 1u]);
                                measurement.gd = GD(problem->front, algorithm->fitness_values, problem->front_size, algorithm->population_size, problem->f_dim);
                                measurement.igd = IGD(problem->front, algorithm->fitness_values, problem->front_size, algorithm->population_size, problem->f_dim);
                                measurement.hv = HV(problem->ref_point, problem->hypervolume_N, algorithm->fitness_values, algorithm->population_size, problem->f_dim);
                                write_measurement_to_csv(test_results_file, &measurement);
                                write_population_to_csv(population_results_file, algorithm, problem->d_dim, problem->f_dim);
                                post_run_print_to_console(&measurement);
                            }
                        }
                    } else { // single test for serial implementation
                        measurement.cpu_threads = 1u;
                        measurement.gpu_block_size = 0u;
                        pre_run_print_to_console(tr, RUNS_PER_TEST_PROBLEM, &measurement);

                        parameters->cpu_threads = 1u;
                        parameters->gpu_block_size = 0u;
                        algorithm->load_parameters(algorithm->self, parameters); // load parameters into algorithm
                        algorithm->initialize(algorithm->self, NULL); // initialize algorithm

                        timer_start = omp_get_wtime();
                        algorithm->evolve(algorithm->self, NULL);
                        timer_end = omp_get_wtime();

                        algorithm->cleanup(algorithm->self, NULL);

                        measurement.execution_time = timer_end - timer_start;
                        omp_set_num_threads(THREADS[THREADS_TEST_COUNT - 1u]);
                        measurement.gd = GD(problem->front, algorithm->fitness_values, problem->front_size, algorithm->population_size, problem->f_dim);
                        measurement.igd = IGD(problem->front, algorithm->fitness_values, problem->front_size, algorithm->population_size, problem->f_dim);
                        measurement.hv = HV(problem->ref_point, problem->hypervolume_N, algorithm->fitness_values, algorithm->population_size, problem->f_dim);
                        write_measurement_to_csv(test_results_file, &measurement);
                        write_population_to_csv(population_results_file, algorithm, problem->d_dim, problem->f_dim);
                        post_run_print_to_console(&measurement);
                    }
                }
            }

            free_algorithm(algorithm);
        }
    }

    fclose(test_results_file);
    fclose(population_results_file);

    // free problems memory
    for (unsigned p = 0u; p < TEST_PROBLEMS_COUNT; ++p)
        free_problem(problems[p]);
    free(problems);

    return 0;
}
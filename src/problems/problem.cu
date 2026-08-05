#include "problems/problem.cuh"
#include "problems/suites/zdt.cuh"
#include "problems/suites/dtlz.cuh"
#include "problems/suites/wfg.cuh"
#include "problems/suites/custom.cuh"
#include "problems/suites/heat_conduction.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Problem* generate_problem(TestSuiteEnum suite, unsigned problem_id, unsigned d_dim, unsigned f_dim) {
    switch (suite) {
        case ZDT:
            return generate_zdt_problem(problem_id, d_dim, f_dim);
        case DTLZ:
            return generate_dtlz_problem(problem_id, d_dim, f_dim);
        case WFG:
            return generate_wfg_problem(problem_id, d_dim, f_dim);
        case CUSTOM:
            return generate_custom_problem(problem_id, d_dim, f_dim);
        case HEAT_CONDUCTION:
            return generate_heat_conduction_problem(problem_id, d_dim, f_dim);
        default:
            return NULL;
    }
}

Problem** generate_all_problems(unsigned d_dim, unsigned f_dim) {
    unsigned problem_counter = 0u;

    Problem** problems = (Problem**)malloc(sizeof(Problem*) * total_test_count());

    for (unsigned suite = 0u; suite < TEST_SUITES_COUNT; ++suite) {
        for (unsigned problem_id = 1u; problem_id <= TEST_SUITES_PROBLEM_COUNT[suite]; ++problem_id) {
            problems[problem_counter++] = generate_problem((TestSuiteEnum)suite, problem_id, d_dim, f_dim);
            // _export_pareto_front(problems[problem_counter - 1u]);
        }
    }

    return problems;
}

void free_problem(Problem *problem) {
    if (problem == NULL) return;
    if (problem->suite == HEAT_CONDUCTION) heat_conduction_cleanup();
    free(problem->upper_bounds);
    free(problem->lower_bounds);
    if (problem->ref_point != NULL)
        free(problem->ref_point);
    if (problem->front != NULL) {
        free(problem->front[0]); // 2d array was allocated as contignous 1d block
        free(problem->front);
    }
    free(problem);
}

// Used to export pareto front to verify/visualize using pymoo
void _export_pareto_front(Problem* problem) {
    if (problem->front) {
        char filename[16];
        sprintf(filename, "%s%d.py", TEST_SUITES_NAMES[problem->suite], problem->problem_id);
        FILE *file = fopen(filename, "w");
        fprintf(file, "%s%d = [", TEST_SUITES_NAMES[problem->suite], problem->problem_id);
        for (unsigned i = 0u; i < problem->front_size; ++i) {
            fprintf(file, "[");
            for (unsigned dim = 0u; dim < problem->f_dim; ++dim) {
                if (dim < problem->f_dim - 1u) fprintf(file, "%.5lf,", problem->front[i][dim]);
                else fprintf(file, "%.5lf", problem->front[i][dim]);
            }
            if (i < problem->front_size - 1u) fprintf(file, "],");
            else fprintf(file, "]");
        }
        fprintf(file, "]\n");
        fclose(file);
    }
}

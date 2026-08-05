#ifndef PMOO_PROBLEM_CUH
#define PMOO_PROBLEM_CUH

#define REF_POINT_MULTIPLIER 1.2

typedef void (*Fitness)(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim);

enum TestSuiteEnum {
    ZDT = 0,
    DTLZ,
    WFG,
    CUSTOM,
    HEAT_CONDUCTION,
    TEST_SUITES_COUNT
};

const TestSuiteEnum TEST_SUITES[TEST_SUITES_COUNT] = { ZDT, DTLZ, WFG, CUSTOM, HEAT_CONDUCTION };
const char* const TEST_SUITES_NAMES[TEST_SUITES_COUNT] = { "ZDT", "DTLZ", "WFG", "CUSTOM", "HEAT_CONDUCTION" };
// Heat conduction is available through generate_problem(HEAT_CONDUCTION, ...)
// but is excluded from the default analytical-suite sweep because it requires
// the dedicated paper configuration d_dim=21 and f_dim=3.
const unsigned TEST_SUITES_PROBLEM_COUNT[TEST_SUITES_COUNT] = { 6u, 7u, 0u, 0u, 0u }; // { 6u, 7u, 9u, 1u, 1u }

typedef struct problem {
    TestSuiteEnum suite;            // test suite id
    unsigned problem_id;            // id of a problem in its test suite
    unsigned d_dim;                 // problem decision dimensions
    unsigned f_dim;                 // problem fitness dimensions
    unsigned i_dim;                 // integer decision dimensions
    double *upper_bounds;           // problem decision variables bounds
    double *lower_bounds;           // problem decision variables bounds
    Fitness fitness;                // problem fitness function
    Fitness gpu_fitness;            // problem fitness function located on device
    // for metrics
    unsigned front_size;            // points count of pareto front
    double **front;                 // set of points belonging to the pareto front
    double *ref_point;              // reference point for hypervolume (f_dim)
    unsigned hypervolume_N;         // suggested value N to calculate hypervolume
} Problem;

inline unsigned total_test_count() {
    unsigned total_problem_count = 0u;
    for (unsigned i = 0u; i < TEST_SUITES_COUNT; ++i) total_problem_count += TEST_SUITES_PROBLEM_COUNT[i];
    return total_problem_count;
}

Problem* generate_problem(TestSuiteEnum suite, unsigned problem_id, unsigned d_dim, unsigned f_dim);

Problem** generate_all_problems(unsigned d_dim, unsigned f_dim);

void free_problem(Problem *problem);

void _export_pareto_front(Problem* problem);

#endif

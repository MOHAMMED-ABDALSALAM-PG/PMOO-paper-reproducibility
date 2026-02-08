#ifndef PMOO_SERIAL_NSGA2_CUH
#define PMOO_SERIAL_NSGA2_CUH

#include "../algorithm.cuh"
#include "../utils.cuh"

#include <random> // mt19937, random_device and distributions
#include <algorithm> // sort

#include <string.h> // memcpy, memset
#include <math.h>

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
struct nsga2_serial {
    // algorithm settings
    unsigned gen;           // Number of generations to run in a single evolve call
    double p_crossover;     // Crossover probability
    double eta_crossover;   // Crossover index (determines how similar are children to parents)
    double p_mutation;      // Mutation probability
    double eta_mutation;    // Mutation index (determines how big of a change the mutated value will be)
    // algorithm and problem metadata
    unsigned pop_size;          // How many individuals are in population
    unsigned d_dim;             // How many decision variables does the problem have
    unsigned f_dim;             // How many objectives does the problem have
    unsigned i_dim;             // How many integer dimensions does the problem have (integer variables are stored at the end of deicision vector)
    double upper_bounds[D_DIM]; // Upper bounds for decision variables values based on problem constraints
    double lower_bounds[D_DIM]; // Lower bounds for decision variables values based on problem constraints
    void (*fitness)(const double *decision_variables, double *fitness_values, const unsigned d_dim, const unsigned f_dim); // fitness evaluation function
    // variables used for calculations
    double decision_variables[2u * POP_SIZE][D_DIM];                            // Population decision variables
    double fitness_values[2u * POP_SIZE][F_DIM];                                // Population fitness values
    double selected_decision_variables[POP_SIZE][D_DIM];                        // Used for selecting best N individuals
    double selected_fitness_values[POP_SIZE][F_DIM];                            // Used for selecting best N individuals
    unsigned shuffle[2u * POP_SIZE];                                            // Used to store random order for tournament selection
    struct array_with_size<2u * POP_SIZE> non_dominated_front[2u * POP_SIZE];   // Used to store fronts calculated from fnds
    struct array_with_size<2u * POP_SIZE> domination_list[2u * POP_SIZE];       // Used to store indexes of individuals who i-th individual dominates
    unsigned non_domination_rank[2u * POP_SIZE];                                // Used to store non domination ranks calculated from fnds
    unsigned domination_count[2u * POP_SIZE];                                   // Used to store how many individuals dominate i-th individual
    double crowding_distance[2u * POP_SIZE];                                    // Used to store crowding distance value for each individual
    struct array_with_size<2u * POP_SIZE> crowding_distance_order;              // Used to calculate individual orders based on crowding distance
    // rng
    std::mt19937 random;
};

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
// Problem metadata should be loaded first before initializing algorithm
void nsga2_serial_initialize(struct nsga2_serial<POP_SIZE, D_DIM, F_DIM> *nsga2) {
    // initialize random number generators
    nsga2->random.seed(std::random_device{}());

    // initialize shuffle arrays
    for (unsigned i = 0u; i < nsga2->pop_size; ++i) nsga2->shuffle[i] = i;
    for (unsigned i = 0u; i < nsga2->pop_size; ++i) nsga2->shuffle[nsga2->pop_size + i] = i;

    // initialize continous decision variables
    for (unsigned dim = 0u; dim < nsga2->d_dim - nsga2->i_dim; ++dim) {
        std::uniform_real_distribution<double> uniform(nsga2->lower_bounds[dim], nsga2->upper_bounds[dim]);
        for (unsigned i = 0u; i < nsga2->pop_size; ++i) {
            nsga2->decision_variables[i][dim] = uniform(nsga2->random);
        }
    }

    // initialize integer decision variables
    for (unsigned dim = nsga2->d_dim - nsga2->i_dim; dim < nsga2->d_dim; ++dim) {
        std::uniform_real_distribution<double> uniform(nsga2->lower_bounds[dim], nsga2->upper_bounds[dim]);
        for (unsigned i = 0u; i < nsga2->pop_size; ++i) {
            nsga2->decision_variables[i][dim] = uniform(nsga2->random);
        }
    }

    // calculate fitness for initialized population
    for (unsigned i = 0u; i < nsga2->pop_size; ++i) {
        nsga2->fitness(nsga2->decision_variables[i], nsga2->fitness_values[i], nsga2->d_dim, nsga2->f_dim);
    }
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
// Includes parameter N to make it usable on population sizes N and 2N
void nsga2_serial_fast_non_dominated_sorting(struct nsga2_serial<POP_SIZE, D_DIM, F_DIM> *nsga2, const unsigned N) {
    // reset all fnds related values
    for (unsigned i = 0u; i < N; ++i) nsga2->non_dominated_front[i].size = 0u;
    for (unsigned i = 0u; i < N; ++i) nsga2->domination_list[i].size = 0u;
    memset(nsga2->domination_count, 0, N * sizeof(unsigned));
    memset(nsga2->non_domination_rank, 0, N * sizeof(unsigned));

    // compare all individuals with each other to figure out domination relations
    for (unsigned i = 0u; i < N; ++i) {
        for (unsigned j = 0u; j < i; ++j) {
            if (pareto_dominance(nsga2->fitness_values[i], nsga2->fitness_values[j], nsga2->f_dim)) {
                nsga2->domination_list[i].elements[nsga2->domination_list[i].size] = j;
                nsga2->domination_list[i].size += 1u;
                nsga2->domination_count[j] += 1u;
            } else if (pareto_dominance(nsga2->fitness_values[j], nsga2->fitness_values[i], nsga2->f_dim)) {
                nsga2->domination_list[j].elements[nsga2->domination_list[j].size] = i;
                nsga2->domination_list[j].size += 1u;
                nsga2->domination_count[i] += 1u;
            }
        }
    }

    // construct first front
    for (unsigned i = 0u; i < N; ++i) {
        if (nsga2->domination_count[i] == 0u) { // if no individual dominates i-th individual
            nsga2->non_dominated_front[0u].elements[nsga2->non_dominated_front[0u].size] = i; // assign it to the first front (also implicitly given rank 0)
            nsga2->non_dominated_front[0u].size += 1u;
            nsga2->non_domination_rank[i] = 0u;
        }
    }

    // construct rest of the fronts
    unsigned current_front_idx = 0u;
    while (current_front_idx < N && nsga2->non_dominated_front[current_front_idx].size > 0u) {
        for (unsigned p = 0u; p < nsga2->non_dominated_front[current_front_idx].size; ++p) { // for each individual in a dominating front
            for (unsigned q = 0u; q < nsga2->domination_list[nsga2->non_dominated_front[current_front_idx].elements[p]].size; ++q) { // for each individual that is dominated by p-th individual
                nsga2->domination_count[nsga2->domination_list[nsga2->non_dominated_front[current_front_idx].elements[p]].elements[q]] -= 1u; // remove domination from p-th individual on q-th individual
                if (nsga2->domination_count[nsga2->domination_list[nsga2->non_dominated_front[current_front_idx].elements[p]].elements[q]] == 0u) { // if no individual is no longer dominating q-th individual
                    nsga2->non_domination_rank[nsga2->domination_list[nsga2->non_dominated_front[current_front_idx].elements[p]].elements[q]] = current_front_idx + 1u; // give q-th individual next rank after p-th individual
                    nsga2->non_dominated_front[current_front_idx + 1u].elements[nsga2->non_dominated_front[current_front_idx + 1u].size] = nsga2->domination_list[nsga2->non_dominated_front[current_front_idx].elements[p]].elements[q]; // assign q-th individual to the next front
                    nsga2->non_dominated_front[current_front_idx + 1u].size += 1u;
                }
            }
        }
        ++current_front_idx;
    }
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
void nsga2_serial_crowding_distance(struct nsga2_serial<POP_SIZE, D_DIM, F_DIM> *nsga2, unsigned front_idx) {
    unsigned N = nsga2->non_dominated_front[front_idx].size; // individual count in the front

    // fill order with indexes of individuals from the front
    nsga2->crowding_distance_order.size = N;
    for (unsigned i = 0u; i < N; ++i) {
        nsga2->crowding_distance_order.elements[i] = nsga2->non_dominated_front[front_idx].elements[i];
    }

    for (unsigned dim = 0u; dim < nsga2->f_dim; ++dim) {
        // sort individuals based on objective value in one dimension in ascending order
        // TODO: implement non std::sort solution
        std::sort(nsga2->crowding_distance_order.elements, nsga2->crowding_distance_order.elements + nsga2->crowding_distance_order.size, [dim, nsga2](unsigned idx1, unsigned idx2) {
            return nsga2->fitness_values[idx1][dim] < nsga2->fitness_values[idx2][dim];
        });
        // set crowding distance to infinity for the first and last individual
        nsga2->crowding_distance[nsga2->crowding_distance_order.elements[0]] = INFINITY;
        nsga2->crowding_distance[nsga2->crowding_distance_order.elements[N - 1u]] = INFINITY;

        double df = nsga2->fitness_values[nsga2->crowding_distance_order.elements[N - 1u]][dim] - nsga2->fitness_values[nsga2->crowding_distance_order.elements[0u]][dim]; // used for normalization
        for (unsigned i = 1u; i < N - 1u; ++i) { // calculate crowding distance parts in this objective dimension for all inner individuals in a front
            nsga2->crowding_distance[nsga2->crowding_distance_order.elements[i]] += (nsga2->fitness_values[nsga2->crowding_distance_order.elements[i + 1u]][dim] - nsga2->fitness_values[nsga2->crowding_distance_order.elements[i - 1u]][dim]) / df; // crowding distance is normalized distance between neighbours of i-th individual
        }
    }
}

// Implementation of std::shuffle
template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
void nsga2_serial_shuffle_selection(struct nsga2_serial<POP_SIZE, D_DIM, F_DIM> *nsga2) {
    std::uniform_int_distribution<unsigned> dist(0u, nsga2->pop_size - 1u);
    unsigned tmp, rnd;

    for (unsigned i = 0u; i < nsga2->pop_size; ++i) {
        rnd = dist(nsga2->random);
        tmp = nsga2->shuffle[i];

        nsga2->shuffle[i] = nsga2->shuffle[rnd];
        nsga2->shuffle[rnd] = tmp;
    }

    for (unsigned i = 0u; i < nsga2->pop_size; ++i) {
        rnd = dist(nsga2->random);
        tmp = nsga2->shuffle[nsga2->pop_size + i];

        nsga2->shuffle[nsga2->pop_size + i] = nsga2->shuffle[nsga2->pop_size + rnd];
        nsga2->shuffle[nsga2->pop_size + rnd] = tmp;
    }
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
unsigned nsga2_serial_tournament_selection(struct nsga2_serial<POP_SIZE, D_DIM, F_DIM> *nsga2, const unsigned individual1, const unsigned individual2) {
    // select individual with lower non domination rank
    if (nsga2->non_domination_rank[individual1] < nsga2->non_domination_rank[individual2]) return individual1;
    if (nsga2->non_domination_rank[individual1] > nsga2->non_domination_rank[individual2]) return individual2;
    // select individual with larger crowding distance
    if (nsga2->crowding_distance[individual1] > nsga2->crowding_distance[individual2]) return individual1;
    if (nsga2->crowding_distance[individual1] < nsga2->crowding_distance[individual2]) return individual2;
    // select randomly
    std::uniform_real_distribution<double> uniform(0., 1.);
    return ((uniform(nsga2->random) < 0.5) ? individual1 : individual2);
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
void nsga2_serial_simulated_binary_crossover(struct nsga2_serial<POP_SIZE, D_DIM, F_DIM> *nsga2, const unsigned parent1_idx, const unsigned parent2_idx, const unsigned child1_idx, const unsigned child2_idx) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    
    // initialize children decision variables same as their parents
    memcpy(nsga2->decision_variables[child1_idx], nsga2->decision_variables[parent1_idx], nsga2->d_dim * sizeof(double));
    memcpy(nsga2->decision_variables[child2_idx], nsga2->decision_variables[parent2_idx], nsga2->d_dim * sizeof(double));

    if (uniform(nsga2->random) < nsga2->p_crossover) { // crossover might not happen with crossover probability
        // SBX - only for continous decision variables
        for (unsigned dim = 0u; dim < nsga2->d_dim - nsga2->i_dim; ++dim) { // for each decision variable in parents
            if (uniform(nsga2->random) < 0.5 && fabs(nsga2->decision_variables[parent1_idx][dim] - nsga2->decision_variables[parent2_idx][dim]) > TOLERANCE_EPSILON) { // each variable has 50% chance to be crossovered
                double y1, y2, beta, betaq, c1, c2, rnd;

                // find smaller variable in parents
                if (nsga2->decision_variables[parent1_idx][dim] < nsga2->decision_variables[parent2_idx][dim]) {
                    y1 = nsga2->decision_variables[parent1_idx][dim];
                    y2 = nsga2->decision_variables[parent2_idx][dim];
                } else {
                    y1 = nsga2->decision_variables[parent2_idx][dim];
                    y2 = nsga2->decision_variables[parent1_idx][dim];
                }

                rnd = uniform(nsga2->random);

                beta = 1.0 + (2.0 * (y1 - nsga2->lower_bounds[dim]) / (y2 - y1));
                betaq = sbx_betaq(beta, nsga2->eta_crossover, rnd);
                c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1));

                beta = 1.0 + (2.0 * (nsga2->upper_bounds[dim] - y2) / (y2 - y1));
                betaq = sbx_betaq(beta, nsga2->eta_crossover, rnd);
                c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1));

                // make sure that new variable values fit in problem bounds
                if (c1 < nsga2->lower_bounds[dim]) c1 = nsga2->lower_bounds[dim];
                if (c2 < nsga2->lower_bounds[dim]) c2 = nsga2->lower_bounds[dim];
                if (c1 > nsga2->upper_bounds[dim]) c1 = nsga2->upper_bounds[dim];
                if (c2 > nsga2->upper_bounds[dim]) c2 = nsga2->upper_bounds[dim];

                // assign new variable values randomly to children
                if (uniform(nsga2->random) < 0.5) {
                    nsga2->decision_variables[child1_idx][dim] = c1;
                    nsga2->decision_variables[child2_idx][dim] = c2;
                } else {
                    nsga2->decision_variables[child1_idx][dim] = c2;
                    nsga2->decision_variables[child2_idx][dim] = c1;
                }
            }
        }

        // Two-points crossover - only for integer decision variables
        if (nsga2->i_dim > 0u) {
            std::uniform_int_distribution<unsigned> i_dim_dist(nsga2->d_dim - nsga2->i_dim, nsga2->d_dim - 1u);
            unsigned cut1 = i_dim_dist(nsga2->random);
            unsigned cut2 = i_dim_dist(nsga2->random);
            if (cut1 < cut2) {
                memcpy(&nsga2->decision_variables[child1_idx][cut1], &nsga2->decision_variables[parent2_idx][cut1], (cut2 - cut1 + 1u) * sizeof(double));
                memcpy(&nsga2->decision_variables[child2_idx][cut1], &nsga2->decision_variables[parent1_idx][cut1], (cut2 - cut1 + 1u) * sizeof(double));
            } else {
                memcpy(&nsga2->decision_variables[child1_idx][cut2], &nsga2->decision_variables[parent2_idx][cut2], (cut1 - cut2 + 1u) * sizeof(double));
                memcpy(&nsga2->decision_variables[child2_idx][cut2], &nsga2->decision_variables[parent1_idx][cut2], (cut1 - cut2 + 1u) * sizeof(double));
            }
        }
    }
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
void nsga2_serial_polynomial_mutation(struct nsga2_serial<POP_SIZE, D_DIM, F_DIM> *nsga2, const unsigned individual) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    // polynomial mutation - only for continous decision variables
    double deltad, delta1, delta2, rnd, mutation_exp, xy, val, deltaq;
    for (unsigned dim = 0u; dim < nsga2->d_dim - nsga2->i_dim; ++dim) {
        if (uniform(nsga2->random) < nsga2->p_mutation) { // mutation might not occur
            deltad = nsga2->upper_bounds[dim] - nsga2->lower_bounds[dim];
            delta1 = (nsga2->decision_variables[individual][dim] - nsga2->lower_bounds[dim]) / (deltad);
            delta2 = (nsga2->upper_bounds[dim] - nsga2->decision_variables[individual][dim]) / (deltad);
            mutation_exp = 1. / (nsga2->eta_mutation + 1.);

            rnd = uniform(nsga2->random);
            if (rnd < 0.5) {
                xy = 1.0 - delta1;
                val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (pow(xy, (nsga2->eta_mutation + 1.0)));
                deltaq = pow(val, mutation_exp) - 1.0;
            } else {
                xy = 1.0 - delta2;
                val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (pow(xy, (nsga2->eta_mutation + 1.0)));
                deltaq = 1.0 - (pow(val, mutation_exp));
            }
            nsga2->decision_variables[individual][dim] += deltaq * deltad;

            // make sure that new variable value fits in problem bounds
            if (nsga2->decision_variables[individual][dim] < nsga2->lower_bounds[dim]) nsga2->decision_variables[individual][dim] = nsga2->lower_bounds[dim];
            if (nsga2->decision_variables[individual][dim] > nsga2->upper_bounds[dim]) nsga2->decision_variables[individual][dim] = nsga2->upper_bounds[dim];
        }
    }

    // integer mutation - only for integer decision variables
    for (unsigned dim = nsga2->d_dim - nsga2->i_dim; dim < nsga2->d_dim; ++dim) {
        std::uniform_int_distribution<int> dim_dist((int)nsga2->lower_bounds[dim], (int)nsga2->upper_bounds[dim]);
        nsga2->decision_variables[individual][dim] = (double)dim_dist(nsga2->random);
    }
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
void nsga2_serial_select_best_individuals(struct nsga2_serial<POP_SIZE, D_DIM, F_DIM> *nsga2) {
    // run fnds and move individuals front-wise to the best vector only if an entire front fits in it
    nsga2_serial_fast_non_dominated_sorting(nsga2, 2u * nsga2->pop_size);

    unsigned last_front_idx = 0u, already_selected = 0u;
    for (unsigned current_front_idx = 0u; current_front_idx < 2u * nsga2->pop_size; ++current_front_idx) {
        if (already_selected + nsga2->non_dominated_front[current_front_idx].size <= nsga2->pop_size) { // if an entire front fits in the remaining selected population
            for (unsigned i = 0u; i < nsga2->non_dominated_front[current_front_idx].size; ++i) {  // move all individuals from the front to the selected population
                memcpy(nsga2->selected_decision_variables[already_selected + i], nsga2->decision_variables[nsga2->non_dominated_front[current_front_idx].elements[i]], nsga2->d_dim * sizeof(double));
                memcpy(nsga2->selected_fitness_values[already_selected + i], nsga2->fitness_values[nsga2->non_dominated_front[current_front_idx].elements[i]], nsga2->f_dim * sizeof(double));
            }
            already_selected += nsga2->non_dominated_front[current_front_idx].size;
            if (already_selected == nsga2->pop_size) {
                return; // if the selected population is exactly filled we can return early
            }
            ++last_front_idx;
        } else {
            break; // if we cant fit next front into selected population we move onto crowding distance based selection
        }
    }

    // calculate crowding distance for the last front
    for (unsigned i = 0u; i < nsga2->non_dominated_front[last_front_idx].size; ++i) {
        nsga2->crowding_distance[i] = 0u;
    }
    nsga2_serial_crowding_distance(nsga2, last_front_idx);

    // sort individuals in the last front based on calculated crowding distance
    nsga2->crowding_distance_order.size = nsga2->non_dominated_front[last_front_idx].size;
    for (unsigned i = 0u; i < nsga2->non_dominated_front[last_front_idx].size; ++i) {
        nsga2->crowding_distance_order.elements[i] = nsga2->non_dominated_front[last_front_idx].elements[i];
    }
    // TODO: implement non std::sort solution
    std::sort(nsga2->crowding_distance_order.elements, nsga2->crowding_distance_order.elements + nsga2->crowding_distance_order.size, [nsga2](unsigned idx1, unsigned idx2) {
        return nsga2->crowding_distance[idx1] > nsga2->crowding_distance[idx2];
    });

    // fill up remaining space in selected population based on largest crowding distance
    for (unsigned i = 0u; i < nsga2->pop_size - already_selected; ++i) {
        memcpy(nsga2->selected_decision_variables[already_selected + i], nsga2->decision_variables[nsga2->crowding_distance_order.elements[i]], nsga2->d_dim * sizeof(double));
        memcpy(nsga2->selected_fitness_values[already_selected + i], nsga2->fitness_values[nsga2->crowding_distance_order.elements[i]], nsga2->f_dim * sizeof(double));
    }
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
void nsga2_serial_evolve(struct nsga2_serial<POP_SIZE, D_DIM, F_DIM> *nsga2) {
    for (unsigned gen = 1u; gen <= nsga2->gen; ++gen) {
        // prepare indexes for tournament selection
        nsga2_serial_shuffle_selection(nsga2);

        // calculate non dominated fronts and ranks
        nsga2_serial_fast_non_dominated_sorting(nsga2, nsga2->pop_size);

        // calculate crowding distance
        for (unsigned i = 0u; i < 2u * nsga2->pop_size; ++i) nsga2->crowding_distance[i] = 0u; // reset crowding distance values before calculating them again
        for (unsigned current_front_idx = 0u; current_front_idx < nsga2->pop_size; ++current_front_idx) {
            if (nsga2->non_dominated_front[current_front_idx].size == 0u) {
                break;
            } else if (nsga2->non_dominated_front[current_front_idx].size == 1u) {
                nsga2->crowding_distance[nsga2->non_dominated_front[current_front_idx].elements[0]] = INFINITY;
            } else if (nsga2->non_dominated_front[current_front_idx].size == 2u) {
                nsga2->crowding_distance[nsga2->non_dominated_front[current_front_idx].elements[0]] = INFINITY;
                nsga2->crowding_distance[nsga2->non_dominated_front[current_front_idx].elements[1]] = INFINITY;
            } else {
                nsga2_serial_crowding_distance(nsga2, current_front_idx);
            }
        }

        // selection, crossover and mutation (generate N children so that the population will reach 2N size)
        for (unsigned i = 0u; i < nsga2->pop_size / 2u; ++i) {
            // calculate index to select parents from shuffle and store new children
            unsigned selection_idx = i * 4u;
            unsigned child1_idx = nsga2->pop_size + i * 2u;
            unsigned child2_idx = child1_idx + 1u;
            // select two parents through tournament selection from 4 random individuals
            unsigned parent1_idx = nsga2_serial_tournament_selection(nsga2, nsga2->shuffle[selection_idx], nsga2->shuffle[selection_idx + 1]);
            unsigned parent2_idx = nsga2_serial_tournament_selection(nsga2, nsga2->shuffle[selection_idx + 2], nsga2->shuffle[selection_idx + 3]);
            // generate two new children by crossover of selected parents
            nsga2_serial_simulated_binary_crossover(nsga2, parent1_idx, parent2_idx, child1_idx, child2_idx);
            // run mutation on both of new children
            nsga2_serial_polynomial_mutation(nsga2, child1_idx);
            nsga2_serial_polynomial_mutation(nsga2, child2_idx);
            // calculate fitness vectors for new children
            nsga2->fitness(nsga2->decision_variables[child1_idx], nsga2->fitness_values[child1_idx], nsga2->d_dim, nsga2->f_dim);
            nsga2->fitness(nsga2->decision_variables[child2_idx], nsga2->fitness_values[child2_idx], nsga2->d_dim, nsga2->f_dim);
        }

        // select best N individuals from 2*N population
        nsga2_serial_select_best_individuals(nsga2);

        // move selected individuals into the new population
        memcpy(nsga2->decision_variables, nsga2->selected_decision_variables, sizeof(double) * nsga2->d_dim * nsga2->pop_size);
        memcpy(nsga2->fitness_values, nsga2->selected_fitness_values, sizeof(double) * nsga2->f_dim * nsga2->pop_size);
    }
}

// pmoo wrappers

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
inline void pmoo_nsga2_serial_load_problem(void *self, Problem* problem) {
    struct nsga2_serial<POP_SIZE, D_DIM, F_DIM> *nsga2 = (struct nsga2_serial<POP_SIZE, D_DIM, F_DIM>*)self;
    nsga2->d_dim = problem->d_dim;
    nsga2->f_dim = problem->f_dim;
    nsga2->i_dim = problem->i_dim;
    memcpy(nsga2->lower_bounds, problem->lower_bounds, sizeof(double) * problem->d_dim);
    memcpy(nsga2->upper_bounds, problem->upper_bounds, sizeof(double) * problem->d_dim);
    nsga2->fitness = problem->fitness;
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
inline void pmoo_nsga2_serial_load_parameters(void *self, Parameters* parameters) {
    struct nsga2_serial<POP_SIZE, D_DIM, F_DIM> *nsga2 = (struct nsga2_serial<POP_SIZE, D_DIM, F_DIM>*)self;

    // make sure that metadata is set
    nsga2->pop_size = POP_SIZE;
    nsga2->d_dim = D_DIM;
    nsga2->f_dim = F_DIM;

    // initialize algorithm settings
    if (parameters) {
        nsga2->gen = parameters->gen;
        nsga2->p_crossover = parameters->p_crossover;
        nsga2->eta_crossover = parameters->eta_crossover;
        nsga2->p_mutation = parameters->p_mutation;
        nsga2->eta_mutation = parameters->eta_mutation;
    } else {
        nsga2->gen = 100u;
        nsga2->p_crossover = 0.95;
        nsga2->eta_crossover = 10.0;
        nsga2->p_mutation = 0.01;
        nsga2->eta_mutation = 50.0;
    }
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
inline void pmoo_nsga2_serial_initialize(void *self, void *gpu_self) {
    nsga2_serial_initialize((struct nsga2_serial<POP_SIZE, D_DIM, F_DIM>*)self);
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
inline void pmoo_nsga2_serial_evolve(void *self, void *gpu_self) {
    nsga2_serial_evolve((struct nsga2_serial<POP_SIZE, D_DIM, F_DIM>*)self);
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
inline void pmoo_nsga2_serial_cleanup(void *self, void *gpu_self) {
    return;
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
inline double** pmoo_nsga2_serial_decision_variables(void *self) {
    struct nsga2_serial<POP_SIZE, D_DIM, F_DIM> *nsga2 = (struct nsga2_serial<POP_SIZE, D_DIM, F_DIM>*)self;

    double **population = (double**)malloc(POP_SIZE * sizeof(double*));
    for (unsigned i = 0u; i < POP_SIZE; ++i)
        population[i] = nsga2->decision_variables[i];
    
    return population;
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
inline double** pmoo_nsga2_serial_fitness_values(void *self) {
    struct nsga2_serial<POP_SIZE, D_DIM, F_DIM> *nsga2 = (struct nsga2_serial<POP_SIZE, D_DIM, F_DIM>*)self;

    double **population = (double**)malloc(POP_SIZE * sizeof(double*));
    for (unsigned i = 0u; i < POP_SIZE; ++i)
        population[i] = nsga2->fitness_values[i];
    
    return population;
}

template <unsigned POP_SIZE, unsigned D_DIM, unsigned F_DIM>
Algorithm* pmoo_nsga2_serial_generate() {
    Algorithm *algorithm = (Algorithm*)malloc(sizeof(Algorithm));

    algorithm->name = NSGA2;
    algorithm->variant = SERIAL;
    algorithm->size = sizeof(struct nsga2_serial<POP_SIZE, D_DIM, F_DIM>);
    algorithm->self = malloc(algorithm->size);
    algorithm->gpu_self = NULL;
    algorithm->population_size = POP_SIZE;
    algorithm->decision_variables = pmoo_nsga2_serial_decision_variables<POP_SIZE, D_DIM, F_DIM>(algorithm->self);
    algorithm->fitness_values = pmoo_nsga2_serial_fitness_values<POP_SIZE, D_DIM, F_DIM>(algorithm->self);
    algorithm->load_problem = pmoo_nsga2_serial_load_problem<POP_SIZE, D_DIM, F_DIM>;
    algorithm->load_parameters = pmoo_nsga2_serial_load_parameters<POP_SIZE, D_DIM, F_DIM>;
    algorithm->initialize = pmoo_nsga2_serial_initialize<POP_SIZE, D_DIM, F_DIM>;
    algorithm->evolve = pmoo_nsga2_serial_evolve<POP_SIZE, D_DIM, F_DIM>;
    algorithm->cleanup = pmoo_nsga2_serial_cleanup<POP_SIZE, D_DIM, F_DIM>;

    return algorithm;
}

#endif
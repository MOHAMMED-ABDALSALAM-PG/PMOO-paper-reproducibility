# pmoo design and details

## overview

The project consists of 4 subprojects:

- metrics
- problems
- algorithms
- benchmark

Each algorithm is implemented in 4 variants:

- serial - reference implementation without any parallelization
- cpu - implementation utilizing OpenMP
- gpu - implementation utilizing CUDA
- hybrid - implementation utilizing OpenMP and CUDA

Here are some abbreviations that we are using in the codebase and later in these notes:

- M / F_DIM - number of objective functions/variables
- N / POP_SIZE - population size
- n / D_DIM - number of decision variables
- genetic - a seperate repository with generic genetic algorithm in 3 variants: serial, cpu and fully gpu
- fully gpu - an implementation using CUDA, that is entirely computed on the device - a single memcpy before starting the algorithm, then running computations completely through kernels and copying over results back to the host - no memcpy during the algorithm execution and no computations on cpu apart from calling kernel functions

Our project utilizes mostly C, but has some features of C++. Due to using C, we had to workaround OOP features using structs and ambigous pointers to functions and structs. We also wanted the project to be easily extendable by providing new test problems or algorithm variants. The definitions for the "interfaces" are provided in these files and we would suggest to start familiarizing with the code in these spots:

- `pmoo/include/problems/problem.cuh` - test suites definitions and Problem struct/interface definition,
- `pmoo/include/algorithms/algorithm.cuh` - Algorithm struct/interface definition,

## metrics

The _metrics_ subproject contains implementation for 3 MOO metrics: Generational Distance (GD), Inverted Generational Distance (IGD) and Hypervolume (HV).

GD and IGD are very similar to each other and follow the original paper formulas. They both utilize OpenMP to make calculations faster, specially on larger population or front sizes. Both require the pareto front population (a vector of vectors in objective spaces) and the tested population (also a vector of vectors in objective spaces).

HV has been implemented using Monte Carlo approach. There are other implementations but they work up to limited number of dimensions and are generally more difficult to implement. In Monte Carlo HV, a randomized point (sample) is tested whether it is pareto dominated by at least one point in the population. The hypervolume is calculated as a ratio of samples that are dominated to the total number of samples. It also utilizes OpenMP to make calculations faster. To achieve decent accuracy, you need to run it for many samples, which requirements increases with number of dimensions. HV requires tested population (a vector of vectors in objective spaces) and a reference point to bound the hypervolume in. Our HV implementation estimates hypervolume inside a space bounded by (0.0, 0.0, ...) and the provided reference point.

## problems

Our project contains two test suites: ZDT and DTLZ.

ZDT is a test suite that contains 6 problems, but we excluded ZDT5, because it has different number of decision variables and is an integer problem. Integer problems require a different handling by the algorithm, specially in crossover and mutation. Our code has some features to handle integer problems, but we havent tested it. The entire test suite only works with F_DIM = 2. It could work with F_DIM > 2, but the rest of the variables wouldnt be utilized.

DTLZ is a many objective test suite, which means it can work with any number of decision and fitness dimensions. It contains 7 test problems.

Each problem struct contains problem metadata, fitness function, pareto front and metric metadata. A fitness function is a function that takes a set of decision variables and calculates fitness values. Each fitness evaluation is done on each individual in the population. There is also gpu fitness, which is a function annotated with \_\_device\_\_ so it can be called inside a CUDA kernel. In order to pass gpu function address you need to first copy it using cudaMemcpyFromSymbol (it is properly explained in the comments in the code). Each problem also contains generated pareto front with metadata such as front size and number of samples for hypervolume. Pareto fronts are calculated by generating optimal solutions in the decision space (optimal solutions are given in the original papers) and then using fitness functions, the pareto front in the fitness space is calculated and saved. Pareto front size increases exponentially with fitness dimensions (curse of dimensionality), so we had to limit the density of the pareto front on larger F_DIM.

There is an empty boiler plate code to implement WFG test suite.

## algorithms

The most important function for each algorithm is the evolve function. The evolve function is the actual algorithm implementation, but each algorithm also has a set of helper functions for initialization, loading problems and etc. (its explained in the comments in the code). We took a very specific approach in implementing this. We are using templates so that each algorithm has a contignous struct that contains all required memory to execute the algorithm. This way it can be instantiated using a single malloc and can be copied over to the device using a single cudamemcpy. It allows us to use fixed-length arrays inside, making the implementation easier and faster. It comes with a drawback, that it has to be fully implemented inside a header file. The header file has to be included wholly, making the compilation longer and the executable larger. It is completely unusable as a library - it loses any flexibility for the ease and decreased time of implementation, specially when you have to implement multiple variants that differ a lot.

There is a leftover code to support a fully gpu implementation. 

## benchmark

Our benchmark executable is meant to take measurements in the most automated way possible. After each test, the measurements are being saved to a .csv file in order to not lose any measurements. Even if the tests were interrupted, you can easily modify the benchmark.cu to resume the tests at a specific algorithm variant.

## suggestions

- A fully gpu implementation would be worth exploring. The genetic repository shows that it can result in better performance than OpenMP. NSGA2 could be fully gpu implemented if not for the fast non dominated sorting. We couldnt get it to work fast enough. We tried a single block kernel and some attempt at a custom front construction, but none of these worked. SPEA2 shows promise to be implemented fully on the gpu - refer to the PlatEMO implementation of SPEA2, it has a different approach to truncation, which is more suitable for gpu than our implementation. In general, algorithms based on genetic/evolutionary algorithm can be fully done on the gpu (as shown in the genetic repository), but the selection mechanisms of an algorithm have to be suitable for the gpu. All the other steps can be easily done on gpu. It is also worth noting the difference between tournament with and without replacement. The difference is explained in our thesis paper - the one without replacement can be more easily done on the gpu (as shown in the genetic repository), but tournament with replacement might be possible on the gpu using either parallel fisher-yates shuffle or linear congruential generator.

- All evolutionary algorithms will have similar performance gains from parallelization. It might be worth exploring to what extent different aproaches can be parallelized.

- In our tests the fitness function were pretty simple and took little time to compute. The parallelization would be probably more beneficial if the fitness function contained more computations. The gpu implementation could show incredible results if each thread in the grid could calculate a long running fitness function.

- We are aware that our codebase isnt that flexible outside of the scope of this specific benchmark app. We would recommend either conforming to our design and extending the research by providing another algorithms, problems and metrics or completely ditching this project, starting a new one and using our implementations as reference (or literally copying over implementations for the algorithms).

- The cmake is a difficult environment to work with if someone is not familiar with it. We needed to make OpenMP and CUDA to work together in a single compiled executable so that the tests could be maximally automated. Maybe the approach presented in the genetic repository using makefile is the way to go. It takes similar time to setup, it has separate build processes and the automation could be done through shell scripts.
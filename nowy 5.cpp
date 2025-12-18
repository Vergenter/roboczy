#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <utility>
#include <cmath>
#include <limits>
#include <type_traits>

//###########################
template<typename T> concept Scalar = std::is_arithmetic_v<T>;

template<typename T> concept VectorLike = requires(T a, T b, std::size_t i) {
        { a.size() } -> std::convertible_to<std::size_t>;
        { a[i] }     -> Scalar;
        { a + b }    -> std::same_as<T>;
        { a - b }    -> std::same_as<T>;
        { a * 1.0 }  -> std::same_as<T>;};
		
template<typename T> concept Gene = Scalar<T> || VectorLike<T>;

template<Gene T> struct GeneOps;

template<Scalar T>
struct GeneOps<T> {
    static T add(const T& a, const T& b) { return a + b; }
    static T mul(const T& a, double s) { return static_cast<T>(a * s); }

    static T random_delta(const T& /*sample*/, std::mt19937& rng, double intensity) {
    std::uniform_real_distribution<double> d(-intensity, intensity);
    return static_cast<T>(d(rng));}


    static double to_scalar(const T& x) {
        return static_cast<double>(x);
    }
};

template<VectorLike T>
struct GeneOps<T> {
    static T add(const T& a, const T& b) {
        T r = a;
        for (std::size_t i = 0; i < r.size(); ++i)
            r[i] += b[i];
        return r;
    }

    static T mul(const T& a, double s) {
        T r = a;
        for (auto& v : r) v *= s;
        return r;
    }

    static T random_delta(const T& sample, std::mt19937& rng, double intensity) {
        T r = sample;
        std::uniform_real_distribution<double> d(-intensity, intensity);
        for (auto& v : r) v = d(rng);
        return r;
    }

    static double to_scalar(const T& x) {
        return std::accumulate(x.begin(), x.end(), 0.0) / x.size();
    }
};

//#############################

//Polityki inicjalizacji
template<typename Type>
class RandomInitiationPolicy {
public:
    RandomInitiationPolicy(Type min, Type max) : min_(min), max_(max) {}

    void init(std::vector<Type>& population, std::size_t size, std::mt19937& rng) const {
        population.resize(size);
        if constexpr (std::is_integral_v<Type>) {
            std::uniform_int_distribution<Type> dist(min_, max_);
            for (auto& x : population) x = dist(rng);
        } else {
            std::uniform_real_distribution<Type> dist(min_, max_);
            for (auto& x : population) x = dist(rng);
        }
    }

private:
    Type min_;
    Type max_;
};


template<typename Type>
class LinSpaceInitiationPolicy {
public:
    LinSpaceInitiationPolicy(Type min, Type max) : min_(min), max_(max) {}

    void init(std::vector<Type>& population, std::size_t size, std::mt19937&) const {
        population.resize(size);
        if (size == 0) return;
        if (size == 1) { population[0] = min_; return; }

        using CT = std::common_type_t<Type, double>;
        CT step = (static_cast<CT>(max_) - static_cast<CT>(min_)) / static_cast<CT>(size - 1);
        for (std::size_t i = 0; i < size; ++i)
            population[i] = static_cast<Type>(static_cast<CT>(min_) + i * step);
    }

private:
    Type min_;
    Type max_;
};



//Polityki mutacji
template<typename Type, int CHANCE, int INTENSITY>
class PercentageMutationPolicy {
public:
    static_assert(CHANCE >= 0 && CHANCE <= 100,"Value must be >=0 and <= 100" );
	static_assert(INTENSITY > 0, "Value must be >0");

    void mutate(std::vector<Type>& population, std::mt19937& rng) const {
        std::uniform_int_distribution<int> prob(0, 99);
        std::uniform_real_distribution<double> factor(
            1.0 - INTENSITY / 100.0,
            1.0 + INTENSITY / 100.0);

        for (auto& x : population)
            if (prob(rng) < CHANCE)
                x = GeneOps<Type>::mul(x, factor(rng));;
    }
};

template<Gene Type, int CHANCE, double INTENSITY>
class AbsoluteMutationPolicy {
public:
	static_assert(CHANCE >= 0 && CHANCE <= 100, "Value must be >=0 and <= 100");
	static_assert(INTENSITY > 0, "Value must be >0");
    void mutate(std::vector<Type>& population, std::mt19937& rng) const {
        std::uniform_int_distribution<int> prob(0, 99);

        for (auto& x : population)
            if (prob(rng) < CHANCE)
                x = GeneOps<Type>::add(
                        x,
                        GeneOps<Type>::random_delta(x, rng, INTENSITY)
                    );
    }
};


//Polityki krzyżowania  
template<typename Type, double WEIGHT>
class AverageCrossoverPolicy {
public:
   static_assert(WEIGHT >= 0 && WEIGHT <= 1, "Value must be >=0 and <= 1");

    Type crossover(const Type& a, const Type& b, std::mt19937&) const {
        return GeneOps<Type>::add(
            GeneOps<Type>::mul(a, WEIGHT),
            GeneOps<Type>::mul(b, 1.0 - WEIGHT)
        );
    }
};

template<typename Type>
class RandomCrossoverPolicy {
public:
   
    Type crossover(const Type& a, const Type& b, std::mt19937& rng) const {
        std::uniform_real_distribution<double> w(0.0, 1.0);
        double weight = w(rng);
        return GeneOps<Type>::add(GeneOps<Type>::mul(a, weight),
            GeneOps<Type>::mul(b, 1.0 - weight)
        );
    }
};

//Polityki selekcji 
template<typename Type>
class RandomSelectionPolicy {
public:
    std::pair<Type, Type> select(const std::vector<Type>& population, std::mt19937& rng) const {
        std::uniform_int_distribution<std::size_t> dist(0, population.size() - 1);
        return { population[dist(rng)], population[dist(rng)] };
    }
};

template<typename Type>
class UniqueRandomSelectionPolicy {
public:
    std::pair<Type, Type> select(const std::vector<Type>& population, std::mt19937& rng) {
        if (pairs.empty() || cachedSize != population.size()) {
            buildPairs(population.size(), rng);
            cachedSize = population.size();
        }

        auto [i, j] = pairs.back();
        pairs.pop_back();
        return { population[i], population[j] };
    }

private:
    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    std::size_t cachedSize = 0;

    void buildPairs(std::size_t n, std::mt19937& rng) {
        pairs.clear();
        pairs.reserve(n * (n - 1));

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                pairs.emplace_back(i, j);
                pairs.emplace_back(j, i);
            }
        }

        std::shuffle(pairs.begin(), pairs.end(), rng);
    }
};


template<typename Type, int FIRST, int LAST, typename Fitness>
class TargetSelectionPolicy {
public:
    static_assert(FIRST >= 0 && FIRST <= 100, "Value must be >=0 and <= 100");
    static_assert(LAST  >= 0 && LAST  <= 100, "Value must be >=0 and <= 100");

    explicit TargetSelectionPolicy(Fitness fit) : fit(fit) {}

    std::pair<Type, Type> select(const std::vector<Type>& population, std::mt19937& rng) const {
        const std::size_t n = population.size();

        std::vector<std::size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);

        std::sort(idx.begin(), idx.end(),
            [&](std::size_t a, std::size_t b) {
                return fit(population[a]) > fit(population[b]);
            });

        std::vector<double> probs(n);
        if (n == 1) {
            probs[0] = 1.0;
        } else {
            double first = FIRST / 100.0;
            double last  = LAST  / 100.0;
            double step  = (first - last) / static_cast<double>(n - 1);

            for (std::size_t i = 0; i < n; ++i)
                probs[i] = first - i * step;
        }

        std::discrete_distribution<std::size_t> dist(probs.begin(), probs.end());

        return {
            population[idx[dist(rng)]],
            population[idx[dist(rng)]]
        };
    }

private:
    Fitness fit;
};


//Polityki warunku stopu
template<typename Type, std::size_t PARAM>
class MaxGenStopConditionPolicy {
public:
    bool shouldStop(const std::vector<Type>&, std::size_t generation) const {
        return generation >= PARAM;
    }
};

template<typename Type, double PARAM>
class StableAvgStopConditionPolicy {
public:
    bool shouldStop(const std::vector<Type>& population, std::size_t) {
        double avg = 0.0;
        for (auto& x : population)
            avg += GeneOps<Type>::to_scalar(x);
        avg /= population.size();

        if (initialized && std::abs(avg - lastAvg) < PARAM) {
            if (++stableCount > 2)
                return true;
        } else {
            stableCount = 0;
        }

        lastAvg = avg;
        initialized = true;
        return false;
    }

private:
    double lastAvg = 0.0;
    std::size_t stableCount = 0;
    bool initialized = false;
};


//Algorytm ewolucyjny

template<Gene Type, typename InitPolicy, typename MutationPolicy,
         typename CrossoverPolicy, typename SelectionPolicy, typename StopPolicy>
class EvolutionaryAlgorithm {
public:
    EvolutionaryAlgorithm(
        std::size_t populationSize,
        SelectionPolicy selection,
        InitPolicy init = InitPolicy{},
        MutationPolicy mutation = MutationPolicy{},
        CrossoverPolicy crossover = CrossoverPolicy{},
        StopPolicy stop = StopPolicy{},
        std::mt19937 rng = std::mt19937{std::random_device{}()})
        : populationSize(populationSize),
          initPolicy(std::move(init)),
          mutationPolicy(std::move(mutation)),
          crossoverPolicy(std::move(crossover)),
          selection(std::move(selection)),
          stopPolicy(std::move(stop)),
          rng(rng)
    {
        initPolicy.init(population, populationSize, this->rng);
    }

    void run() {
        std::size_t generation = 0;
        while (!stopPolicy.shouldStop(population, generation)) {
            std::vector<Type> newPopulation;
            newPopulation.reserve(populationSize);

            for (std::size_t i = 0; i < populationSize; ++i) {
                auto [p1, p2] = selection.select(population, rng);
                Type child = crossoverPolicy.crossover(p1, p2, rng);
                newPopulation.push_back(child);
            }

            population = std::move(newPopulation);
            mutationPolicy.mutate(population, rng);

        std::cout << "Generacja " << generation << ": ";
        for (const auto& x : population)
            std::cout << std::round(x* 1000.0) / 1000.0 << ' ';
        std::cout << '\n';

        ++generation;
    }
}

private:
    std::size_t populationSize{};
    std::vector<Type> population;
    InitPolicy initPolicy;
    MutationPolicy mutationPolicy;
    CrossoverPolicy crossoverPolicy;
    SelectionPolicy selection;
    StopPolicy stopPolicy;
    std::mt19937 rng;
};


//###############
template<Scalar T>
double fitness(const T& x) {
    return -std::abs(x);
}

template<VectorLike T>
double fitness(const T& x) {
    double sum = 0.0;
    for (auto v : x) sum += v * v;
    return -std::sqrt(sum);
}
//#################

int main() {
	std::mt19937 rng(42);
	auto fit = [](const auto& x) {return fitness(x);};
	
    using Sel = TargetSelectionPolicy<double, 50, 5, decltype(fit)>; 
	Sel sel(fit);


    EvolutionaryAlgorithm<double,
    LinSpaceInitiationPolicy<double>,
    AbsoluteMutationPolicy<double, 20, 10.0>,
    AverageCrossoverPolicy<double, 1.0>,
    
    //RandomSelectionPolicy<double>,
    decltype(sel),
    
    StableAvgStopConditionPolicy<double, 0.5> > algo(20,
    sel,
    //{},
	
	LinSpaceInitiationPolicy<double>(-10.0, 10.0),
     {}, {}, {},   // domyślne polityki
    rng);

    algo.run();}



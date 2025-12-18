#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <utility>
#include <cmath>
#include <limits>
#include <type_traits>

class RNG {
public:
    explicit RNG(unsigned seed = std::random_device{}()) : gen(seed) {}
    std::mt19937& engine() { return gen;}
private:
    std::mt19937 gen;};

//Polityki inicjalizacji
template<typename Type, Type MIN, Type MAX>
class RandomInitiationPolicy {
public:
    void init(std::vector<Type>& population, std::size_t size, RNG& rng) const {
        population.resize(size);
        if constexpr (std::is_integral_v<Type>) {
            std::uniform_int_distribution<Type> dist(MIN, MAX);
            for (auto& x : population) x = dist(rng.engine());
        } else {
            std::uniform_real_distribution<Type> dist(MIN, MAX);
            for (auto& x : population) x = dist(rng.engine());
        }
    }
};

template<typename Type, Type MIN, Type MAX>
class LinSpaceInitiationPolicy {
public:

    void init(std::vector<Type>& population, std::size_t size, RNG&) const {
        population.resize(size);
        if (size == 0) return;
        if (size == 1) { population[0] = MIN; return; }

        using CT = std::common_type_t<Type, double>;
        CT step = (static_cast<CT>(MAX) - static_cast<CT>(MIN)) / static_cast<CT>(size - 1);
        for (std::size_t i = 0; i < size; ++i)
            population[i] = static_cast<Type>(static_cast<CT>(MIN) + i * step);
    }
};

//Polityki mutacji
template<typename Type, int CHANCE, int INTENSITY>
class PercentageMutationPolicy {
public:
    static_assert(CHANCE >= 0 && CHANCE <= 100,"Value must be >=0 and <= 100" );
	static_assert(INTENSITY > 0, "Value must be >0");

    void mutate(std::vector<Type>& population, RNG& rng) const {
        std::uniform_int_distribution<int> prob(0, 99);
        std::uniform_real_distribution<double> factor(
            1.0 - INTENSITY / 100.0,
            1.0 + INTENSITY / 100.0
        );

        for (auto& x : population)
            if (prob(rng.engine()) < CHANCE)
                x = static_cast<Type>(x * factor(rng.engine()));
    }
};

template<typename Type, int CHANCE, Type INTENSITY>
class AbsoluteMutationPolicy {
public:
    static_assert(CHANCE >= 0 && CHANCE <= 100, "Value must be >=0 and <= 100");
	static_assert(INTENSITY > 0, "Value must be >0");

    void mutate(std::vector<Type>& population, RNG& rng) const {
        std::uniform_int_distribution<int> prob(0, 99);

        if constexpr (std::is_integral_v<Type>) {
            std::uniform_int_distribution<Type> delta(-INTENSITY, INTENSITY);
            for (auto& x : population)
                if (prob(rng.engine()) < CHANCE)
                    x += delta(rng.engine());
        } else {
            std::uniform_real_distribution<Type> delta(-INTENSITY, INTENSITY);
            for (auto& x : population)
                if (prob(rng.engine()) < CHANCE)
                    x += delta(rng.engine());
        }
    }
};

//Polityki krzyżowania  
template<typename Type, double WEIGHT>
class AverageCrossoverPolicy {
public:
   static_assert(WEIGHT >= 0 && WEIGHT <= 1, "Value must be >=0 and <= 1");

    Type crossover(const Type& a, const Type& b, RNG&) const {
        return static_cast<Type>(WEIGHT * a + (1.0 - WEIGHT) * b);
    }
};

template<typename Type>
class RandomCrossoverPolicy {
public:
   
    Type crossover(const Type& a, const Type& b, RNG& rng) const {
        std::uniform_real_distribution<double> w(0.0, 1.0);
        double weight = w(rng.engine());
        return static_cast<Type>(weight * a + (1.0 - weight) * b);
    }
};

//Polityki selekcji 
template<typename Type>
class RandomSelectionPolicy {
public:
    std::pair<Type, Type> select(const std::vector<Type>& population, RNG& rng) const {
        std::uniform_int_distribution<std::size_t> dist(0, population.size() - 1);
        return { population[dist(rng.engine())], population[dist(rng.engine())] };
    }
};

template<typename Type>
class UniqueRandomSelectionPolicy {
public:
    std::pair<Type, Type> select(const std::vector<Type>& population, RNG& rng) {
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

    void buildPairs(std::size_t n, RNG& rng) {
        pairs.clear();
        pairs.reserve(n * (n - 1));

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                pairs.emplace_back(i, j);
                pairs.emplace_back(j, i);
            }
        }

        std::shuffle(pairs.begin(), pairs.end(), rng.engine());
    }
};


template<typename Type, typename Fitness, int FIRST, int LAST>
class TargetSelectionPolicy {
public:
    static_assert(FIRST >= 0 && FIRST <= 100, "Value must be >=0 and <= 100");
    static_assert(LAST  >= 0 && LAST  <= 100, "Value must be >=0 and <= 100");

    explicit TargetSelectionPolicy(Fitness fit) : fit(fit) {}

    std::pair<Type, Type> select(const std::vector<Type>& population, RNG& rng) const {
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
            population[idx[dist(rng.engine())]],
            population[idx[dist(rng.engine())]]
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
        double avg = std::accumulate(population.begin(), population.end(), 0.0)
                     / population.size();

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

///////////////////////////////
template<typename T>
concept Genome =
    std::is_copy_constructible_v<T> && std::is_move_constructible_v<T>;




template<typename Type, typename InitPolicy, typename MutationPolicy,
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
        RNG rng = RNG{}
    )
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
    RNG rng;
};



double fitness(double x) { return -(x - 3) * (x - 3); }

int main() {
    auto sel = TargetSelectionPolicy<double, decltype(&fitness), 50, 5>(&fitness);

    EvolutionaryAlgorithm<
    double,
    LinSpaceInitiationPolicy<double, -10.0, 10.0>,
    AbsoluteMutationPolicy<double, 20, 10.0>,
    AverageCrossoverPolicy<double, 1.0>,
    
    //RandomSelectionPolicy<double>,
    decltype(sel),
    
    StableAvgStopConditionPolicy<double, 0.5> > algo(20,
    sel,
    //{},
	
    {}, {}, {}, {},   // domyślne polityki
    RNG(42)
);

    algo.run();

}



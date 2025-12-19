// Adapted from your pasted code: removes GeneOps<> but keeps all policies,
// by replacing GeneOps calls with small free functions (scalar + vector-like).
// :contentReference[oaicite:0]{index=0}

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <utility>
#include <cmath>
#include <limits>
#include <type_traits>
#include <concepts>
#include <iterator>
#include <functional>
#include <cassert>
// ===================== Concepts =====================

template<typename T>
concept Scalar = std::is_arithmetic_v<std::remove_cvref_t<T>>;

template<typename T>
concept VectorLike =
  requires(T a) {
    std::begin(a);
    std::end(a);
  } && Scalar<std::remove_cvref_t<decltype(*std::begin(std::declval<T&>()))>>;

template<typename T>
concept Gene = Scalar<T> || VectorLike<T>;


namespace gene {

template<class U>
U random_between(const U& mn, const U& mx, std::mt19937& rng) requires Scalar<U> {
  if constexpr (std::is_integral_v<U>) {
    std::uniform_int_distribution<U> d(mn, mx);
    return d(rng);
  } else {
    std::uniform_real_distribution<double> d((double)mn, (double)mx);
    return static_cast<U>(d(rng));
  }
}

template<class U>
U random_between(const U& mn, const U& mx, std::mt19937& rng) requires VectorLike<U> {
  U r = mn; // uses mn as shape carrier; no size() call needed
  auto ri = std::begin(r);
  auto mi = std::begin(mn);
  auto ma = std::begin(mx);
  for (; ri != std::end(r); ++ri, ++mi, ++ma) {
    using Elem = std::remove_cvref_t<decltype(*ri)>;
    *ri = random_between<Elem>(*mi, *ma, rng);
  }
  return r;
}

template<class U>
U random_delta(const U&, std::mt19937& rng, double intensity) requires Scalar<U> {
  if constexpr (std::is_integral_v<U>) {
    // integer delta: symmetric range [-I, I]
    const auto I = static_cast<U>(std::llround(intensity));
    std::uniform_int_distribution<long long> d(-static_cast<long long>(I), static_cast<long long>(I));
    return static_cast<U>(d(rng));
  } else {
    std::uniform_real_distribution<double> d(-intensity, intensity);
    return static_cast<U>(d(rng));
  }
}

template<class U>
U random_delta(const U& sample, std::mt19937& rng, double intensity) requires VectorLike<U> {
  U r = sample; // keeps shape; overwrites with deltas
  std::uniform_real_distribution<double> d(-intensity, intensity);
  for (auto& v : r)
    v = static_cast<std::remove_cvref_t<decltype(v)>>(d(rng));
  return r;
}

template<class U>
U add(const U& a, const U& b) requires Scalar<U> {
  return a + b;
}

template<class U>
U add(const U& a, const U& b) requires VectorLike<U> {
  U r = a;
  auto ri = std::begin(r);
  auto bi = std::begin(b);
  for (; ri != std::end(r); ++ri, ++bi) *ri += *bi;
  return r;
}

template<class U>
U mul(const U& a, double s) requires Scalar<U> {
  return static_cast<U>(a * s);
}

template<class U>
U mul(const U& a, double s) requires VectorLike<U> {
  U r = a;
  for (auto& v : r) v = static_cast<std::remove_cvref_t<decltype(v)>>(v * s);
  return r;
}

template<class U>
double to_scalar(const U& x) requires Scalar<U> {
  return static_cast<double>(x);
}

template<class U>
double to_scalar(const U& x) requires VectorLike<U> {
  double sum = 0.0;
  std::size_t n = 0;
  for (auto v : x) { sum += static_cast<double>(v); ++n; }
  return (n == 0) ? 0.0 : (sum / static_cast<double>(n));
}

}

template<typename Type>
class RandomInitiationPolicy {
public:
  RandomInitiationPolicy(const Type& min, const Type& max) : min_(min), max_(max) {}

  void init(std::vector<Type>& population, std::mt19937& rng) const {
    for (auto& x : population)
      x = gene::random_between(min_, max_, rng);
  }

private:
  Type min_;
  Type max_;
};

template<typename Type>
class LinSpaceInitiationPolicy {
public:
  LinSpaceInitiationPolicy(Type min, Type max) : min_(std::move(min)), max_(std::move(max)) {}

  // Scalar: evenly spaced in [min_, max_]
  void init(std::vector<Type>& population, std::mt19937&) const
    requires Scalar<Type>
  {
    const std::size_t n = population.size();
    if (n == 0) return;
    if (n == 1) { population[0] = min_; return; }

    using CT = std::common_type_t<Type, double>;
    CT step = (static_cast<CT>(max_) - static_cast<CT>(min_)) / static_cast<CT>(n - 1);
    for (std::size_t i = 0; i < n; ++i)
      population[i] = static_cast<Type>(static_cast<CT>(min_) + static_cast<CT>(i) * step);
  }

  // VectorLike: linear interpolation between min_ and max_ per element
  void init(std::vector<Type>& population, std::mt19937&) const
    requires VectorLike<Type>
  {
    const std::size_t n = population.size();
    if (n == 0) return;
    if (n == 1) { population[0] = min_; return; }

    // shape comes from min_
    for (std::size_t i = 0; i < n; ++i) {
      double t = static_cast<double>(i) / static_cast<double>(n - 1); // 0..1
      Type x = min_;
      auto xi = std::begin(x);
      auto mi = std::begin(min_);
      auto ma = std::begin(max_);
      for (; xi != std::end(x); ++xi, ++mi, ++ma) {
        using Elem = std::remove_cvref_t<decltype(*xi)>;
        double v = static_cast<double>(*mi) + (static_cast<double>(*ma) - static_cast<double>(*mi)) * t;
        *xi = static_cast<Elem>(v);
      }
      population[i] = std::move(x);
    }
  }

private:
  Type min_;
  Type max_;
};

// ===================== Mutation policies =====================

template<typename Type, int CHANCE, int INTENSITY>
class PercentageMutationPolicy {
public:
  static_assert(CHANCE >= 0 && CHANCE <= 100, "Value must be >=0 and <= 100");
  static_assert(INTENSITY > 0, "Value must be >0");

  void mutate(std::vector<Type>& population, std::mt19937& rng) const {
    std::uniform_int_distribution<int> prob(0, 99);
    std::uniform_real_distribution<double> factor(
      1.0 - INTENSITY / 100.0,
      1.0 + INTENSITY / 100.0
    );

    for (auto& x : population)
      if (prob(rng) < CHANCE)
        x = gene::mul(x, factor(rng));
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
        x = gene::add(x, gene::random_delta(x, rng, INTENSITY));
  }
};


template<typename Type, double WEIGHT>
class AverageCrossoverPolicy {
public:
  static_assert(WEIGHT >= 0 && WEIGHT <= 1, "Value must be >=0 and <= 1");

  Type crossover(const Type& a, const Type& b, std::mt19937&) const {
    return gene::add(gene::mul(a, WEIGHT), gene::mul(b, 1.0 - WEIGHT));
  }
};

template<typename Type>
class RandomCrossoverPolicy {
public:
  Type crossover(const Type& a, const Type& b, std::mt19937& rng) const {
    std::uniform_real_distribution<double> w(0.0, 1.0);
    double weight = w(rng);
    return gene::add(gene::mul(a, weight), gene::mul(b, 1.0 - weight));
  }
};


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
      [&](std::size_t a, std::size_t b) { return fit(population[a]) > fit(population[b]); });

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

    return { population[idx[dist(rng)]], population[idx[dist(rng)]] };
  }

private:
  Fitness fit;
};

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
      avg += gene::to_scalar(x);
    avg /= population.size();

    if (initialized && std::abs(avg - lastAvg) < PARAM) {
      if (++stableCount > 2) return true;
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
      population(populationSize),
      initPolicy(std::move(init)),
      mutationPolicy(std::move(mutation)),
      crossoverPolicy(std::move(crossover)),
      selection(std::move(selection)),
      stopPolicy(std::move(stop)),
      rng(rng)
  {
    initPolicy.init(population, this->rng);
  }

  void run() {
    std::size_t generation = 0;
    while (!stopPolicy.shouldStop(population, generation)) {
      std::vector<Type> newPopulation;
      newPopulation.reserve(populationSize);

      for (std::size_t i = 0; i < populationSize; ++i) {
        auto [p1, p2] = selection.select(population, rng);
        Type child = crossoverPolicy.crossover(p1, p2, rng);
        newPopulation.push_back(std::move(child));
      }

      population = std::move(newPopulation);
      mutationPolicy.mutate(population, rng);

      ++generation;
    }

    std::cout << "Algorithm stopped.\n";
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

static bool approx(double a, double b, double eps = 1e-9) {
  return std::fabs(a - b) <= eps;
}

template <class F>
static void run_test(const std::string& name, F&& fn) {
  std::cout << "[TEST] " << name << " ... ";
  try {
    fn();
    std::cout << "OK\n";
  } catch (...) {
    std::cout << "FAIL\n";
    throw;
  }
}

int main() {
  std::cout << "=== Manual policy tests (no framework) ===\n";

  // -------------------- INIT POLICIES --------------------
  run_test("RandomInitiationPolicy: range", [] {
    std::mt19937 rng(123);
    std::vector<double> pop(2000);

    RandomInitiationPolicy<double> init(-2.0, 3.0);
    init.init(pop, rng);

    auto [mn, mx] = std::minmax_element(pop.begin(), pop.end());
    assert(*mn >= -2.0);
    assert(*mx <=  3.0);
  });

  run_test("LinSpaceInitiationPolicy (scalar): endpoints + monotonic", [] {
    std::mt19937 rng(123);
    std::vector<double> pop(5);

    LinSpaceInitiationPolicy<double> init(-2.0, 3.0);
    init.init(pop, rng);

    assert(approx(pop.front(), -2.0));
    assert(approx(pop.back(),   3.0));
    for (std::size_t i = 1; i < pop.size(); ++i) assert(pop[i] >= pop[i - 1]);
  });

  // -------------------- MUTATION POLICIES --------------------
  run_test("PercentageMutationPolicy: CHANCE=0 does nothing", [] {
    std::mt19937 rng(123);
    std::vector<double> pop = {1,2,3,4,5};
    auto before = pop;

    PercentageMutationPolicy<double, 0, 50> mut;
    mut.mutate(pop, rng);

    assert(pop == before);
  });

  run_test("PercentageMutationPolicy: CHANCE=100 changes something (INTENSITY>0)", [] {
    std::mt19937 rng(123);
    std::vector<double> pop = {1,2,3,4,5};
    auto before = pop;

    PercentageMutationPolicy<double, 100, 50> mut;
    mut.mutate(pop, rng);

    bool anyDiff = false;
    for (std::size_t i = 0; i < pop.size(); ++i) {
      if (!approx(pop[i], before[i])) { anyDiff = true; break; }
    }
    assert(anyDiff);
  });

  run_test("AbsoluteMutationPolicy: CHANCE=0 does nothing", [] {
    std::mt19937 rng(123);
    std::vector<double> pop = {10,10,10,10,10};
    auto before = pop;

    AbsoluteMutationPolicy<double, 0, 2.0> mut;
    mut.mutate(pop, rng);

    assert(pop == before);
  });

  run_test("AbsoluteMutationPolicy: CHANCE=100 deltas within [-I, I]", [] {
    std::mt19937 rng(123);
    std::vector<double> pop = {10,10,10,10,10};
    auto before = pop;

    AbsoluteMutationPolicy<double, 100, 2.0> mut;
    mut.mutate(pop, rng);

    for (std::size_t i = 0; i < pop.size(); ++i) {
      double d = pop[i] - before[i];
      assert(d >= -2.0 - 1e-9 && d <= 2.0 + 1e-9);
    }
  });

  // -------------------- CROSSOVER POLICIES --------------------
  run_test("AverageCrossoverPolicy: deterministic weight", [] {
    std::mt19937 rng(123);
    AverageCrossoverPolicy<double, 0.25> cx;

    double a = 10.0, b = 2.0;
    double c = cx.crossover(a, b, rng);

    assert(approx(c, 0.25*a + 0.75*b));
  });

  run_test("RandomCrossoverPolicy: convex combination bounds", [] {
    std::mt19937 rng(123);
    RandomCrossoverPolicy<double> cx;

    double a = 10.0, b = 2.0;
    for (int i = 0; i < 2000; ++i) {
      double c = cx.crossover(a, b, rng);
      assert(c >= 2.0 && c <= 10.0);
    }
  });

  // -------------------- SELECTION POLICIES --------------------
  run_test("RandomSelectionPolicy: parents come from population", [] {
    std::mt19937 rng(123);
    std::vector<int> pop = {5,6,7};
    RandomSelectionPolicy<int> sel;

    for (int k = 0; k < 200; ++k) {
      auto [a,b] = sel.select(pop, rng);
      assert(std::find(pop.begin(), pop.end(), a) != pop.end());
      assert(std::find(pop.begin(), pop.end(), b) != pop.end());
    }
  });

  run_test("UniqueRandomSelectionPolicy: unique ordered pairs for n=4 (12 draws)", [] {
    std::mt19937 rng(123);
    std::vector<int> pop = {0,1,2,3};
    UniqueRandomSelectionPolicy<int> sel;

    std::vector<std::pair<int,int>> pairs;
    pairs.reserve(12);
    for (int i = 0; i < 12; ++i) pairs.push_back(sel.select(pop, rng));

    std::sort(pairs.begin(), pairs.end());
    assert(std::adjacent_find(pairs.begin(), pairs.end()) == pairs.end());
  });

  run_test("TargetSelectionPolicy: biases towards best (statistical smoke)", [] {
    std::mt19937 rng(123);
    std::vector<double> pop(50);
    {
      LinSpaceInitiationPolicy<double> init(0.0, 49.0);
      init.init(pop, rng);
    }

    auto fit = [](double x) { return x; }; // best is 49
    // NOTE TEMPLATE ORDER: <Type, FIRST, LAST, Fitness>
    TargetSelectionPolicy<double, 30, 1, decltype(fit)> sel(fit);

    int pickedBest = 0;
    for (int i = 0; i < 3000; ++i) {
      auto [a,b] = sel.select(pop, rng);
      if (approx(a, 49.0) || approx(b, 49.0)) ++pickedBest;
    }

    // random selection would pick best in pair ~ 1 - (49/50)^2 ≈ 0.0396
    // with bias it should be noticeably larger; keep threshold loose to avoid flakiness.
    assert(pickedBest > 3000 * 0.08);
  });

  // -------------------- STOP POLICIES --------------------
  run_test("MaxGenStopConditionPolicy: stops at PARAM", [] {
    MaxGenStopConditionPolicy<int, 5> stop;
    std::vector<int> dummyPop = {1,2,3};

    std::size_t gen = 0;
    while (!stop.shouldStop(dummyPop, gen)) ++gen;
    assert(gen == 5);
  });

  run_test("StableAvgStopConditionPolicy: stable population stops within <= 10 checks", [] {
    StableAvgStopConditionPolicy<double, 1e-12> stop;
    std::vector<double> pop = {1,1,1,1};

    bool stopped = false;
    for (std::size_t gen = 0; gen < 10; ++gen) {
      if (stop.shouldStop(pop, gen)) { stopped = true; break; }
    }
    assert(stopped); // your implementation needs >2 stable checks, so should stop by gen ~3-4
  });

  std::cout << "=== ALL POLICY TESTS PASSED ===\n";
  return 0;
}


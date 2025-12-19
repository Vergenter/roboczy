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

//Concepts
template<typename T> concept Scalar = std::is_arithmetic_v<std::remove_cvref_t<T>>; //std::remove_cvref_t<T> remove from T: const, volatile and references &, && (for std::is_arithmetic_v to work)

template<typename T> concept VectorLike = requires(T a) {
    std::begin(a);
    std::end(a);
	} && Scalar<std::remove_cvref_t<decltype(*std::begin(std::declval<T&>()))>>;

template<typename T> concept Gene = Scalar<T> || VectorLike<T>;

namespace gene {template <class U>
U random_between(const U& mn, const U& mx, std::mt19937& rng) requires Scalar<U> {
  if constexpr (std::is_integral_v<U>) {
    std::uniform_int_distribution<U> d(mn, mx);
    return d(rng);} 
	else {
    std::uniform_real_distribution<double> d((double)mn, (double)mx);
    return static_cast<U>(d(rng));}
}

template<class U>
U random_between(const U& mn, const U& mx, std::mt19937& rng) requires VectorLike<U> {
  U r = mn;
  auto ri = std::begin(r);
  auto mi = std::begin(mn);
  auto ma = std::begin(mx);
  for (; ri != std::end(r); ++ri, ++mi, ++ma) {
    using Elem = std::remove_cvref_t<decltype(*ri)>;
    *ri = random_between<Elem>(*mi, *ma, rng);}
  return r;
}

template<class U>
U random_delta(const U&, std::mt19937& rng, double intensity) requires Scalar<U> {
  if constexpr (std::is_integral_v<U>) {
    const auto I = static_cast<U>(std::llround(intensity));
    std::uniform_int_distribution<long long> d(-static_cast<long long>(I), static_cast<long long>(I));
    return static_cast<U>(d(rng));} 
    else {
    std::uniform_real_distribution<double> d(-intensity, intensity);
    return static_cast<U>(d(rng));
  }
}

template <class U>
U random_delta(const U& sample, std::mt19937& rng, double intensity) requires VectorLike<U>
{
  U r = sample;
  for (auto& v : r) {
    using Elem = std::remove_cvref_t<decltype(v)>;
    v = gene::random_delta<Elem>(Elem{}, rng, intensity); // delegate to scalar
  }
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
  return static_cast<long double>(x);
}

template<class U>
double to_scalar(const U& x) requires VectorLike<U> {
  long double sum = 0.0;
  std::size_t n = 0;
  for (auto v : x) { sum += static_cast<long double>(v); ++n; }
  return (n == 0) ? 0.0 : (sum / static_cast<long double>(n));}
}

namespace pretty { 
template <Scalar T> void print_value(std::ostream& os, const T& x) { os << x;}

template <VectorLike V> void print_value(std::ostream& os, const V& v) {
  os << '[';
  auto it = std::begin(v);
  auto end = std::end(v);
  if (it != end) { print_value(os, *it); ++it; }  //też dla wielowymiarowych tablic
  for (; it != end; ++it) { os << ", "; print_value(os, *it); }
  os << ']';}
  }

template<typename Type> class RandomInitiationPolicy {
public:
  RandomInitiationPolicy(const Type& min, const Type& max) : min_(min), max_(max) {}

  void init(std::vector<Type>& population, std::mt19937& rng) const {
    for (auto& x : population)
      x = gene::random_between(min_, max_, rng);}

private:
  const Type min_;
  const Type max_;
};

template<typename Type> class LinSpaceInitiationPolicy {
public:
  LinSpaceInitiationPolicy( Type min, Type max) : min_(std::move(min)), max_(std::move(max)) {}

  void init(std::vector<Type>& population, std::mt19937&) 
	const requires Scalar<Type>{
    const std::size_t n = population.size();
    if (n == 0) return;
    if (n == 1) { population[0] = min_; return; }

    using CT = std::common_type_t<Type, long double>;
    CT step = (static_cast<CT>(max_) - static_cast<CT>(min_)) / static_cast<CT>(n - 1);
    for (std::size_t i = 0; i < n; ++i)
      population[i] = static_cast<Type>(static_cast<CT>(min_) + static_cast<CT>(i) * step);}


  void init(std::vector<Type>& population, std::mt19937&) 
  const requires VectorLike<Type>{
    const std::size_t n = population.size();
    if (n == 0) return;
    if (n == 1) { population[0] = min_; return;}

    for (std::size_t i = 0; i < n; ++i) {
      long double t = static_cast<long double>(i) / static_cast<long double>(n - 1);
      Type x = min_;
      auto xi = std::begin(x);
      auto mi = std::begin(min_);
      auto ma = std::begin(max_);
      for (; xi != std::end(x); ++xi, ++mi, ++ma) {
        using Elem = std::remove_cvref_t<decltype(*xi)>;
        long double v = static_cast<long double>(*mi) + (static_cast<long double>(*ma) - static_cast<long double>(*mi)) * t;
        *xi = static_cast<Elem>(v);}
      population[i] = std::move(x);}}

private:
  const Type min_;
  const Type max_;
};

//Mutation policies
template<typename Type, int CHANCE, int INTENSITY> class PercentageMutationPolicy {
public:
  static_assert(CHANCE >= 0 && CHANCE <= 100, "Value must be >=0 and <= 100");
  static_assert(INTENSITY > 0, "Value must be >0");

  void mutate(std::vector<Type>& population, std::mt19937& rng) const {
    std::uniform_int_distribution<int> prob(0, 99);
    std::uniform_real_distribution<double> factor(
      1.0 - INTENSITY / 100.0,
      1.0 + INTENSITY / 100.0);

    for (auto& x : population)
      if (prob(rng) < CHANCE)
        x = gene::mul(x, factor(rng));}
};

template<Gene Type, int CHANCE, double INTENSITY> class AbsoluteMutationPolicy {
public:
  static_assert(CHANCE >= 0 && CHANCE <= 100, "Value must be >=0 and <= 100");
  static_assert(INTENSITY > 0, "Value must be >0");

  void mutate(std::vector<Type>& population, std::mt19937& rng) const {
    std::uniform_int_distribution<int> prob(0, 99);

    for (auto& x : population)
      if (prob(rng) < CHANCE)
        x = gene::add(x, gene::random_delta(x, rng, INTENSITY));}
};

//Crossover policies
template<typename Type, double WEIGHT> class AverageCrossoverPolicy {
public:
  static_assert(WEIGHT >= 0 && WEIGHT <= 1, "Value must be >=0 and <= 1");

  Type crossover(const Type& a, const Type& b, std::mt19937&) const {
    return gene::add(gene::mul(a, WEIGHT), gene::mul(b, 1.0 - WEIGHT));}
};

template<typename Type> class RandomCrossoverPolicy {
public:
  Type crossover(const Type& a, const Type& b, std::mt19937& rng) const {
    std::uniform_real_distribution<double> w(0.0, 1.0);
    double weight = w(rng);
    return gene::add(gene::mul(a, weight), gene::mul(b, 1.0 - weight));}
};

//Selection policies
template<typename Type> class RandomSelectionPolicy {
public:
  std::pair<Type, Type> select(const std::vector<Type>& population, std::mt19937& rng) const {
    std::uniform_int_distribution<std::size_t> dist(0, population.size() - 1);
    return { population[dist(rng)], population[dist(rng)] };}
};

template<typename Type>
class UniqueRandomSelectionPolicy {
public:
  std::pair<Type, Type> select(const std::vector<Type>& population, std::mt19937& rng) {
    const std::size_t n = population.size();
    if (n < 2) throw std::runtime_error("UniqueRandomSelectionPolicy requires population.size() >= 2");

    if (n != cachedSize || produced == n) {
      cachedSize = n;
      produced = 0;

      first.resize(n);
      std::iota(first.begin(), first.end(), 0);
      std::shuffle(first.begin(), first.end(), rng);

      shift.resize(n);
      std::uniform_int_distribution<std::size_t> dist(0, std::numeric_limits<std::size_t>::max());
      for (std::size_t i = 0; i < n; ++i) shift[i] = dist(rng);
    }

    const std::size_t i = produced;

    // partner = (i + 1 + (shift[i] % (n-1))) % n
    const std::size_t offset = 1 + (shift[i] % (cachedSize - 1));
    const std::size_t partner_pos = (i + offset) % cachedSize;

    const std::size_t a = first[i];
    const std::size_t b = first[partner_pos];

    ++produced;
    return { population[a], population[b] };
  }

private:
  std::vector<std::size_t> first;
  std::vector<std::size_t> shift;
  std::size_t cachedSize = 0;
  std::size_t produced = 0;
};

template<typename Type, int FIRST, int LAST, typename Fitness>
class TargetSelectionPolicy {
public:
  static_assert(FIRST >= 0 && FIRST <= 100, "FIRST must be in [0,100]");
  static_assert(LAST  >= 0 && LAST  <= 100, "LAST must be in [0,100]");

  explicit TargetSelectionPolicy(Fitness f = {}) : fit(std::move(f)) {}

  std::pair<Type, Type> select(const std::vector<Type>& population, std::mt19937& rng) {
    const std::size_t n = population.size();
    if (n == 0) throw std::runtime_error("TargetSelectionPolicy requires population.size() > 0");

    if (n != cachedSize || produced == n) {
      rebuild_cache(population);
      cachedSize = n;
      produced = 0;
    }

    ++produced;

    const std::size_t a_rank = dist(rng);
    const std::size_t b_rank = dist(rng);
    return { population[idx[a_rank]], population[idx[b_rank]] };
  }

private:
  void rebuild_cache(const std::vector<Type>& population) {
    const std::size_t n = population.size();

    idx.resize(n);
    std::iota(idx.begin(), idx.end(), 0);

    std::sort(idx.begin(), idx.end(),
      [&](std::size_t a, std::size_t b) {
        return fit(population[a]) > fit(population[b]);
      });

    std::vector<double> probs(n);
    if (n == 1) {
      probs[0] = 1.0;
    } else {
      const double first = static_cast<double>(FIRST);
      const double last  = static_cast<double>(LAST);
      const double step  = (first - last) / static_cast<double>(n - 1);
      for (std::size_t i = 0; i < n; ++i)
        probs[i] = first - static_cast<double>(i) * step;
    }

    dist = std::discrete_distribution<std::size_t>(probs.begin(), probs.end());
  }

private:
  // Stores stateful fitness; zero-size when Fitness is empty (EBO in C++20)
  [[no_unique_address]] Fitness fit;

  std::vector<std::size_t> idx;
  std::discrete_distribution<std::size_t> dist;

  std::size_t cachedSize = 0;
  std::size_t produced = 0;
};

//Stop policies
template<typename Type, std::size_t PARAM> class MaxGenStopConditionPolicy {
public:
  bool shouldStop(const std::vector<Type>&, std::size_t generation) const {
    return generation >= PARAM;}
};

template<typename Type, double PARAM> class StableAvgStopConditionPolicy {
public:
  bool shouldStop(const std::vector<Type>& population, std::size_t) {
    long double avg = 0.0;
    for (auto& x : population)
      avg += gene::to_scalar(x);
    avg /= population.size();

    if (initialized && std::abs(avg - lastAvg) < PARAM) {
      if (++stableCount > 2) return true;} 
	else {
      stableCount = 0;}

    lastAvg = avg;
    initialized = true;
    return false;
  }

private:
  double lastAvg = 0.0;
  std::size_t stableCount = 0;
  bool initialized = false;
};

//Evolutionary algorithm
template<Gene Type, typename InitPolicy, typename MutationPolicy,
         typename CrossoverPolicy, typename SelectionPolicy, typename StopPolicy>
class EvolutionaryAlgorithm {
public:
  EvolutionaryAlgorithm(
    std::size_t populationSize, SelectionPolicy selection,
    InitPolicy init = InitPolicy{}, MutationPolicy mutation = MutationPolicy{},
    CrossoverPolicy crossover = CrossoverPolicy{}, StopPolicy stop = StopPolicy{},
    std::mt19937 rng = std::mt19937{std::random_device{}()})
	
    : populationSize(populationSize), population(populationSize),
      initPolicy(std::move(init)), mutationPolicy(std::move(mutation)),
      crossoverPolicy(std::move(crossover)), selection(std::move(selection)),
      stopPolicy(std::move(stop)), rng(rng)
	  
  {initPolicy.init(population, this->rng);}

  void printPopulation(std::ostream& os = std::cout) const {
	for (const auto& x : population) {
      pretty::print_value(os, x);
      os << '\n';}}
  
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
    std::cout << "Algorithm stopped after " << generation << " generations.\n";
	printPopulation();	
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

//Fitness
template<Scalar T> double fitness(const T& x) {
  return -std::abs(x);}

template<VectorLike T> double fitness(const T& x) {
  double sum = 0.0;
  for (auto v : x) sum += v * v;
  return -std::sqrt(sum);}

static bool approx(double a, double b, double eps = 1e-9) {
  return std::fabs(a - b) <= eps;}

template <class F> static void run_test(const std::string& name, F&& fn) {
  std::cout << "[TEST] " << name << " ... ";
  try {
    fn();
    std::cout << "OK\n";} 
	catch (...) {
    std::cout << "FAIL\n";
    throw;}
}

int main() {

  //INIT POLICIES 
 run_test("RandomInitiationPolicy: range", [] {
    std::mt19937 rng(123);
    std::vector<double> pop(2000);
    RandomInitiationPolicy<double> init(2.0, 3.0);
    init.init(pop, rng);
    auto [mn, mx] = std::minmax_element(pop.begin(), pop.end());
    assert(*mn >= 2.0);
    assert(*mx <=  3.0);});

 run_test("RandomInitiationPolicy (vector): per-component range", [] {
  std::mt19937 rng(123);
  using V = std::array<double, 3>;
  std::vector<V> pop(2000);
  V mn{ { -1.0,  0.0,  10.0 } };
  V mx{ {  1.0,  2.0,  13.0 } };
  RandomInitiationPolicy<V> init(mn, mx);
  init.init(pop, rng);
  for (const auto& x : pop) {
    for (std::size_t i = 0; i < x.size(); ++i) {
      assert(x[i] >= mn[i]);
      assert(x[i] <= mx[i]);}}
});

  run_test("LinSpaceInitiationPolicy (scalar): endpoints + monotonic", [] {
    std::mt19937 rng(123);
    std::vector<double> pop(5);
    LinSpaceInitiationPolicy<double> init(-2.0, 3.0);
    init.init(pop, rng);
    assert(approx(pop.front(), -2.0));
    assert(approx(pop.back(),   3.0));
    for (std::size_t i = 1; i < pop.size(); ++i) assert(pop[i] >= pop[i - 1]);});

  //MUTATION POLICIES
  run_test("PercentageMutationPolicy: CHANCE=0 does nothing", [] {
    std::mt19937 rng(123);
    std::vector<double> pop = {1,2,3,4,5};
    auto before = pop;
    PercentageMutationPolicy<double, 0, 50> mut;
    mut.mutate(pop, rng);
    assert(pop == before);});

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

  //CROSSOVER POLICIES
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
      assert(c >= 2.0 && c <= 10.0);}
  });

  //SELECTION POLICIES
  run_test("RandomSelectionPolicy: parents come from population", [] {
    std::mt19937 rng(123);
    std::vector<int> pop = {5,6,7};
    RandomSelectionPolicy<int> sel;
    for (int k = 0; k < 200; ++k) {
      auto [a,b] = sel.select(pop, rng);
      assert(std::find(pop.begin(), pop.end(), a) != pop.end());
      assert(std::find(pop.begin(), pop.end(), b) != pop.end());}
  });

  run_test("UniqueRandomSelectionPolicy: unique ordered pairs for n=4", [] {
    std::mt19937 rng(123);
    std::vector<int> pop = {0,1,2,3};
    UniqueRandomSelectionPolicy<int> sel;

    std::vector<std::pair<int,int>> pairs;
    pairs.reserve(4);
    for (int i = 0; i < 4; ++i) pairs.push_back(sel.select(pop, rng));

    std::sort(pairs.begin(), pairs.end());
    assert(std::adjacent_find(pairs.begin(), pairs.end()) == pairs.end());
  });

  run_test("TargetSelectionPolicy: biases towards best", [] {
    std::mt19937 rng(123);
    std::vector<double> pop(50);{
      LinSpaceInitiationPolicy<double> init(0.0, 49.0);
      init.init(pop, rng);}
    auto fit = [](double x) { return x; };
    TargetSelectionPolicy<double, 30, 1, decltype(fit)> sel(fit);
    int pickedBest = 0;
    for (int i = 0; i < 3000; ++i) {
      auto [a,b] = sel.select(pop, rng);
      if (approx(a, 49.0) || approx(b, 49.0)) ++pickedBest;}
    assert(pickedBest > 3000 * 0.065);
  });

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
    for (std::size_t gen = 0; gen < 4; ++gen) {
      if (stop.shouldStop(pop, gen)) { stopped = true; break; }}
    assert(stopped);
     });

  run_test("E2E scalar: works for scalar", [] {
    using T = double;
    std::mt19937 rng(123);
    RandomSelectionPolicy<T> selection;
    RandomInitiationPolicy<T> init(-10.0, 10.0);
    AbsoluteMutationPolicy<T, 40, 0.5> mutation;
    AverageCrossoverPolicy<T, 0.5> crossover;
    StableAvgStopConditionPolicy<T, 1e-2> stop;
    EvolutionaryAlgorithm<T,
    decltype(init),
    decltype(mutation),
    decltype(crossover),
    decltype(selection),
    decltype(stop)
  > ea(50, selection, init, mutation, crossover, stop, rng);
  ea.run();
 });

run_test("E2E vector<2>: works for vectors", [] {
  using V = std::array<double, 2>;
  std::mt19937 rng(123);
  V mn{ {-5.0, -5.0} };
  V mx{ { 5.0,  5.0} };
  RandomSelectionPolicy<V> selection;
  RandomInitiationPolicy<V> init(mn, mx);
  AbsoluteMutationPolicy<V, 40, 0.4> mutation;
  AverageCrossoverPolicy<V, 0.5> crossover;
  StableAvgStopConditionPolicy<V, 1e-2> stop;

  EvolutionaryAlgorithm<V,
    decltype(init),
    decltype(mutation),
    decltype(crossover),
    decltype(selection),
    decltype(stop)
  > ea(60, selection, init, mutation, crossover, stop, rng);
    ea.run();
});

  std::cout << "=== ALL POLICY TESTS PASSED ===\n";
  return 0;
}

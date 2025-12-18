#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <ctime>


/***
 * Przygotować prosty algorytm ewolucyjny dla populacji liczb, wektorów lub tablic.
 * Klasy wytycznych mają kontrolować działanie algorytmu.
 * Jako parametry szablonu algorytmu ewolucyjnego mogą zostać podane typy, szablony lub instancje policy,
 * według Państwa wyboru.
 * Celem jest, poza funkcjonalnym algorytmem, wygodna konfiguracja algorytmu, i na to będę zwracał szczególną uwagę!
 * Mogą Państwo dowolnie modyfikować kod, o ile uznają Państwo, że to poprawi użytkowość,
 * przy czym sama zasada działania algorytmu ma się nie zmienić.
 * Istotne zmiany API powinny zostać opisane i podane przykłady użycia.
 *
 * Klasy wytycznych dla inicjalizacji populacji:
 *  - RandomInitiationPolicy<Type, MIN, MAX>: wypełnia populację losowymi osobnikami z zakresu (MIN, MAX)
 *  - LinSpaceInitiationPolicy<Type, MIN, MAX>: wypełnia populację osobnikami z zakresu (MIN, MAX),
 *    tak aby osobniki były liniowo rozłożone pomiędzy MIN i MAX.
 * Klasy wytycznych dla mutacji populacji:
 *  - PercentageMutationPolicy<Type, CHANCE, INTENSITY>: mutuje z prawdopodobieństwem CHANCE
 *    i mnoży wartość osobnika przez liczbę z zakresu (1 - INTENSITY/100, 1 + INTENSITY/100).
 *  - AbsoluteMutationPolicy<Type, CHANCE, INTENSITY>: mutuje z prawdopodobieństwem CHANCE
 *    i dodaje do wartości osobnika losową wartość z zakresu (-INTENSITY, INTENSITY).
 * Klasy wytycznych dla krzyżowania populacji:
 *  - AverageCrossoverPolicy<Type, WEIGHT>: tworzy nowego osobnika jako średnią ważoną rodziców (wagi to WEIGHT i 1 - WEIGHT).
 *    W przypadku wektorów wagi powinny być wektorami o wartościach z zakresu (0, 1) i tej samej długości co Type.
 *  - RandomCrossoverPolicy<Type>: To samo co wyżej, ale waga jest losowa.
 
 * Klasy wytycznych dla selekcji:
 *  - RandomSelectionPolicy<Type>: wybiera dwoje rodziców w sposób losowy.
 *  - UniqueRandomSelectionPolicy<Type>: wybiera dwoje rodziców w sposób losowy, ale bez powtórzeń (każda para jest wybierana dwukrotnie).
 *  - TargetSelectionPolicy<Type, Fit, FIRST, LAST>: wybiera dwoje rodziców w sposób losowy,
 *    ale w taki sposób, że cała populacja jest ułożona w rankingu od najlepszego do najgorszego wg funkcji `double Fit(Type)`.
 *    Osobnik pierwszy w rankingu ma mieć FIRST% szans, a ostatni LAST%. Reszta ma szanse malejące w sposób liniowy.
 
 * Klasy wytycznych dla zakończenia algorytmu:
 *  - MaxGenStopConditionPolicy<Type, PARAM>: przerywa algorytm po PARAM generacjach.
 *  - StableAvgStopConditionPolicy<Type, PARAM>: przerywa algorytm, jeżeli od poprzedniego sprawdzenia warunku średnia generacji
 *    nie zmieniła się o więcej niż PARAM.
 */



template<Type, MIN, MAX>
struct InitiationPolicy {
    void init(std::vector<Type>& population, std::size_t populationSize);
};

template<Type, CHANCE, INTENSITY>
struct MutationPolicy {
    static void mutate(std::vector<Type>& population);
};

template<Type, WEIGHT>
struct CrossoverPolicy {
    static Type crossover(const Type, const Type);
};

template<Type, FIRST, LAST>
struct SelectionPolicy {
    static std::pair<Type, Type> select(const std::vector<Type>& population);
};

template<Type, PARAM>
struct StopConditionPolicy {
    static bool shouldStop(const std::vector<Type>& population, std::size_t generation);
};

template <Type, InitiationPolicy, MutationPolicy, CrossoverPolicy, selectionPolicy, StopConditionPolicy>
class EvolutionaryAlgorithm {
public:
    EvolutionaryAlgorithm(int populationSize)
            : populationSize(populationSize) {
        std::srand(std::time(nullptr));
        InitiationPolicy<Type>::init(population, populationSize);
    }

    void run() {
        int generation = 0;
        while (!StopConditionPolicy::shouldStop(population, generation)) {
            std::vector<Type> newPopulation;

            selectionPolicy.init();
            for (int i = 0; i < populationSize; ++i) {
                auto [parent1, parent2] = selectionPolicy.select(population);
                Type offspring = CrossoverPolicy::crossover(parent1, parent2);
                newPopulation.push_back(offspring);
            }

            population = newPopulation;
            MutationPolicy::mutate(population);
            generation++;
        }

        std::cout << "Algorithm stopped after " << generation << " generations.\n";
        printPopulation();
    }

private:
    std::vector<Type> population;
    int populationSize;

    void printPopulation() const {
        for (int individual : population) {
            std::cout << individual << " ";
        }
        std::cout << "\n";
    }
};


int main() {
    const int populationSize = 10; // Rozmiar populacji

    EvolutionaryAlgorithm<
        double,
        InitiationPolicy,
        MutationPolicy,
        CrossoverPolicy,
        SelectionPolicy<> selectionPolicy(),
        StopConditionPolicy
        > algorithm(populationSize);
    algorithm.run();

    return 0;
}

import math
import random
import sys
from time import time
from typing import List, Tuple, Callable, TypeVar

import deap
import matplotlib.pyplot as pyplot
import numpy
import pandas
from IPython.display import display
from deap import algorithms, base, creator, tools

print("----- Initializing Python Script -----")
print(f"Python version: {sys.version}")
print(f"DEAP version: {deap.__version__}")
print(f"Numpy version: {numpy.__version__}")

# =========================
#  Genetic Algorithm
# =========================
# minimize fitness
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

T = TypeVar('T')
def evolution(*,
      individual_generator: Callable[[], T],
      fitness_evaluation: Callable[[T], float],
      population_size: int,
      generations: int,
      crossover_rate: float,
      mutation_rate: float,
      mutation_function: Tuple[Callable, dict],
):
    # track start time
    start_time = time()

    toolbox = base.Toolbox()

    # register population generators
    toolbox.register("individual", tools.initIterate, creator.Individual, individual_generator)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # register fitness evaluation function
    toolbox.register("evaluate", lambda individual: (fitness_evaluation(individual),))

    # register mutators for individuals
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutation_function[0], **mutation_function[1])

    # register next generation selection function
    toolbox.register("select", tools.selTournament, tournsize=4)

    # generation statistics logging
    stats = tools.Statistics(key=lambda individual: individual.fitness.values)
    stats.register("min", lambda population: numpy.min([fitness for fitness in population if fitness[0] != math.inf]))
    stats.register("avg", lambda population: numpy.mean([fitness for fitness in population if fitness[0] != math.inf]))
    stats.register("max", lambda population: numpy.max([fitness for fitness in population if fitness[0] != math.inf]))

    # log fittest individual with a hall of fame record
    hall_of_fame = tools.HallOfFame(maxsize=1)

    # run genetic algorithm
    _, log = algorithms.eaSimple(
        # generate all individuals in population
        toolbox.population(n=population_size),
        toolbox,
        ngen=generations,
        cxpb=crossover_rate, mutpb=mutation_rate,
        stats=stats, halloffame=hall_of_fame, verbose=False,
    )

    # return duration, statistics log, and the fittest individual
    return time() - start_time, log, hall_of_fame[0]

# =========================
#  Pop Visualizer
# =========================
def plot_generations(generation: List[int], average: List[float], minimum: List[int], maximum: List[int]):
    pyplot.figure(figsize=(16, 8))
    pyplot.grid(True)
    pyplot.plot(generation, average, label="average")
    pyplot.plot(generation, minimum, label="minimum")
    pyplot.plot(generation, maximum, label="maximum")
    pyplot.xlabel("Generation")
    pyplot.ylabel("Fitness")
    pyplot.ylim(-1)
    pyplot.legend(loc="upper right")
    pyplot.show()


# =========================
#  Rendering
# =========================
def display_positional_grid(individual: List[int]):
    # unpack dimensions
    dimension = len(individual)

    # construct board using pandas
    board = pandas.DataFrame("", index = range(1, dimension + 1), columns = range(1, dimension + 1))

    # draw all conflicts with a red line
    for x in range(dimension):
        x_row, x_column = individual[x] // dimension, individual[x] % dimension
        for y in range(x + 1, dimension):
            y_row, y_column = individual[y] // dimension, individual[y] % dimension

            diff_row, diff_column = y_row - x_row, y_column - x_column
            # check if queens are conflicting
            if x_row == y_row or x_column == y_column or abs(diff_row) == abs(diff_column):
                # draw a line of the conflict
                for i in range(1 + max(abs(diff_row), abs(diff_column))):
                    board[1 + x_column + i * numpy.sign(diff_column)][1 + x_row + i * numpy.sign(diff_row)] = "x"

    # draw all queens
    for queen in individual:
        row, column = queen // dimension, queen % dimension
        # use a crown if
        board[1 + column][1 + row] = "Q" if board[1 + column][1 + row] == "" else "q"

    # render board with pandas
    display(board)

# =========================
#  Example Code
# =========================
example_individual = random.sample(range(8**2), 8)
print(f"Positions: {example_individual}")
print(f"Duplicate positions: {len(example_individual) - len(set(example_individual))}")
display_positional_grid(example_individual)

# =========================
#  Fitness Function
# =========================
def evaluate_position_indexed_fitness(individual: List[int]) -> float:
    # duplicate values should be removed with the severest penalty
    if len(individual) != len(set(individual)):
        return math.inf

    # unpack dimensions
    dimension = len(individual)
    # count all pairs of conflicts
    fitness: float = 0
    for x in range(len(individual)):
        x_row, x_column = individual[x] // dimension, individual[x] % dimension
        for y in range(x + 1, len(individual)):
            y_row, y_column = individual[y] // dimension, individual[y] % dimension
            if x_row == y_row or x_column == y_column or abs(x_row - y_row) == abs(x_column - y_column):
                fitness += 1
    return fitness

# =========================
#  N = 16
# =========================
duration, log, fittest_individual_p8 = evolution(
    individual_generator=lambda: random.choices(range(8**2), k=8),
    fitness_evaluation=evaluate_position_indexed_fitness,
    population_size=2500,
    generations=100,
    crossover_rate=.5,
    mutation_rate=.5,
    mutation_function=(tools.mutUniformInt, {"low": 0, "up": 8**2 - 1, "indpb": 1/4})
)
print(f"Computed in {duration:.3f} seconds")
plot_generations(*log.select("gen", "avg", "min", "max"))

# Render
print(f"Positions: {fittest_individual_p16}")
print(f"Duplicate queens: {len(fittest_individual_p16) - len(set(fittest_individual_p16))}")
print(f"Fitness: {abs(fittest_individual_p16.fitness.values[0])}")
display_positional_grid(fittest_individual_p16)
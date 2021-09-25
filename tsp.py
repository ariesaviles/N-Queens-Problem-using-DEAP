import os
import pickle
import array
import csv
import codecs
import random
from urllib.request import urlopen
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

URL_PREFIX = 'http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/'

# =========================
#  Read Data
# =========================
def read_data():
    # an array which contains numpy arrays of the coordinates
    locations = []
    with open("TSPDATA.txt", "r") as f:
        # skip the first 2 lines of the txt file
        for i in range(2):
            next(f)
        # remove the index column of the remaining text file
        content = [x.strip('\n')[5:] for x in f.readlines()]
        col_num = 1 # originally pointing at y column
        for row in content:
            # specify the x and y columns 
            locations.append(np.asarray([row.split()[col_num-1], row.split()[col_num]], dtype=np.int32))

    
    number_cities = len(locations)
    distances = [[0] * number_cities for _ in locations]

    # calculate distances
    for i in range(number_cities):
        for j in range(i + 1, number_cities):
            # distance found using vector norm from numpy package
            distance = np.linalg.norm(locations[j] - locations[i])
            distances[i][j] = distances[j][i] = distance

    print(f"\nlocations: {locations}")
    return locations, distances

# =========================
#  Set Parameters
# =========================
CITIES, DISTANCES = read_data()
NUMBER_CITIES = len(CITIES)
NUM_GENERATIONS = 3000
POPULATION_SIZE = 100
P_CROSSOVER = 0.9
P_MUTATION = 0.1

individual = list(range(NUMBER_CITIES))
print(f"individ: {individual}")
individual = random.sample(individual, len(individual))
print(individual)

# =========================
#  Calculate Distances
# =========================
def tsp_distance(individual: list) -> float:
    # get distance between first and last city
    distance = DISTANCES[individual[0]][individual[-1]]
    # add all other distances
    for i in range(NUMBER_CITIES - 1):
        distance += DISTANCES[individual[i]][individual[i + 1]]
    return distance

def tspFitness(individual) -> tuple:
    return tsp_distance(individual),


creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', array.array, typecode='i', fitness=creator.FitnessMin)
toolbox = base.Toolbox()
# Create operator to shuffle the cities
toolbox.register('randomOrder', random.sample, range(NUMBER_CITIES), NUMBER_CITIES)
# Create initial random individual operator
toolbox.register('individualCreator', tools.initIterate, creator.Individual, toolbox.randomOrder)
# Create random population operator
toolbox.register('populationCreator', tools.initRepeat, list, toolbox.individualCreator)

toolbox.register('evaluate', tspFitness)
toolbox.register('select', tools.selTournament, tournsize=3)
toolbox.register('mate', tools.cxOrdered)
toolbox.register('mutate', tools.mutShuffleIndexes, indpb=1.0 / NUMBER_CITIES)

population = toolbox.populationCreator(n=POPULATION_SIZE)

HALL_OF_FAME_SIZE = 10
hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register('min', np.min)
stats.register('avg', np.mean)

logbook = tools.Logbook()
logbook.header = ['gen'] + stats.fields

invalid_individuals = [ind for ind in population if not ind.fitness.valid]
fitnesses = toolbox.map(toolbox.evaluate, invalid_individuals)
for ind, fit in zip(invalid_individuals, fitnesses):
    ind.fitness.values = fit

hof.update(population)
hof_size = len(hof.items)

record = stats.compile(population)
logbook.record(gen=0, **record)
print(logbook.stream)

for gen in range(1, NUM_GENERATIONS + 1):
    # Select the next generation individuals
    offspring = toolbox.select(population, len(population) - hof_size)

    # Vary the pool of individuals
    offspring = algorithms.varAnd(offspring, toolbox, P_CROSSOVER, P_MUTATION)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # add the best back to population:
    offspring.extend(hof.items)

    # Update the hall of fame with the generated individuals
    hof.update(offspring)

    # Replace the current population by the offspring
    population[:] = offspring

    # Append the current generation statistics to the logbook
    record = stats.compile(population) if stats else {}
    logbook.record(gen=gen, **record)
    print(logbook.stream)

best = hof.items[0]
print('Best Fitness = ', best.fitness.values[0])
plt.figure(1)

# plot genetic flow statistics:
minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
plt.figure(2)
sns.set_style("whitegrid")
plt.plot(minFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Generation')
plt.ylabel('Min / Average Fitness')
plt.title('Min and Average fitness over Generations')
# show both plots:
plt.show()

# now plot the best travelling path.
plt.scatter(*zip(*CITIES), marker='.', color='red')
locs = [CITIES[i] for i in best]
locs.append(locs[0])
plt.plot(*zip(*locs), linestyle='-', color='blue')

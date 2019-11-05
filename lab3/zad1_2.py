from deap import creator, base, tools, algorithms
import random
import numpy
import matplotlib.pyplot as plt

ITEMS_COUNT = 11
INIT_SIZE = 5
MAX_WEIGHT = 25

items = {0: ('zegar', (100, 7)), 1: ('obraz-pejzaż', (300, 7)), 2: ('obraz-portret', (200, 6)), 3: ('radio', (40, 2)),
         4: ('laptop', (500, 5)), 5: ('lampka nocna', (70, 6)), 6: ('srebrne sztućce', (100, 1)),
         7: ('porcelana', (250, 3)), 8: ('figurka z brazu', (300, 10)), 9: ('skorzana torebka', (280, 3)),
         10: ('odkurzacz', (300, 15))}

creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", set, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("attr_item", random.randrange, ITEMS_COUNT)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_item, INIT_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalKnapsack(individual):
    weight = 0.0
    value = 0.0
    for item in individual:
        weight += items[item][1][1]
        value += items[item][1][0]
    if weight > MAX_WEIGHT:
        return 0, 10000
    return value, weight


def cxSet(ind1, ind2):
    temp = set(ind1)
    ind1 &= ind2
    ind2 ^= temp
    return ind1, ind2


def mutSet(individual):
    if random.random() < 0.5:
        if len(individual) > 0:
            individual.remove(random.choice(sorted(tuple(individual))))
    else:
        individual.add(random.randrange(ITEMS_COUNT))
    return individual,


toolbox.register("evaluate", evalKnapsack)
toolbox.register("mate", cxSet)
toolbox.register("mutate", mutSet)
toolbox.register("select", tools.selNSGA2)

NGEN = 100
MU = 200
LAMBDA = 100
CXPB = 0.5
MUTPB = 0.5

pop = toolbox.population(n=MU)
hof = tools.ParetoFront()
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", numpy.mean, axis=0)
stats.register("std", numpy.std, axis=0)
stats.register("min", numpy.min, axis=0)
stats.register("max", numpy.max, axis=0)

_pa, logs = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats, halloffame=hof)

print(hof[0])
for i in hof[0]:
    print(items[i])

gen = []
avg = []
max = []

for i in logs:
    gen.append(i['gen'])
    avg.append(i['avg'][0])
    max.append(i['max'][0])

x_pos = [i for i, _ in enumerate(gen)]
fig, ax1 = plt.subplots()
line1 = ax1.plot(gen, avg, "b-", label="AVG")
ax1.set_xlabel("GEN")
ax1.set_ylabel("AVG", color="b")
for tl in ax1.get_yticklabels():
    tl.set_color("b")

ax2 = ax1.twinx()
line2 = ax2.plot(gen, max, "r-", label="MAX")
ax2.set_ylabel("MAX", color="r")
for tl in ax2.get_yticklabels():
    tl.set_color("r")

lns = line1 + line2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="center right")

plt.show()

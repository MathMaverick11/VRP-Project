# ga_module.py

import random
import math
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# function to find distance between two points
def euclidean_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# function to generate the random coordinates
def generate_coordinates(n, x_range=(100, 1000), y_range=(100, 1000)):
    return [
        (random.uniform(*x_range), random.uniform(*y_range))
        for _ in range(n)
    ]

# evaluation function returning total_distance traversed and the variance between distance traversed by each vehicle
def evalVRP(individual, locations, depot, num_vehicles):
    distance_by_truck = np.zeros(num_vehicles)
    last_loc = [depot] * num_vehicles

    # looping over all the location coordinates
    for idx_pos, loc_idx in enumerate(individual):
        truck_id = idx_pos % num_vehicles # to get known this location index is traversed by which number truck
        current = locations[loc_idx]   # to get the current target location
        distance_by_truck[truck_id] += euclidean_distance(
            last_loc[truck_id], current
        )  # euclidean distance will be calculated between previous location of the truck and the new location where does it want to go
        last_loc[truck_id] = current  # then setting the current location as previous location of the truck for the next iteration

    # after traversing to all the locations all truck will return to the depot as well calculate that euclidean distance as well
    for truck_id in range(num_vehicles):
        distance_by_truck[truck_id] += euclidean_distance(
            last_loc[truck_id], depot
        )

    total_dist = np.sum(distance_by_truck)
    variance = np.var(distance_by_truck)
    return (total_dist, variance)

# running the genetic algorithm function
def run_ga(
    locations,
    depot,
    num_vehicles,
    pop_size=200,
    cxpb=0.7,
    mutpb=0.2,
    tournsize=3,
    ngen=30,
    random_seed=42,
):
    random.seed(random_seed)

    # recreate the fitness and individual classes if they are not created
    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    num_locations = len(locations)

    # each individual is just a random permutation of the indices
    toolbox.register("indices", random.sample, range(num_locations), num_locations)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # to find the fitness value of each individual
    def eval_local(ind):
        return evalVRP(ind, locations, depot, num_vehicles)

    toolbox.register("evaluate", eval_local)
    toolbox.register("mate", tools.cxPartialyMatched)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    pop, logbook = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=ngen,
        stats=stats,
        halloffame=hof,
        verbose=False,
    )

    return hof[0], logbook

# plotting the routes using the maplotlib
def plot_routes(individual, locations, depot, num_vehicles, title="Routes"):
    total_dist, variance = evalVRP(individual, locations, depot, num_vehicles)

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["b", "g", "m", "c", "y", "k"]

    # Plot each location
    for idx, (x, y) in enumerate(locations):
        ax.plot(x, y, "o", color="blue", zorder=3)
        ax.text(x + 5, y + 5, str(idx), fontsize=9, zorder=4)

    # Plot depot
    ax.plot(depot[0], depot[1], marker="s", color="red", markersize=12, zorder=5)
    ax.text(
        depot[0],
        depot[1] + 20,
        "Depot",
        fontsize=12,
        color="red",
        weight="bold",
        ha="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
    )

    # Plot each truck’s route
    for i in range(num_vehicles):
        route_idxs = [individual[j] for j in range(i, len(individual), num_vehicles)]
        coords = [depot] + [locations[k] for k in route_idxs] + [depot]
        xs, ys = zip(*coords)
        ax.plot(xs, ys, "-", color=colors[i % len(colors)], label=f"Vehicle {i+1}")

    ax.set_title(f"{title}\nTotal Dist: {total_dist:.2f}, Variance: {variance:.2f}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig

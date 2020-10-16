import numpy as np
from utils import get_CA
from tqdm import tqdm
from config import config as cfg
from copy import deepcopy


class Tree:
    def __init__(self, m, X, Y):
        self.vector = np.random.randint(0, 2, size=m)
        self.age = 0
        self.CA = get_CA(self.vector, X, Y)


def local_seeding(forest, LSC, X, Y):
    gen_trees = []
    for tree in forest:
        if tree.age == 0:
            seeds = set()
            while len(seeds) < LSC:
                seeds.add(np.random.randint(0, tree.vector.shape[0]))
            for seed in seeds:
                gen_tree = deepcopy(tree)
                gen_tree.vector[seed] = 0 if gen_tree.vector[seed] == 1 else 1
                gen_tree.CA = get_CA(gen_tree.vector, X, Y)
                gen_trees.append(gen_tree)
        tree.age += 1
    forest += gen_trees
    return forest


def population_limiting(forest, candidate_forest, life_time, area_limit):
    # life time
    temp_forest = []
    for tree in forest:
        if tree.age > life_time:
            candidate_forest.append(tree)
        else:
            temp_forest.append(tree)

    # area limit
    if len(temp_forest) > area_limit:
        temp_forest.sort(key=lambda tree: tree.CA, reverse=True)
        forest = temp_forest[:area_limit]
        candidate_forest += temp_forest[area_limit:]
    else:
        forest = temp_forest
    return forest, candidate_forest


def global_seeding(forest, candidate_forest, transfer_rate, GSC, X, Y):
    select_trees = set()
    while len(select_trees) < len(candidate_forest) * transfer_rate:
        select_trees.add(np.random.randint(0, len(candidate_forest)))
    for idx in select_trees:
        gen_tree = deepcopy(candidate_forest[idx])
        seeds = set()
        while len(seeds) < GSC:
            seeds.add(np.random.randint(
                0, candidate_forest[idx].vector.shape[0]))
        for seed in seeds:
            gen_tree.vector[seed] = 0 if gen_tree.vector[seed] == 1 else 1
        gen_tree.CA = get_CA(gen_tree.vector, X, Y)
        candidate_forest.append(gen_tree)
    return forest, candidate_forest


def update_best_tree(forest, candidate_forest):
    if len(candidate_forest) == 0:
        return forest, candidate_forest

    candidate_forest.sort(key=lambda tree: tree.CA, reverse=True)
    candidate_forest[0].age = 0
    forest.append(candidate_forest[0])
    return forest, candidate_forest[1:]


def FSFOA(X, Y, LSC, life_time, area_limit, transfer_rate, GSC):
    """
    X: features, size (n, m), ndarray format
    Y: labels, size (n), ndarray format
    """
    _, m = X.shape

    # Initialize trees
    forest = [Tree(m, X, Y) for _ in range(area_limit // 10)]
    candidate_forest = []

    training_info = []
    for epoch in tqdm(range(cfg.epoch)):
        training_info.append({'epoch': epoch, 'CA': forest[0].CA, 'len(forest)': len(
            forest), 'len(candidate_forest)': len(candidate_forest)})
        forest = local_seeding(forest, LSC, X, Y)
        forest, candidate_forest = population_limiting(
            forest, candidate_forest, life_time, area_limit)
        forest, candidate_forest = global_seeding(
            forest, candidate_forest, transfer_rate, GSC, X, Y)
        forest, candidate_forest = update_best_tree(forest, candidate_forest)
        # choose the best CA
        forest.sort(key=lambda tree: tree.CA, reverse=True)

    return forest[0].vector, training_info

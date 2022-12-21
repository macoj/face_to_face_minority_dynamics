import random
import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform, pdist
from numpy.linalg import norm


class HomophilyModel:
    def __init__(self, n=200, d=1.0, v=1.0, L=100, balance=0.5, h_ii=0.8):
        self.n = n  # number of agents
        self.d = d  # interaction distance threshold
        self.v = v  # step length
        self.L = L  # side size of the square box

        # uniformly distributed attractiveness
        self.a = np.random.rand(self.n)

        # random positions
        self.positions = np.array(list(zip(
            np.random.rand(self.n) * self.L,
            np.random.rand(self.n) * self.L)))

        # random activation with probability 0.5
        self.active = np.random.randint(0, 2, self.n) == 1

        # activation probability
        self.r = np.random.rand(self.n)

        # social trajectories
        self.trajectories = []

        # edges
        self.edges = {}

        self.iterations = 0

        self.current_neighborhood = None

        # the 'type of the node'
        n_0 = int(self.n * balance)
        n_1 = self.n - n_0
        self.t = np.array([0]*n_0 + [1]*n_1)
        np.random.shuffle(self.t)

        if type(h_ii) != tuple:
            h_00, h_11 = h_ii, h_ii
        else:
            h_00, h_11 = h_ii

        # the homophily matrix:
        self.h = np.array([[h_00, 1 - h_00],
                           [1 - h_11, h_11]])

    def run(self, number_of_iterations, verbose=False, max_edges=None, **kargs):
        import datetime
        iteration = 0
        while True:
            self.iteration(**kargs)
            self.iterations += 1
            if verbose:
                if not iteration % 1000:
                    print("[%d]: %s (edges: %d / interactions: %d)" % (
                        self.iterations, datetime.datetime.now(), len(self.edges), len(self.trajectories)))
            iteration += 1
            if max_edges is not None:
                if len(self.edges) >= max_edges:
                    break
            elif iteration >= number_of_iterations:
                break

    def iteration(self):
        self.update_neighborhood()
        for i in range(self.n):
            if self.active[i]:
                max_aj = 0 if len(self.current_neighborhood[i]) == 0 else max(
                    self.a[self.current_neighborhood[i]]  # the attractiveness of each j
                )
                p_i = 1 - max_aj

                to_move = p_i > random.random()

                if to_move:
                    self.move(i)
                else:
                    # if the agent is active and chose to stay
                    max_h_j = 0 if len(self.current_neighborhood[i]) == 0 else max(
                        self.h[
                            [self.t[i]] * len(self.current_neighborhood[i]),  # the type of 'i' many times
                            self.t[self.current_neighborhood[i]]  # the type of each j
                        ])
                    p_i = max_h_j
                    to_interact = p_i > random.random()
                    if to_interact:
                        self.interact_with_highest_h_j_neighbors(i)
                # active isolated individuals become inactive with probability r_i
                if len(self.current_neighborhood[i]) == 0:
                    # (note that the neighborhood is not the updated one, but it should be ok)
                    self.active[i] = self.r[i] > random.random()
            else:
                # inactive nodes become active with probability 1 - r_i
                self.active[i] = 1 - self.r[i] > random.random()

    def update_social_trajectory(self, i):
        for j in self.current_neighborhood[i]:
            if i < j:  # neighborhood is reciprocal, let's avoid duplicates
                self.add_edge_ordered(i, j)

    def add_edge_ordered(self, i, j):
        i_, j_ = min([i, j]), max([i, j])
        self.trajectories.append((self.iterations, i_, j_))
        self.add_edge(i_, j_)

    def interact_with_highest_h_j_neighbors(self, i):
        highest_h_j = float('-inf')
        highest_h_j_neighbors = None
        h_ij = [(self.h[self.t[i], self.t[j]], j) for j in self.current_neighborhood[i]]
        while len(h_ij) > 0:
            value, value_index = h_ij.pop()
            if value > highest_h_j:
                highest_h_j = value
                highest_h_j_neighbors = [value_index]
            elif value == highest_h_j:
                highest_h_j_neighbors.append(value_index)
        for j in highest_h_j_neighbors:
            self.add_edge_ordered(i, j)

    def interact_only_with_highest_h_j_neighbor(self, i):
        highest_h_j = float('-inf')
        highest_h_j_neighbor = None
        for j in self.current_neighborhood[i]:
            if self.h[self.t[i], self.t[j]] > highest_h_j:
                highest_h_j = self.h[self.t[i], self.t[j]]
                highest_h_j_neighbor = j
        self.add_edge_ordered(i, highest_h_j_neighbor)

    def move(self, i):
        angle = random.random() * 2 * np.pi
        self.positions[i] += [self.v * np.sin(angle), self.v * np.cos(angle)]
        self.positions[i] %= self.L

    def update_neighborhood(self):
        # we only calculate distance between active nodes because only active nodes can be part of neighborhoods
        # but for this we need to play a bit with the indices!
        distance_df = pd.DataFrame(self.distance_in_a_periodic_box(self.positions[self.active], self.L))
        within_d = (distance_df <= self.d).values
        indices = distance_df.index.values

        # indices of in/active nodes
        active_nodes = np.arange(self.n)[self.active]
        inactive_nodes = np.arange(self.n)[~self.active]

        # building the neighborhood:
        neighbors = {}
        for distance_active_i in distance_df.index:
            distance_neighbors = indices[within_d[distance_active_i, :]]
            # translating the 'distance index' to actual model index:
            model_i = active_nodes[distance_active_i]
            neighbors[model_i] = active_nodes[distance_neighbors]
            neighbors[model_i] = neighbors[model_i][neighbors[model_i] != model_i]
        for i in inactive_nodes:
            neighbors[i] = []
        self.current_neighborhood = neighbors
        return self.current_neighborhood

    def get_social_interactions_data_frame(self):
        return pd.DataFrame(self.trajectories, columns=['time', 'i', 'j'])

    def get_types_data_frame(self):
        return pd.DataFrame(self.t, columns=['type'])

    def get_attractiveness_data_frame(self):
        return pd.DataFrame(self.a, columns=['attractiveness'])

    @staticmethod
    def distance_in_a_periodic_box(points, boundary):
        out = np.empty((2, points.shape[0] * (points.shape[0] - 1) // 2))
        for o, i in zip(out, points.T):
            pdist(i[:, None], 'cityblock', out=o)
        out[out > boundary / 2] -= boundary
        return squareform(norm(out, axis=0))

    def add_edge(self, i, j):
        edge = (i, j)
        if edge not in self.edges:
            self.edges[edge] = 1
        else:
            self.edges[edge] += 1

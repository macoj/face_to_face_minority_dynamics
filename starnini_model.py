import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform, pdist
from numpy.linalg import norm


class StarniniModel:
    def __init__(self, n=200, d=1.0, v=1.0, L=100):
        self.n = n  # number of agents
        self.d = d  # interaction distance threshold
        self.v = v  # step length
        self.L = L  # side size of the square box

        # uniformly distributed attractiveness
        self.a = np.random.rand(self.n)

        # random positions
        self.positions = np.array(zip(np.random.rand(self.n) * self.L, np.random.rand(self.n) * self.L))

        # random activation with probability 0.5
        self.active = np.random.randint(0, 2, self.n) == 1

        # activation probability
        self.r = np.random.rand(self.n)

        # social trajectories
        self.trajectories = []

        self.iterations = 0

        self.current_neighborhood = None

    def run(self, number_of_iterations, verbose=False):
        import datetime
        for _ in range(number_of_iterations):
            self.iteration()
            self.iterations += 1
            if verbose:
                if not _ % 100:
                    print "[%d]: %s" % (self.iterations, datetime.datetime.now())

    def iteration(self):
        self.update_neighborhood()
        for i in range(self.n):
            if self.active[i]:
                isolated = len(self.current_neighborhood[i]) == 0
                # active agent chooses to move based on the attractiveness of its neighborhood
                max_aj = 0 if isolated else max(self.a[self.current_neighborhood[i]])
                p_i = 1 - max_aj
                to_move = p_i > np.random.rand()
                if to_move:
                    self.move(i)
                else:
                    # if the agent is active and chose to stay:
                    self.update_social_trajectory(i)
                # active isolated individuals become inactive with probability r_i
                if isolated:
                    # (note that we do not re-update the neighborhood at this point, but it should be ok)
                    self.active[i] = self.r[i] > np.random.rand()
            else:
                # inactive nodes become active with probability 1 - r_i
                self.active[i] = 1 - self.r[i] > np.random.rand()

    def update_social_trajectory(self, i):
        for j in self.current_neighborhood[i]:
            if i < j:
                self.trajectories.append((self.iterations, i, j))

    def move(self, i):
        angle = np.random.rand() * 2 * np.pi
        self.positions[i] += [self.v * np.sin(angle), self.v * np.cos(angle)]
        self.positions[i] %= self.L

    def update_neighborhood(self):
        # we only calculate distance between active nodes because only active nodes can be part of neighborhoods
        # but for this we need to play a bit with the indices!
        distance_df = pd.DataFrame(self.distance_in_a_periodic_box(self.positions[self.active], self.L))

        # indices of in/active nodes
        active_nodes = np.arange(self.n)[self.active]
        inactive_nodes = np.arange(self.n)[~self.active]

        # building the neighborhood:
        neighbors = {}
        for distance_active_i in distance_df.index:
            distance_neighbors = distance_df.loc[distance_active_i][
                distance_df.loc[distance_active_i] <= self.d].index.values
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

    @staticmethod
    def distance_in_a_periodic_box(points, boundary):
        out = np.empty((2, points.shape[0] * (points.shape[0] - 1) // 2))
        for o, i in zip(out, points.T):
            pdist(i[:, None], 'cityblock', out=o)
        out[out > boundary / 2] -= boundary
        return squareform(norm(out, axis=0))

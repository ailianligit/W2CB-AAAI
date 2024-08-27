import numpy as np
import torch
from abc import ABC, abstractmethod
import pandas as pd

def inv_sherman_morrison(u, A_inv):
    """Inverse of a matrix with rank 1 update.
    """
    Au = np.dot(A_inv, u)
    A_inv -= np.outer(Au, Au)/(1+np.dot(u.T, Au))

    # device = A_inv.device
    # u = u.to(device)
    # Au = torch.matmul(A_inv, u)
    # A_inv -= torch.outer(Au, Au) / (1 + torch.dot(u, Au))
    return A_inv

def minimax(vec):
    vec = vec - np.min(vec)
    vec = (vec - np.min(vec)) / (np.max(vec) - np.min(vec))
    vec = vec / np.sum(vec)
    return vec

class UCB(ABC):
    def __init__(self, d, confidence_scaling_factor, reg_factor, delta, beta, bound_features):
        self.n_features = d
        self.confidence_scaling_factor = confidence_scaling_factor
        self.bound_features = bound_features
        self.reg_factor = reg_factor
        self.delta = delta
        self.beta = beta

    @property
    @abstractmethod
    def approximator_dim(self):
        """Number of parameters used in the approximator.
        """
        pass

    @property
    @abstractmethod
    def alpha(self):
        """Multiplier for the confidence exploration bonus.
        To be defined in children classes.
        """
        pass

    @abstractmethod
    def ucb_multiplier(self, context):
        """Compute output gradient of the approximator w.r.t its parameters.
        """
        pass

    @abstractmethod
    def update(self, policy, scores, contexts):
        """Update approximator.
        To be defined in children classes.
        """
        pass

    ###
    @abstractmethod
    def predict(self, context):
        """Predict rewards based on an approximator.
        To be defined in children classes.
        """
        pass

    ###
    def update_A_inv(self, multiplier):
        self.A_inv = inv_sherman_morrison(
            multiplier,
            self.A_inv
        )

    def select_arms(self, budget, contexts, config):
        selected_arms = []

        p_array = np.zeros(len(contexts))
        for i in range(len(contexts)):
            ###
            p_array[i] = self.predict(contexts[i])

        # p_array = minimax(p_array)
        print(p_array)

        # for k in range(budget):
        #     r_array = np.zeros(len(contexts))
        #     for i in range(len(contexts)):
        #         if i in selected_arms or (i not in config['available']):
        #             continue
        #         r_array[i] = p_array[i]
        #     selected_arms.append(np.argmax(r_array))
        
        # policy = tuple(sorted(selected_arms))
        # return policy

        path = f"/home/ubuntu/data/sim_selection/sim_ot_sources/resnet18_{config['dataset']}/size_{config['block_size']}_block_{config['block_num']}_domain_{config['domain_num']}_200.csv"
        df = pd.read_csv(path, header=None)
        for k in range(budget):
            r_array = np.zeros(len(contexts))
            for i in range(len(contexts)):
                if (i in selected_arms) or (i not in config['available']):
                    continue
                diversity = 0.0
                for j in selected_arms:
                    for index, row in df.iterrows():
                        if row[0] == i and row[1] == j:
                            break
                    diversity += df.iloc[index, 2]
                if k >= 1:
                    r_array[i] = p_array[i] + self.beta * diversity / k
                else:
                    r_array[i] = p_array[i]
            selected_arms.append(np.argmax(r_array))
        
        policy = tuple(sorted(selected_arms))
        return policy
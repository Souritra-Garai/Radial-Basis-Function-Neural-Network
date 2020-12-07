import numpy as np
from itertools import permutations

from Neural_Network import Multi_Layer_Neural_Network

def f1(x) :

    return np.tanh(x)

def f1_derivative(x) :

    return 1 - np.square(np.tanh(x))

def f2(x) :

    return np.copy(x)

def f2_derivative(x) :

    return np.ones_like(x)

nn = Multi_Layer_Neural_Network.generate_network_structure(
    [4, 10, 1],
    [f1, f2],
    [f1_derivative, f2_derivative]
)

nn.set_learning_rate(0.1)
nn.set_learning_rate_decay_factor(0.7)
nn.set_learning_rate_growth_factor(1.05)
nn.set_momentum_coefficient(0.9)
nn.set_critical_error_growth_ratio(0.04)

nn.init_network(train_model=True)

A = [161, 112, 165]
B = [92, 163]
C = [71, 195]

unlock_vectors = []

for i, j in permutations(A, 2) :

    for k in B :

        for l in C :

            unlock_vectors.append([i, j, k, l])

num_training_vectors = 1000 # > 24
train_vectors = np.append(  np.transpose(unlock_vectors),
                            np.random.randint(0, 256, (4, num_training_vectors - 24)),
                            axis=1  )

train_targets = np.zeros((1, num_training_vectors), dtype=int)
train_targets[0, :24] = 1

for i in range(24, num_training_vectors) :

    if np.all(train_vectors[:, i] == unlock_vectors, axis=1).any() :

        train_targets[0, i] = 1

for i in range(100) :
    
    nn.train_model(train_vectors, train_targets)

num_test_vectors = 10000 # > 24
test_vectors = np.append(   np.transpose(unlock_vectors),
                            np.random.randint(0, 256, (4, num_test_vectors - 24)),
                            axis=1  )

test_results = nn.predict(test_vectors)

my_min = test_results[0, :24].min()
prob = 0

for i in range(24, num_test_vectors) :

    if np.all(test_vectors[:, i] == unlock_vectors, axis=1).any() :

        if test_results[0, i] >= my_min :

            prob += 1

    else :

        if test_results[0, i] < my_min :

            prob += 1

print(prob / num_test_vectors)


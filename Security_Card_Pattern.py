import numpy as np

from itertools import permutations
from Radial_Basis_Network import Radial_Basis_Neural_Network

A = [161, 112, 165]
B = [92, 163]
C = [71, 195]

num_unlock_vectors = 24
unlock_vectors = np.zeros((4, 24), int)
m = 0
for i, j in permutations(A, 2) :

    for k in B :

        for l in C :

            unlock_vectors[:, m] = [i, j, k, l]
            m += 1

train_targets = np.ones((1, num_unlock_vectors), dtype=int) * 255

nn = Radial_Basis_Neural_Network()
nn.set_number_of_Inputs(4)
nn.set_number_of_Outputs(1)
nn.set_number_of_RBFs(12)

nn.init_network()

print('Training Model...')
nn.train_model(unlock_vectors, train_targets)
print('Done\n')

num_test_vectors = 10000024 # > 24
test_vectors = np.append(   unlock_vectors,
                            np.random.randint(0, 256, (4, num_test_vectors - 24)),
                            axis=1  )

test_results = nn.predict(test_vectors)

my_min = test_results[0, :24].min()
prob = 0

print('Performing Tests...')

for i in range(24, num_test_vectors) :

    if np.all(test_vectors[:, i:i+1] == unlock_vectors, axis=0).any() :

        if test_results[0, i] >= my_min :

            prob += 1

    else :

        if test_results[0, i] < my_min :

            prob += 1

print('Tests Completed\n')
print('Test Results :', prob / (num_test_vectors-24))
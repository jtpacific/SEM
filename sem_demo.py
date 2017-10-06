import numpy as np
import random
import matplotlib.pyplot as plt

import sem
import simple_rnn
import story_generation

dim = 5
train_len = 100
test_len = 10

# initialize sem structure
opts = sem.sem_options(None, dim, eta = 0.2, alpha = 1.0, beta = 2.0, lambd = 5000)
sem_obj = sem.sem_init(opts)

# generate training and testing scene sequences (testing is just a smaller # of events to simplify evaluation) 
training, generating_train, testing, generating_test = story_generation.generate_stories(train_len, test_len, dim)

# perform segmentation 
train_events, train_predictions = sem.sem_segment(training, sem_obj, opts)
test_events, test_predictions = sem.sem_segment(testing, sem_obj, opts)

ground_truth = []
ground_truth.extend(training)
predictions = []
predictions.extend(train_predictions)
del ground_truth[0]

plt.figure(1)
plt.subplot(211)
plt.scatter(range(len(generating_test)), generating_test, color='r')
plt.scatter(range(len(test_events)), test_events, color='b')

plt.show()

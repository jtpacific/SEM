import random
import sem
import numpy as np
import matplotlib.pyplot as plt

def generate(events):
    seqs = []
    generating = []
    for event in range(events):
        # random length and offset events
        length = np.random.randint(15, 20)
        start = np.random.randint(0, 10)
        base = range(start, start + length)

        # randomly choose either sin or linear function to generate each event
        category = np.random.randint(1, 3)
        if category == 1:
            # sin
            freq = np.random.randint(1, 4)
            for i in base:
                generating.append(category)
                seqs.append([np.sin(i / float(freq)) + (np.random.rand() - 0.5) * 0.1])
        elif category == 2:
            # linear
            slope = (np.random.rand() - 0.5) * 2
            for i in base:
                generating.append(category)
                seqs.append([i * slope / base[-1] + (np.random.rand() - 0.5) * 0.1])
    return seqs, generating

# initialize sem structure
opts = sem.sem_options(None, 1 , eta = 0.2, alpha = 0.5, lambd = 0.1)
sem_obj = sem.sem_init(opts)

training, generating_train = generate(100)
sem.sem_segment(training, sem_obj, opts)

testing, generating_test = generate(5)
test_events, test_predictions = sem.sem_segment(testing, sem_obj, opts)

ground_truth = []
ground_truth.extend(list(map(lambda x: x[0], testing)))
ground_truth.append(None)
predictions = []
predictions.append(None)
predictions.extend(test_predictions)

plt.figure(1)
plt.subplot(211)
plt.plot(range(len(ground_truth)), ground_truth, color='r')
plt.scatter(range(len(predictions)), predictions, color='b')

plt.subplot(212)
plt.scatter(range(len(generating_test)), generating_test, color='r')
plt.scatter(range(len(test_events)), test_events, color='b')
plt.show()
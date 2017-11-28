from hrr import encode, decode, embed_2d, normalize
import numpy as np
import random
# import event generators
from coffeeshop_generator import enter_coffeeshop
from mall_generator import enter_mall

EVENT_GENERATORS = [enter_coffeeshop, enter_mall]
ENCODING_SCHEMES = ['causal_attention', 'trait_addition', 'trait_boolean_binding']

# generate events with ground truth fillers using a single generator
def generate_evaluation_events(context, encoding, generator = EVENT_GENERATORS[0], num_events = 1, testing = False):
    events = []
    generating_fillers_list = []
    for i in range(num_events):
        if testing:
            ind = np.random.permutation(len(context.test_actors))
            variables = {}
            variables['p1'] = context.test_actors[ind[0]]
            variables['p2'] = context.test_actors[ind[1]]
        else: 
            ind = np.random.permutation(len(context.train_actors))
            variables = {}
            variables['p1'] = context.train_actors[ind[0]]
            variables['p2'] = context.train_actors[ind[1]]
        event, generating_fillers = generate_event(variables, context, encoding, generator)
        events.append(event)
        generating_fillers_list.append(generating_fillers)
    return events, generating_fillers_list

def generate_event(variables, context, encoding, generator):
    event = []
    generating_fillers = []
    while (generator != None):
        generator = generator(event, generating_fillers, variables, context, encoding)
    return event, generating_fillers
from hrr import encode, decode, embed_2d, normalize
import numpy as np
import random
# import event generators
from coffeeshop_generator import enter_coffeeshop
from mall_generator import enter_mall

EVENT_GENERATORS = [enter_coffeeshop, enter_mall]

# generate stream of scenes from random sequence of generators
def generate_segmentation_scenes(context, encoding, generators = EVENT_GENERATORS, num_events = 1):
    scenes = []
    generator_indices = []
    for i in range(num_events):
        generator_index = random.randint(0, len(generators) - 1)
        generator = generators[generator_index]
        ind = np.random.permutation(len(context.train_actors))
        variables = {}
        variables['p1'] = context.train_actors[ind[0]]
        variables['p2'] = context.train_actors[ind[1]]
        event, generating_fillers = generate_event(variables, context, encoding, generator)
        scenes.extend(event)
        for i in range(len(event)):
            generator_indices.append(generator_index)
    return scenes, generator_indices

# generate events with ground truth fillers using a single generator
def generate_evaluation_events(context, encoding, generator = EVENT_GENERATORS[0], num_events = 1, testing = False):
    events = []
    generating_fillers_list = []
    chosen_properties_list = []
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
        event, generating_fillers, chosen_properties = generate_event(variables, context, encoding, generator)
        events.append(event)
        generating_fillers_list.append(generating_fillers)
        chosen_properties_list.append(chosen_properties)
    if 'bayesian' in encoding:
        return events, generating_fillers_list, chosen_properties_list
    return events, generating_fillers_list

def generate_event(variables, context, encoding, generator):
    event = []
    generating_fillers = []
    chosen_properties = []
    while (generator != None):
        generator = generator(event, generating_fillers, chosen_properties, variables, context, encoding)
    return event, generating_fillers, chosen_properties
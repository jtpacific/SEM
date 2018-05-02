from hrr import encode, decode, embed_2d, normalize
import numpy as np
import random
from coffeeshop_generator import enter_coffeeshop_fight
from poetry_generator import enter_coffeeshop_poetry

EVENT_GENERATORS = [enter_coffeeshop_fight, enter_coffeeshop_poetry]

# generate list of scenes and ground truth labels from random sequence of generators
def generate_segmentation_scenes(context, encoding, generators = EVENT_GENERATORS, num_events = 10, testing = False):
    scenes = []
    generator_indices = []
    if testing:
        actors = context.test_actors
    else:
        actors = context.train_actors
    for i in range(num_events):
        generator_index = random.randint(0, len(generators) - 1)
        generator = generators[generator_index]
        ind = np.random.permutation(len(actors))
        variables = {}
        variables['p1'] = actors[ind[0]]
        variables['p2'] = actors[ind[1]]
        variables['p3'] = actors[ind[2]]
        event, generating_fillers, chosen_properties = generate_event(variables, context, encoding, generator)
        scenes.extend(event)
        for i in range(len(event)):
            generator_indices.append(generator_index)
    return scenes, generator_indices

# generate list of events and ground truth labels from random sequence of generators
def generate_clustering_events(context, encoding, generators = EVENT_GENERATORS, num_events = 10, testing = False):
    events = []
    generator_indices = []
    if testing:
        actors = context.test_actors
    else:
        actors = context.train_actors
    for i in range(num_events):
        generator_index = random.randint(0, len(generators) - 1)
        generator = generators[generator_index]
        ind = np.random.permutation(len(actors))
        variables = {}
        variables['p1'] = actors[ind[0]]
        variables['p2'] = actors[ind[1]]
        variables['p3'] = actors[ind[2]]
        event, generating_fillers, chosen_properties = generate_event(variables, context, encoding, generator)
        events.append(event)
        generator_indices.append(generator_index)
    return events, generator_indices

# generate events with ground truth fillers using a single generator
def generate_evaluation_events(context, encoding, generator = EVENT_GENERATORS[0], num_events = 10, testing = False):
    events = []
    generating_fillers_list = []
    chosen_properties_list = []
    if testing:
        actors = context.test_actors
    else:
        actors = context.train_actors
    for i in range(num_events):
        ind = np.random.permutation(len(actors))
        variables = {}
        variables['p1'] = actors[ind[0]]
        variables['p2'] = actors[ind[1]]
        variables['p3'] = actors[ind[2]]
        event, generating_fillers, chosen_properties = generate_event(variables, context, encoding, generator)
        events.append(event)
        generating_fillers_list.append(generating_fillers)
        chosen_properties_list.append(chosen_properties)
    if 'bayesian' in encoding:
        return events, generating_fillers_list, chosen_properties_list
    return events, generating_fillers_list

# generate single event using a given generator by recursively calling functions of the generator
def generate_event(variables, context, encoding, generator):
    event = []
    generating_fillers = []
    chosen_properties = []
    while (generator != None):
        generator = generator(event, generating_fillers, chosen_properties, variables, context, encoding)
    return event, generating_fillers, chosen_properties
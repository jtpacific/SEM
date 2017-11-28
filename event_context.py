from hrr import encode, decode, embed_2d, normalize
import numpy as np
import random

def vectorize_roles(role_strings, dim):
    roles = {}
    for i in range(len(role_strings)):
        roles[role_strings[i]] = embed_2d(dim, None)
    return roles

# vectorize list of strings
def vectorize(strings, dim):
    vectors = {}
    for i in strings:
        vectors[i] = embed_2d(dim, None)
    return vectors

# vectorize list of strings with properties
def vectorize_with_properties(strings, dim, properties = []):
    objects = {}
    for i in strings:
        o = {}
        o['identity'] = embed_2d(dim, None)
        # all possible properties that the object may have
        o['properties'] = {}
        for j in properties:
            # 1 if object actually possesses the property
            o['properties'][j] = random.random() < 0.5
        objects[i] = o
    return objects

def vectorize_actors(num, dim, properties = []):
    actors = []
    for i in range(num):
        a = {}
        a['identity'] = embed_2d(dim, None)
        # all possible properties that the object may have
        a['properties'] = {}
        for j in properties:
            # 1 if object actually possesses the property
            a['properties'][j] = random.random() < 0.5
        actors.append(a)
    return actors

class EventContext(object):
    def __init__(self, dim, noun_strings, verb_strings, property_strings, role_strings = ['subject', 'action', 'object', 'noun', 'verb', 'property'], constant_strings = ['TRUE', 'FALSE', 'NULL'], num_train_actors = 10, num_test_actors = 10):
        self.roles = dim
        # initialize vectors
        self.roles = vectorize_roles(role_strings, dim)
        self.properties = vectorize(property_strings, dim)
        self.constants = vectorize(constant_strings, dim)
        # each filler has an identity vector as well as a list of properties
        self.train_actors = vectorize_actors(num_train_actors, dim, self.properties)
        self.test_actors = vectorize_actors(num_test_actors, dim, self.properties)
        self.nouns = vectorize_with_properties(noun_strings, dim)
        self.verbs = vectorize_with_properties(verb_strings, dim)
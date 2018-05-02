from hrr import encode, decode, embed_2d, normalize
import numpy as np
import random
from scipy import spatial

# TODO: refactor to decouple property encoding scheme and property inclusion class
# TODO: patient oriented implementation of encoding methods

#-----encoding and decoding functions-----

def encode_scene(context, encoding, chosen_properties, subj, action, obj, agent_property = [], action_property = [], patient_property = []):
    if encoding == 'selective_property_addition':
        scene, generating = encode_selective_property_addition(subj, action, obj, context, agent_property = agent_property, action_property = action_property, patient_property = patient_property)
    elif encoding == 'selective_property_binding':
        scene, generating = encode_selective_property_binding(subj, action, obj, context, agent_property = agent_property, action_property = action_property, patient_property = patient_property)
    elif encoding == 'all_property_addition':
        scene, generating = encode_all_property_addition(subj, action, obj, context)
    elif encoding == 'all_property_binding':
        scene, generating = encode_all_property_binding(subj, action, obj, context)
    else:
        scene, generating = encode_baseline(subj, action, obj, context)
    return scene, generating

# decode filler vectors using role vectors for scenes encoded using direct property addition
def decode_prediction(vector, context):
    subj = decode(vector, context.roles['agent'])
    action = decode(vector, context.roles['action'])
    obj = decode(vector, context.roles['patient'])
    predictions = {}
    predictions['agent'] = subj
    predictions['action'] = action
    predictions['patient'] = obj
    return predictions

# decode filler vectors using role vectors for scenes encoded using memory trace addition
def decode_prediction_binding(vector, context):
    subj = decode(vector, context.roles['agent'])
    subj_noun = decode(subj, context.roles['agent_id'])
    action = decode(vector, context.roles['action'])
    action_verb = decode(action, context.roles['action_id'])
    obj = decode(vector, context.roles['patient'])
    obj_noun = decode(obj, context.roles['patient_id'])
    predictions = {}
    predictions['agent'] = subj_noun
    predictions['action'] = action_verb
    predictions['patient'] = obj_noun
    return predictions

# calculate match errors and euclidean errors for each role
def filler_errors(encoding, vector, generating_fillers, context, testing = False):
    # TODO: clean up hack for dealing with pretrained encoding
    if type(vector) == dict:
        predictions = vector
    elif 'binding' in encoding:
        predictions = decode_prediction_binding(vector, context)
    else:
        predictions = decode_prediction(vector, context)

    if testing:
        noun_pool = list(map(lambda x: add_properties(x, context), context.test_actors)) + list(map(lambda x: add_properties(x, context), context.nouns.values()))
    else:
        noun_pool = list(map(lambda x: add_properties(x, context), context.train_actors)) + list(map(lambda x: add_properties(x, context), context.nouns.values()))
    verb_pool = list(map(lambda x: add_properties(x, context), context.verbs.values()))

    agent_match_error = list(match(predictions['agent'], noun_pool)) == list(add_properties(generating_fillers['agent'], context)) 
    action_match_error = list(match(predictions['action'], verb_pool)) == list(add_properties(generating_fillers['action'], context))
    patient_match_error = list(match(predictions['patient'], noun_pool)) == list(add_properties(generating_fillers['patient'], context))
    agent_error = np.linalg.norm(normalize(predictions['agent']) - normalize(add_properties(generating_fillers['agent'], context)))
    action_error = np.linalg.norm(normalize(predictions['action']) - normalize(add_properties(generating_fillers['action'], context)))
    patient_error = np.linalg.norm(normalize(predictions['patient']) - normalize(add_properties(generating_fillers['patient'], context)))

    return [1 - float(agent_match_error), 1 - float(action_match_error), 1 - float(patient_match_error), float(agent_error), float(action_error), float(patient_error)]

def get_match(encoding, vector, context, testing = False):
    if 'binding' in encoding:
        predictions = decode_prediction_binding(vector, context)
    else:
        predictions = decode_prediction(vector, context)

    if testing:
        noun_pool = list(map(lambda x: add_properties(x, context), context.test_actors)) + list(map(lambda x: add_properties(x, context), context.nouns.values()))
    else:
        noun_pool = list(map(lambda x: add_properties(x, context), context.train_actors)) + list(map(lambda x: add_properties(x, context), context.nouns.values()))
    verb_pool = list(map(lambda x: add_properties(x, context), context.verbs.values()))

    decoded = {}
    decoded['agent'] = list(match(predictions['agent'], noun_pool)) #== list(add_properties(generating_fillers['agent'], context)) 
    decoded['action'] = list(match(predictions['action'], verb_pool)) #== list(add_properties(generating_fillers['action'], context))
    decoded['patient'] = list(match(predictions['patient'], noun_pool)) #== list(add_properties(generating_fillers['patient'], context))
    #agent_error = np.linalg.norm(normalize(predictions['agent']) - normalize(add_properties(generating_fillers['agent'], context)))
    #action_error = np.linalg.norm(normalize(predictions['action']) - normalize(add_properties(generating_fillers['action'], context)))
    #patient_error = np.linalg.norm(normalize(predictions['patient']) - normalize(add_properties(generating_fillers['patient'], context)))
    return decoded


#-----helpers-----

# helper function for property encoding
def add_properties(actor, context):
    #print (actor['identity'])
    #print actor
    actor_vector = np.zeros(context.dim) + actor['identity']
    for p in actor['properties'].keys():
        actor_vector += context.properties[p][actor['properties'][p]]
    #print "done"
    #print ""
    return actor_vector

# obtain baseline encoding from fillers: used for asymmetric property inclusion classes
def fillers_to_baseline(fillers, context):
    vector = normalize(encode(context.roles['agent'], fillers['agent']['identity']) + encode(context.roles['action'], fillers['action']['identity']) + encode(context.roles['patient'], fillers['patient']['identity']))
    return vector

# map vector to closest element of vector pool in cosine proximity
def match(vector, pool):
    dists = list(map(lambda x: spatial.distance.cosine(np.asarray(vector), np.asarray(x)), pool))
    index = dists.index(min(dists))
    return pool[index]

# obtain filler's property associated with a property choice, distinguishing between boolean and categorial properties
def retrieve_property(filler, property, context):
    # boolean adjective property
    if set(context.properties[property].keys()) == set([False, True]):
        return context.properties[property][int(filler['properties'][property])]
    # categorial noun property
    return context.nouns[filler['properties'][property]]['identity']

def bool_string(boolean):
    if boolean:
        return 'TRUE'
    return 'FALSE'

#-----baseline-----

def encode_baseline(subj, action, obj, context):
    generating_fillers = {}
    generating_fillers['agent'] = subj
    generating_fillers['action'] = action
    generating_fillers['patient'] = obj
    agent_vector = np.zeros(context.dim) + subj['identity']
    action_vector = np.zeros(context.dim) + action['identity']
    patient_vector = np.zeros(context.dim) + obj['identity']
    vector = normalize(encode(context.roles['agent'], agent_vector) + encode(context.roles['action'], action_vector) + encode(context.roles['patient'], patient_vector))
    return vector, generating_fillers

#-----selective property addition-----

def encode_selective_property_addition(subj, action, obj, context, agent_property = [], action_property = [], patient_property = []):
    generating_fillers = {}
    generating_fillers['agent'] = subj
    generating_fillers['action'] = action
    generating_fillers['patient'] = obj
    agent_vector = np.zeros(context.dim) + subj['identity']
    action_vector = np.zeros(context.dim) + action['identity']
    patient_vector = np.zeros(context.dim) + obj['identity']
    for p in agent_property:
        if p in subj['properties'].keys():
            agent_vector += retrieve_property(subj, p, context)
    for p in action_property:
        if p in action['properties'].keys():
            action_vector += retrieve_property(action, p, context)
    for p in patient_property:
        if p in obj['properties'].keys():
            patient_vector += retrieve_property(obj, p, context)
    vector = normalize(encode(context.roles['agent'], agent_vector) + encode(context.roles['action'], action_vector) + encode(context.roles['patient'], patient_vector))
    return vector, generating_fillers

#-----all property addition-----

def encode_all_property_addition(subj, action, obj, context):
    generating_fillers = {}
    generating_fillers['agent'] = subj
    generating_fillers['action'] = action
    generating_fillers['patient'] = obj
    agent_vector = np.zeros(context.dim) + subj['identity']
    action_vector = np.zeros(context.dim) + action['identity']
    patient_vector = np.zeros(context.dim) + obj['identity']
    for p in subj['properties'].keys():
        agent_vector += retrieve_property(subj, p, context)
    for p in action['properties'].keys():
        action_vector += retrieve_property(action, p, context)
    for p in obj['properties'].keys():
        patient_vector += retrieve_property(obj, p, context)
    vector = normalize(encode(context.roles['agent'], agent_vector) + encode(context.roles['action'], action_vector) + encode(context.roles['patient'], patient_vector))
    return vector, generating_fillers

#-----selective property binding-----

def encode_selective_property_binding(subj, action, obj, context, agent_property = None, action_property = None, patient_property = None):
    generating_fillers = {}
    generating_fillers['agent'] = subj
    generating_fillers['action'] = action
    generating_fillers['patient'] = obj
    agent_vector = encode(context.roles['agent_id'], subj['identity'])
    action_vector = encode(context.roles['action_id'], action['identity'])
    patient_vector = encode(context.roles['patient_id'], obj['identity'])
    if agent_property:
        boolean = bool_string(subj['properties'][agent_property])
        agent_vector += encode(context.properties[agent_property][1], context.constants[boolean])
    if action_property:
        boolean = bool_string(action['properties'][action_property])
        action_vector += encode(context.properties[action_property][1], context.constants[boolean])
    if patient_property:
        boolean = bool_string(obj['properties'][patient_property])
        patient_vector += encode(context.properties[patient_property][1], context.constants[boolean])
    vector = normalize(encode(context.roles['agent'], agent_vector) + encode(context.roles['action'], action_vector) + encode(context.roles['patient'], patient_vector))
    return vector, generating_fillers

#-----all property binding-----

def encode_all_property_binding(subj, action, obj, context):
    generating_fillers = {}
    generating_fillers['agent'] = subj
    generating_fillers['action'] = action
    generating_fillers['patient'] = obj
    agent_vector = encode(context.roles['agent_id'], subj['identity'])
    action_vector = encode(context.roles['action_id'], action['identity'])
    patient_vector = encode(context.roles['patient_id'], obj['identity'])
    for p in subj['properties'].keys():
        boolean = bool_string(subj['properties'][p])
        agent_vector += encode(context.properties[p][1], context.constants[boolean])
    for p in action['properties'].keys():
        boolean = bool_string(action['properties'][p])
        action_vector += encode(context.properties[p][1], context.constants[boolean])
    for p in obj['properties'].keys():
        boolean = bool_string(obj['properties'][p])
        patient_vector += encode(context.properties[p][1], context.constants[boolean])
    vector = normalize(encode(context.roles['agent'], agent_vector) + encode(context.roles['action'], action_vector) + encode(context.roles['patient'], patient_vector))
    return vector, generating_fillers
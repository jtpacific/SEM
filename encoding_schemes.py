from hrr import encode, decode, embed_2d, normalize
import numpy as np
import random
from scipy import spatial

#-----helpers-----

def encode_scene(context, encoding, chosen_properties, subj, action, obj, subject_property = None, action_property = None, object_property = None):
	if encoding == 'baseline':
		scene, generating = encode_baseline(subj, action, obj, context)
	if encoding == 'selective_property_addition':
		scene, generating = encode_selective_property_addition(subj, action, obj, context, subject_property = subject_property, action_property = action_property, object_property = object_property)
	if encoding == 'selective_property_binding':
		scene, generating = encode_selective_property_binding(subj, action, obj, context, subject_property = subject_property, action_property = action_property, object_property = object_property)
	elif encoding == 'all_property_addition':
		scene, generating = encode_all_property_addition(subj, action, obj, context)
	elif encoding == 'all_property_binding':
		scene, generating = encode_all_property_binding(subj, action, obj, context)
	elif encoding == 'bayesian_property_addition': 
		scene, generating = encode_bayesian_property_addition(chosen_properties, subj, action, obj, context)
	elif encoding == 'bayesian_property_binding': 
		scene, generating = encode_bayesian_property_binding(chosen_properties, subj, action, obj, context)
	return scene, generating

def match(vector, pool):
    dists = list(map(lambda x: spatial.distance.cosine(np.asarray(vector), np.asarray(x)), pool))
    index = dists.index(min(dists))
    return pool[index]

def decode_prediction(vector, context):
	subj = decode(vector, context.roles['subject'])
	action = decode(vector, context.roles['action'])
	obj = decode(vector, context.roles['object'])
	predictions = {}
	predictions['subject'] = subj
	predictions['action'] = action
	predictions['object'] = obj
	return predictions

def decode_prediction_binding(vector, context):
	subj = decode(vector, context.roles['subject'])
	subj_noun = decode(subj, context.roles['agent_id'])
	action = decode(vector, context.roles['action'])
	action_verb = decode(action, context.roles['action_id'])
	obj = decode(vector, context.roles['object'])
	obj_noun = decode(obj, context.roles['patient_id'])
	predictions = {}
	predictions['subject'] = subj_noun
	predictions['action'] = action_verb
	predictions['object'] = obj_noun
	return predictions

def add_properties(actor, context):
	actor_vector = np.zeros(context.dim)
	actor_vector += actor['identity']
	for p in actor['properties'].keys():
		actor_vector += context.properties[p][int(actor['properties'][p])]
	return actor_vector

def filler_errors(encoding, vector, generating_fillers, context, testing = False):
	if type(vector) == dict:
		predictions = vector
	else:
		if 'binding' in encoding:
			predictions = decode_prediction_binding(vector, context)
		else:
			predictions = decode_prediction(vector, context)

	if testing:
		noun_pool = list(map(lambda x:add_properties(x, context), context.test_actors)) + list(map(lambda x:x['identity'], context.nouns.values()))
	else:
		noun_pool = list(map(lambda x:x['identity'], context.train_actors)) + list(map(lambda x:x['identity'], context.nouns.values()))
	verb_pool = list(map(lambda x:x['identity'], context.verbs.values()))

	subject_match_error = list(match(predictions['subject'], noun_pool)) == list(add_properties(generating_fillers['subject'], context)) #list(generating_fillers['subject']['identity'])
	action_match_error = list(match(predictions['action'], verb_pool)) == list(generating_fillers['action']['identity'])
	object_match_error = list(match(predictions['object'], noun_pool)) == list(add_properties(generating_fillers['object'], context))
	subject_error = np.linalg.norm(normalize(predictions['subject']) - normalize(add_properties(generating_fillers['subject'], context)))
	action_error = np.linalg.norm(normalize(predictions['action']) - normalize(generating_fillers['action']['identity']))
	object_error = np.linalg.norm(normalize(predictions['object']) - normalize(add_properties(generating_fillers['object'], context)))

	return [float(subject_match_error), float(action_match_error), float(object_match_error), float(subject_error), float(action_error), float(object_error)]

#-----baseline-----

def fillers_to_baseline(fillers, context):
	vector = normalize(encode(context.roles['subject'], fillers['subject']['identity']) + encode(context.roles['action'], fillers['action']['identity']) + encode(context.roles['object'], fillers['object']['identity']))
	return vector

def encode_baseline(subj, action, obj, context):
	generating_fillers = {}
	generating_fillers['subject'] = subj
	generating_fillers['action'] = action
	generating_fillers['object'] = obj
	subject_vector = subj['identity']
	action_vector = action['identity']
	object_vector = obj['identity']
	vector = normalize(encode(context.roles['subject'], subject_vector) + encode(context.roles['action'], action_vector) + encode(context.roles['object'], object_vector))
	return vector, generating_fillers

#-----bayesian property learning-----

def action_from_vector(context, vector):
    for verb in context.verbs.keys():
        if list(context.verbs[verb]['identity']) == list(vector['identity']):
            return verb

def encode_bayesian_property_addition(chosen_properties, subj, action, obj, context):
	generating_fillers = {}
	generating_fillers['subject'] = subj
	generating_fillers['action'] = action
	generating_fillers['object'] = obj
	subject_vector = np.zeros(context.dim)
	action_vector = np.zeros(context.dim)
	object_vector = np.zeros(context.dim)
	subject_vector += subj['identity']
	action_vector +=  action['identity']
	object_vector += obj['identity']

	subject_props = {}
	object_props = {}

	chosen_props = {}
	# arrays of property strings
	chosen_props['subject'] = []
	chosen_props['object'] = []
	# probabilistically choose properties of subject, limited to properties that the subject actually possesses
	for prop in subj['properties'].keys():
		subject_props[prop] = context.distribution_params[action_from_vector(context, action)]['subject'][prop][0] / (context.distribution_params[action_from_vector(context, action)]['subject'][prop][0] + context.distribution_params[action_from_vector(context, action)]['subject'][prop][1])
	total_subject = sum(subject_props.values())
	for prop in subject_props.keys():
		if random.random() < min(subject_props[prop], subject_props[prop] / (total_subject/1.)):
			subject_vector += context.properties[prop][int(subj['properties'][prop])]
			chosen_props['subject'].append(prop)
	# probabilistically choose properties of object, limited to properties that the object actually possesses
	for prop in obj['properties'].keys():
		object_props[prop] = context.distribution_params[action_from_vector(context, action)]['object'][prop][0] / (context.distribution_params[action_from_vector(context, action)]['object'][prop][0] + context.distribution_params[action_from_vector(context, action)]['object'][prop][1])
	total_object = sum(object_props.values())
	for prop in object_props.keys():
		if random.random() < min(object_props[prop], object_props[prop] / (total_object/1.)):
			object_vector += context.properties[prop][int(obj['properties'][prop])]
			chosen_props['object'].append(prop)
	chosen_properties.append(chosen_props)
	vector = normalize(encode(context.roles['subject'], subject_vector) + encode(context.roles['action'], action_vector) + encode(context.roles['object'], object_vector))
	return vector, generating_fillers

#-----selective property addition-----

def encode_selective_property_addition(subj, action, obj, context, subject_property = None, action_property = None, object_property = None):
	generating_fillers = {}
	generating_fillers['subject'] = subj
	generating_fillers['action'] = action
	generating_fillers['object'] = obj
	subject_vector = np.zeros(context.dim)
	action_vector = np.zeros(context.dim)
	object_vector = np.zeros(context.dim)
	subject_vector += subj['identity']
	action_vector +=  action['identity']
	object_vector += obj['identity']
	if subject_property != None and subject_property in subj['properties'].keys():
		subject_vector += context.properties[subject_property][int(subj['properties'][subject_property])]
		generating_fillers['subject_property'] = subject_property
	else: 
		generating_fillers['subject_property'] = None
	if action_property != None and action_property in action['properties'].keys():
		action_vector += context.properties[action_property][int(action['properties'][action_property])]
		generating_fillers['action_property'] = action_property
	else: 
		generating_fillers['action_property'] = None
	if object_property != None and object_property in obj['properties'].keys():
		object_vector += context.properties[object_property][int(obj['properties'][object_property])]
		generating_fillers['object_property'] = object_property
	else: 
		generating_fillers['object_property'] = None
	vector = normalize(encode(context.roles['subject'], subject_vector) + encode(context.roles['action'], action_vector) + encode(context.roles['object'], object_vector))
	return vector, generating_fillers

#-----all property addition-----

def encode_all_property_addition(subj, action, obj, context):
	generating_fillers = {}
	generating_fillers['subject'] = subj
	generating_fillers['action'] = action
	generating_fillers['object'] = obj
	subject_vector = np.zeros(context.dim)
	action_vector = np.zeros(context.dim)
	object_vector = np.zeros(context.dim)
	subject_vector += subj['identity']
	action_vector +=  action['identity']
	object_vector += obj['identity']
	for p in subj['properties'].keys():
		subject_vector += context.properties[p][int(subj['properties'][p])]
	for p in action['properties'].keys():
		action_vector += context.properties[p][int(action['properties'][p])]
	for p in obj['properties'].keys():
		object_vector += context.properties[p][int(obj['properties'][p])]
	vector = normalize(encode(context.roles['subject'], subject_vector) + encode(context.roles['action'], action_vector) + encode(context.roles['object'], object_vector))
	return vector, generating_fillers

#-----selective property binding-----

def encode_selective_property_binding(subj, action, obj, context, subject_property = None, action_property = None, object_property = None):
	generating_fillers = {}
	generating_fillers['subject'] = subj
	generating_fillers['action'] = action
	generating_fillers['object'] = obj
	subject_vector = encode(context.roles['agent_id'], subj['identity'])
	action_vector = encode(context.roles['action_id'], action['identity'])
	object_vector = encode(context.roles['patient_id'], obj['identity'])
	if subject_property:
		boolean = bool_string(subj['properties'][subject_property])
		subject_vector += encode(context.properties[subject_property][1], context.constants[boolean])
	if action_property:
		boolean = bool_string(action['properties'][action_property])
		action_vector += encode(context.properties[action_property][1], context.constants[boolean])
	if object_property:
		boolean = bool_string(obj['properties'][object_property])
		object_vector += encode(context.properties[object_property][1], context.constants[boolean])
	vector = normalize(encode(context.roles['subject'], subject_vector) + encode(context.roles['action'], action_vector) + encode(context.roles['object'], object_vector))
	return vector, generating_fillers

#-----all property binding-----

def bool_string(boolean):
	if boolean:
		return 'TRUE'
	return 'FALSE'

def encode_all_property_binding(subj, action, obj, context):
	generating_fillers = {}
	generating_fillers['subject'] = subj
	generating_fillers['action'] = action
	generating_fillers['object'] = obj
	subject_vector = encode(context.roles['agent_id'], subj['identity'])
	action_vector = encode(context.roles['action_id'], action['identity'])
	object_vector = encode(context.roles['patient_id'], obj['identity'])
	for p in subj['properties'].keys():
		boolean = bool_string(subj['properties'][p])
		subject_vector += encode(context.properties[p][1], context.constants[boolean])
	for p in action['properties'].keys():
		boolean = bool_string(action['properties'][p])
		action_vector += encode(context.properties[p][1], context.constants[boolean])
	for p in obj['properties'].keys():
		boolean = bool_string(obj['properties'][p])
		object_vector += encode(context.properties[p][1], context.constants[boolean])
	vector = normalize(encode(context.roles['subject'], subject_vector) + encode(context.roles['action'], action_vector) + encode(context.roles['object'], object_vector))
	return vector, generating_fillers
from hrr import encode, decode, embed_2d, normalize
import numpy as np

#-----helpers-----

def encode_scene(context, encoding, subj, action, obj, subject_property = None, action_property = None, object_property = None):
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
	return scene, generating

# filler error is obtained by mapping each decoded vector to the closest in the pool of fillers
def filler_errors(encoding, vector, generating_fillers, context, is_training):
	if encoding == 'baseline':
		return baseline_filler_error(vector, generating_fillers, context, is_training)
	if encoding == 'selective_property_addition':
		return selective_property_addition_filler_error(vector, generating_fillers, context, is_training)
	if encoding == 'selective_property_binding':
		return selective_property_binding_filler_error(vector, generating_fillers, context, is_training)
	if encoding == 'all_property_addition':
		return all_property_addition_filler_error(vector, generating_fillers, context, is_training)
	if encoding == 'all_property_binding':
		return all_property_binding_filler_error(vector, generating_fillers, context, is_training)

def match(vector, pool):
    dists = list(map(lambda x: np.linalg.norm(np.asarray(vector) - np.asarray(x)), pool))
    index = dists.index(min(dists))
    return pool[index]

#-----baseline-----

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

def decode_baseline(vector, context):
	subj = decode(vector, context.roles['subject'])
	action = decode(vector, context.roles['action'])
	obj = decode(vector, context.roles['object'])
	predictions = {}
	predictions['subject_noun'] = subj
	predictions['action_verb'] = action
	predictions['object_noun'] = obj
	return predictions

def baseline_filler_error(vector, generating_fillers, context, is_training):
	predictions = decode_baseline(vector, context)
	if is_training:
		noun_pool = list(map(lambda x:x['identity'], context.train_actors)) + list(map(lambda x:x['identity'], context.nouns.values()))
	else:
		noun_pool = list(map(lambda x:x['identity'], context.test_actors)) + list(map(lambda x:x['identity'], context.nouns.values()))
	verb_pool = list(map(lambda x:x['identity'], context.verbs.values()))
	subject_error = float(list(match(predictions['subject_noun'], noun_pool)) == list(generating_fillers['subject']['identity']))
	action_error = float(list(match(predictions['action_verb'], verb_pool)) == list(generating_fillers['action']['identity']))
	object_error = float(list(match(predictions['object_noun'], noun_pool)) == list(generating_fillers['object']['identity']))
	return [subject_error, action_error, object_error]

#-----selective property addition-----

def encode_selective_property_addition(subj, action, obj, context, subject_property = None, action_property = None, object_property = None):
	generating_fillers = {}
	generating_fillers['subject'] = subj
	generating_fillers['action'] = action
	generating_fillers['object'] = obj
	subject_vector = subj['identity']
	action_vector = action['identity']
	object_vector = obj['identity']
	if subject_property != None and subject_property in subj['properties'].keys() and subj['properties'][subject_property]:
		subject_vector += context.properties[subject_property]
		generating_fillers['subject_property'] = subject_property
	else: 
		generating_fillers['subject_property'] = None
	if action_property != None and action_property in action['properties'].keys() and action['properties'][action_property]:
		action_vector += context.properties[action_property]
		generating_fillers['action_property'] = action_property
	else: 
		generating_fillers['action_property'] = None
	if object_property != None and object_property in obj['properties'].keys() and obj['properties'][object_property]:
		object_vector += context.properties[object_property]
		generating_fillers['object_property'] = object_property
	else: 
		generating_fillers['object_property'] = None
	vector = normalize(encode(context.roles['subject'], subject_vector) + encode(context.roles['action'], action_vector) + encode(context.roles['object'], object_vector))
	return vector, generating_fillers

def decode_selective_property_addition(vector, context):
	subj = decode(vector, context.roles['subject'])
	action = decode(vector, context.roles['action'])
	obj = decode(vector, context.roles['object'])
	predictions = {}
	predictions['subject'] = subj
	predictions['action'] = action
	predictions['object'] = obj
	return predictions

def selective_property_addition_filler_error(vector, generating_fillers, context, is_training):
	predictions = decode_selective_property_addition(vector, context)
	if is_training:
		noun_pool = list(map(lambda x:x['identity'], context.train_actors)) + list(map(lambda x:x['identity'], context.nouns.values()))
	else:
		noun_pool = list(map(lambda x:x['identity'], context.test_actors)) + list(map(lambda x:x['identity'], context.nouns.values()))
	verb_pool = list(map(lambda x:x['identity'], context.verbs.values()))
	subject_error = float(list(match(predictions['subject'], noun_pool)) == list(generating_fillers['subject']['identity']))
	action_error = float(list(match(predictions['action'], verb_pool)) == list(generating_fillers['action']['identity']))
	object_error = float(list(match(predictions['object'], noun_pool)) == list(generating_fillers['object']['identity']))
	return [subject_error, action_error, object_error]

#-----selective property binding-----

def encode_selective_property_binding(subj, action, obj, context, subject_property = None, action_property = None, object_property = None):
	generating_fillers = {}
	generating_fillers['subject'] = subj
	generating_fillers['action'] = action
	generating_fillers['object'] = obj
	if subject_property != None and subject_property in subj['properties'].keys() and subj['properties'][subject_property]:
		subject_vector = encode(context.roles['noun'], subj['identity']) + encode(context.roles['property'], context.properties[subject_property])
		generating_fillers['subject_property'] = subject_property
	else: 
		subject_vector = encode(context.roles['noun'], subj['identity']) + encode(context.roles['property'], context.constants['NULL'])
		generating_fillers['subject_property'] = None
	if action_property != None and action_property in action['properties'].keys() and action['properties'][action_property]:
		action_vector = encode(context.roles['verb'], action['identity']) + encode(context.roles['property'], context.properties[action_property])
		generating_fillers['action_property'] = action_property
	else: 
		action_vector = encode(context.roles['verb'], action['identity']) + encode(context.roles['property'], context.constants['NULL'])
		generating_fillers['action_property'] = None
	if object_property != None and object_property in obj['properties'].keys() and obj['properties'][object_property]:
		object_vector = encode(context.roles['noun'], obj['identity']) + encode(context.roles['property'], context.properties[object_property])
		generating_fillers['object_property'] = object_property
	else: 
		object_vector = encode(context.roles['noun'], obj['identity']) + encode(context.roles['property'], context.constants['NULL'])
		generating_fillers['object_property'] = None
	vector = normalize(encode(context.roles['subject'], subject_vector) + encode(context.roles['action'], action_vector) + encode(context.roles['object'], object_vector))
	return vector, generating_fillers

def decode_selective_property_binding(vector, context):
	subj = decode(vector, context.roles['subject'])
	subject_noun = decode(subj, context.roles['noun'])
	subject_property = decode(subj, context.roles['property'])
	action = decode(vector, context.roles['action'])
	action_verb = decode(action, context.roles['verb'])
	action_property = decode(action, context.roles['property'])
	obj = decode(vector, context.roles['object'])
	object_noun = decode(obj, context.roles['noun'])
	object_property = decode(obj, context.roles['property'])
	predictions = {}
	predictions['subject_noun'] = subject_noun
	predictions['subject_property'] = subject_property
	predictions['action_verb'] = action_verb
	predictions['action_property'] = action_property
	predictions['object_noun'] = object_noun
	predictions['object_property'] = object_property
	return predictions

def selective_property_binding_filler_error(vector, generating_fillers, context, is_training):
	predictions = decode_selective_property_binding(vector, context)
	if is_training:
		noun_pool = list(map(lambda x:x['identity'], context.train_actors)) + list(map(lambda x:x['identity'], context.nouns.values()))
	else:
		noun_pool = list(map(lambda x:x['identity'], context.test_actors)) + list(map(lambda x:x['identity'], context.nouns.values()))
	verb_pool = list(map(lambda x:x['identity'], context.verbs.values()))
	subject_error = float(list(match(predictions['subject_noun'], noun_pool)) == list(generating_fillers['subject']['identity']))
	action_error = float(list(match(predictions['action_verb'], verb_pool)) == list(generating_fillers['action']['identity']))
	object_error = float(list(match(predictions['object_noun'], noun_pool)) == list(generating_fillers['object']['identity']))
	return [subject_error, action_error, object_error]

#-----all property addition-----

def encode_all_property_addition(subj, action, obj, context):
	generating_fillers = {}
	generating_fillers['subject'] = subj
	generating_fillers['action'] = action
	generating_fillers['object'] = obj
	subject_vector = subj['identity']
	action_vector = action['identity']
	object_vector = obj['identity']
	for p in subj['properties'].keys():
		if subj['properties'][p]:
			subject_vector += context.properties[p]
	for p in action['properties'].keys():
		if action['properties'][p]:
			action_vector += context.properties[p]
	for p in obj['properties'].keys():
		if obj['properties'][p]:
			object_vector += context.properties[p]
	vector = normalize(encode(context.roles['subject'], subject_vector) + encode(context.roles['action'], action_vector) + encode(context.roles['object'], object_vector))
	return vector, generating_fillers

def decode_all_property_addition(vector, context):
	subj = decode(vector, context.roles['subject'])
	action = decode(vector, context.roles['action'])
	obj = decode(vector, context.roles['object'])

	predictions = {}
	predictions['subject'] = subj
	predictions['action'] = action
	predictions['object'] = obj
	return predictions

def all_property_addition_filler_error(vector, generating_fillers, context, is_training):
	predictions = decode_all_property_addition(vector, context)
	if is_training:
		noun_pool = list(map(lambda x:x['identity'], context.train_actors)) + list(map(lambda x:x['identity'], context.nouns.values()))
	else:
		noun_pool = list(map(lambda x:x['identity'], context.test_actors)) + list(map(lambda x:x['identity'], context.nouns.values()))
	verb_pool = list(map(lambda x:x['identity'], context.verbs.values()))
	subject_error = float(list(match(predictions['subject'], noun_pool)) == list(generating_fillers['subject']['identity']))
	action_error = float(list(match(predictions['action'], verb_pool)) == list(generating_fillers['action']['identity']))
	object_error = float(list(match(predictions['object'], noun_pool)) == list(generating_fillers['object']['identity']))
	return [subject_error, action_error, object_error]

#-----all property binding-----

def encode_all_property_binding(subj, action, obj, context):
	generating_fillers = {}
	generating_fillers['subject'] = subj
	generating_fillers['action'] = action
	generating_fillers['object'] = obj
	subject_vector = encode(context.roles['noun'], subj['identity'])
	action_vector = encode(context.roles['verb'], action['identity'])
	object_vector = encode(context.roles['noun'], obj['identity'])
	for p in subj['properties'].keys():
		if subj['properties'][p]:
			subject_vector += encode(context.properties[p], context.constants['TRUE'])
		else:
			subject_vector += encode(context.properties[p], context.constants['FALSE'])
	for p in action['properties'].keys():
		if action['properties'][p]:
			action_vector += encode(context.properties[p], context.constants['TRUE'])
		else:
			action_vector += encode(context.properties[p], context.constants['FALSE'])
	for p in obj['properties'].keys():
		if obj['properties'][p]:
			object_vector += encode(context.properties[p], context.constants['TRUE'])
		else:
			object_vector += encode(context.properties[p], context.constants['FALSE'])
	vector = normalize(encode(context.roles['subject'], subject_vector) + encode(context.roles['action'], action_vector) + encode(context.roles['object'], object_vector))
	return vector, generating_fillers

def decode_all_property_binding(vector, context):
	subj = decode(vector, context.roles['subject'])
	subject_noun = decode(subj, context.roles['noun'])
	action = decode(vector, context.roles['action'])
	action_verb = decode(action, context.roles['verb'])
	obj = decode(vector, context.roles['object'])
	object_noun = decode(obj, context.roles['noun'])
	predictions = {}
	predictions['subject_noun'] = subject_noun
	predictions['action_verb'] = action_verb
	predictions['object_noun'] = object_noun
	return predictions

def all_property_binding_filler_error(vector, generating_fillers, context, is_training):
	predictions = decode_all_property_binding(vector, context)
	if is_training:
		noun_pool = list(map(lambda x:x['identity'], context.train_actors)) + list(map(lambda x:x['identity'], context.nouns.values()))
	else:
		noun_pool = list(map(lambda x:x['identity'], context.test_actors)) + list(map(lambda x:x['identity'], context.nouns.values()))
	verb_pool = list(map(lambda x:x['identity'], context.verbs.values()))
	subject_error = float(list(match(predictions['subject_noun'], noun_pool)) == list(generating_fillers['subject']['identity']))
	action_error = float(list(match(predictions['action_verb'], verb_pool)) == list(generating_fillers['action']['identity']))
	object_error = float(list(match(predictions['object_noun'], noun_pool)) == list(generating_fillers['object']['identity']))
	return [subject_error, action_error, object_error]
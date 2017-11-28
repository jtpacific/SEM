from hrr import encode, decode, embed_2d, normalize
import numpy as np

def encode_causal_attention(subj, action, obj, context, subject_property = None, action_property = None, object_property = None):
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

def encode_trait_addition(subj, action, obj, context):
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

def encode_trait_boolean_binding(subj, action, obj, context):
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

def causal_attention_errors(vector, generating_fillers, context):
	predictions = decode_causal_attention(vector, context)
	subject_error = np.linalg.norm(generating_fillers['subject']['identity'] - predictions['subject_noun'])
	action_error = np.linalg.norm(generating_fillers['action']['identity'] - predictions['action_verb'])
	object_error = np.linalg.norm(generating_fillers['object']['identity'] - predictions['object_noun'])
	return subject_error, action_error, object_error

def trait_addition_errors(vector, generating_fillers, context):
	predictions = decode_trait_addition(vector, context)
	ground_truth_encoded_subject = generating_fillers['subject']['identity']
	for p in generating_fillers['subject']['properties'].keys():
		if generating_fillers['subject']['properties'][p]:
			ground_truth_encoded_subject += context.properties[p]
	ground_truth_encoded_action = generating_fillers['action']['identity']
	for p in generating_fillers['action']['properties'].keys():
		if generating_fillers['action']['properties'][p]:
			ground_truth_encoded_action += context.properties[p]
	ground_truth_encoded_object = generating_fillers['object']['identity']
	for p in generating_fillers['object']['properties'].keys():
		if generating_fillers['object']['properties'][p]:
			ground_truth_encoded_object += context.properties[p]
	subject_error = np.linalg.norm(normalize(ground_truth_encoded_subject) - predictions['subject'])
	action_error = np.linalg.norm(normalize(ground_truth_encoded_action) - predictions['action'])
	object_error = np.linalg.norm(normalize(ground_truth_encoded_object) - predictions['object'])
	return subject_error, action_error, object_error

def trait_boolean_binding_errors(vector, generating_fillers, context):
	predictions = decode_trait_boolean_binding(vector, context)
	subject_error = np.linalg.norm(generating_fillers['subject']['identity'] - predictions['subject_noun'])
	action_error = np.linalg.norm(generating_fillers['action']['identity'] - predictions['action_verb'])
	object_error = np.linalg.norm(generating_fillers['object']['identity'] - predictions['object_noun'])
	return subject_error, action_error, object_error

def decode_causal_attention(vector, context):
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

def decode_trait_addition(vector, context):
	subj = decode(vector, context.roles['subject'])
	action = decode(vector, context.roles['action'])
	obj = decode(vector, context.roles['object'])
	predictions = {}
	predictions['subject'] = subj
	predictions['action'] = action
	predictions['object'] = obj
	return predictions

def decode_trait_boolean_binding(vector, context):
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
from hrr import encode, decode, embed_2d, normalize
import numpy as np
import random

ACCEPT_THRESHOLD = 1.0

# ----- fight event schema -----

def assign_variable_properties(constants, variable, property_string):
	if property_string in variable.keys() and variable[property_string]:
		return encode(constants['placeholders']['noun'], variable['vector']) + encode(constants['placeholders']['property'], constants['properties'][property_string])
	return encode_null_property(constants, variable['vector'])

def encode_null_property(constants, vector):
	return encode(constants['placeholders']['noun'], vector) + encode(constants['placeholders']['property'], constants['placeholders']['null_property'])

def encode_scene(constants, subj, obj, verb):
	return np.asarray(encode(constants['placeholders']['subject'], subj) + encode(constants['placeholders']['verb'], verb) + encode(constants['placeholders']['object'], obj))

def transition(decision, options):
	if decision:
		threshold = ACCEPT_THRESHOLD
	else:
		threshold = 1 - ACCEPT_THRESHOLD
	if random.random() < threshold:
		return options[0]
	else:
		return options[1]

def construct_relation_vector(subject_noun, subject_property, verb, object_noun, object_property):
	relation = {}
	relation['subject_noun'] = subject_noun
	relation['subject_property'] = subject_property
	relation['verb'] = verb
	relation['object_noun'] = object_noun
	relation['object_property'] = object_property
	return relation

# ----- fight event schema -----

def enter_coffeeshop(seqs, variables, constants, relation_vectors):
	subj = assign_variable_properties(constants, variables['p1'], 'thirsty')
	obj = encode_null_property(constants, constants['nouns']['coffeeshop'])
	seqs.append(encode_scene(constants, subj, constants['verbs']['enter'], obj))
	relation_vectors.append(construct_relation_vector(variables['p1']['vector'], constants['properties']['thirsty'], constants['verbs']['enter'], constants['nouns']['coffeeshop'], constants['placeholders']['null_property']))
	return transition(variables['p1']['thirsty'], [walk_to_front, walk_to_back])

def walk_to_back(seqs, variables, constants, relation_vectors):
	subj = assign_variable_properties(constants, variables['p1'], 'impatient')
	obj = encode_null_property(constants, constants['nouns']['line'])
	seqs.append(encode_scene(constants, subj, constants['verbs']['obey'], obj))
	relation_vectors.append(construct_relation_vector(variables['p1']['vector'], constants['properties']['impatient'], constants['verbs']['obey'], constants['nouns']['line'], constants['placeholders']['null_property']))
	return transition(variables['p1']['impatient'], [walk_to_front, buy_coffee])

def walk_to_front(seqs, variables, constants, relation_vectors):
	subj = encode_null_property(constants, variables['p1']['vector'])
	obj = assign_variable_properties(constants, variables['p2'], 'violent')
	seqs.append(encode_scene(constants, subj, constants['verbs']['cut'], obj))
	relation_vectors.append(construct_relation_vector(variables['p1']['vector'], constants['placeholders']['null_property'], constants['verbs']['cut'], variables['p2']['vector'], constants['properties']['violent']))
	return transition(variables['p2']['violent'], [confront, buy_coffee])

def confront(seqs, variables, constants, relation_vectors):
	subj = encode_null_property(constants, variables['p2']['vector'])
	obj = assign_variable_properties(constants, variables['p1'], 'violent')
	seqs.append(encode_scene(constants, subj, constants['verbs']['confront'], obj))
	relation_vectors.append(construct_relation_vector(variables['p2']['vector'], constants['placeholders']['null_property'], constants['verbs']['confront'], variables['p1']['vector'], constants['properties']['violent']))
	return transition(variables['p1']['violent'], [hit, apologize])

def apologize(seqs, variables, constants, relation_vectors):
	subj = encode_null_property(constants, variables['p1']['vector'])
	obj = encode_null_property(constants, variables['p2']['vector'])
	seqs.append(encode_scene(constants, subj, constants['verbs']['apologize'], obj))
	relation_vectors.append(construct_relation_vector(variables['p1']['vector'], constants['placeholders']['null_property'], constants['verbs']['apologize'], variables['p2']['vector'], constants['placeholders']['null_property']))
	return transition(True, [buy_coffee])

def hit(seqs, variables, constants, relation_vectors):
	subj = encode_null_property(constants, variables['p1']['vector'])
	obj = encode_null_property(constants, variables['p2']['vector'])
	seqs.append(encode_scene(constants, subj, constants['verbs']['hit'], obj))
	relation_vectors.append(construct_relation_vector(variables['p1']['vector'], constants['placeholders']['null_property'], constants['verbs']['hit'], variables['p2']['vector'], constants['placeholders']['null_property']))
	return transition(True, [leave_coffeeshop])

def buy_coffee(seqs, variables, constants, relation_vectors):
	subj = encode_null_property(constants, variables['p1']['vector'])
	obj = encode_null_property(constants, constants['nouns']['coffee'])
	seqs.append(encode_scene(constants, subj, constants['verbs']['buy'], obj))
	relation_vectors.append(construct_relation_vector(variables['p1']['vector'], constants['placeholders']['null_property'], constants['verbs']['buy'], constants['nouns']['coffee'], constants['placeholders']['null_property']))
	return transition(True, [leave_coffeeshop])

def leave_coffeeshop(seqs, variables, constants, relation_vectors):
	subj = encode_null_property(constants, variables['p1']['vector'])
	obj = encode_null_property(constants, constants['nouns']['coffeeshop'])
	seqs.append(encode_scene(constants, subj, constants['verbs']['leave'], obj))
	relation_vectors.append(construct_relation_vector(variables['p1']['vector'], constants['placeholders']['null_property'], constants['verbs']['leave'], constants['nouns']['coffeeshop'], constants['placeholders']['null_property']))
	return None

# ----- mall event schema -----

def enter_mall(seqs, variables, constants, relation_vectors):
	subj = assign_variable_properties(constants, variables['p1'], 'thirsty')
	obj = encode_null_property(constants, constants['nouns']['mall'])
	seqs.append(encode_scene(constants, subj, constants['verbs']['enter'], obj))
	relation_vectors.append(construct_relation_vector(variables['p1']['vector'], constants['properties']['thirsty'], constants['verbs']['enter'], constants['nouns']['mall'], constants['placeholders']['null_property']))
	return transition(variables['p1']['thirsty'], [go_to_cafe, go_to_store])

def go_to_cafe(seqs, variables, constants, relation_vectors):
	subj = assign_variable_properties(constants, variables['p1'], 'rich')
	obj = encode_null_property(constants, constants['nouns']['coffee'])
	seqs.append(encode_scene(constants, subj, constants['verbs']['buy'], obj))
	relation_vectors.append(construct_relation_vector(variables['p1']['vector'], constants['properties']['rich'], constants['verbs']['buy'], constants['nouns']['coffee'], constants['placeholders']['null_property']))
	return transition(variables['p1']['rich'], [go_to_store, leave_mall])

def go_to_store(seqs, variables, constants, relation_vectors):
	subj = assign_variable_properties(constants, variables['p1'], 'impatient')
	obj = encode_null_property(constants, constants['nouns']['store'])
	seqs.append(encode_scene(constants, subj, constants['verbs']['enter'], obj))
	relation_vectors.append(construct_relation_vector(variables['p1']['vector'], constants['properties']['impatient'], constants['verbs']['enter'], constants['nouns']['store'], constants['placeholders']['null_property']))
	return transition(variables['p1']['impatient'], [try_shirt, greet])

def greet(seqs, variables, constants, relation_vectors):
	subj = encode_null_property(constants, variables['p1']['vector'])
	obj = assign_variable_properties(constants, variables['p2'], 'impatient')
	seqs.append(encode_scene(constants, subj, constants['verbs']['greet'], obj))
	relation_vectors.append(construct_relation_vector(variables['p1']['vector'], constants['placeholders']['null_property'], constants['verbs']['greet'], variables['p2']['vector'], constants['properties']['impatient']))
	return transition(variables['p2']['impatient'], [leave_mall, try_shirt])

def try_shirt(seqs, variables, constants, relation_vectors):
	subj = encode_null_property(constants, variables['p1']['vector'])
	obj = encode_null_property(constants, constants['nouns']['shirt'])
	seqs.append(encode_scene(constants, subj, constants['verbs']['try'], obj))
	relation_vectors.append(construct_relation_vector(variables['p1']['vector'], constants['placeholders']['null_property'], constants['verbs']['try'], constants['nouns']['shirt'], constants['placeholders']['null_property']))
	return transition(True, [buy_shirt])

def buy_shirt(seqs, variables, constants, relation_vectors):
	subj = encode_null_property(constants, variables['p1']['vector'])
	obj = encode_null_property(constants, constants['nouns']['shirt'])
	seqs.append(encode_scene(constants, subj, constants['verbs']['buy'], obj))
	relation_vectors.append(construct_relation_vector(variables['p1']['vector'], constants['placeholders']['null_property'], constants['verbs']['buy'], constants['nouns']['shirt'], constants['placeholders']['null_property']))
	return transition(True, [leave_mall])

def leave_mall(seqs, variables, constants, relation_vectors):
	subj = encode_null_property(constants, variables['p1']['vector'])
	obj = encode_null_property(constants, constants['nouns']['mall'])
	seqs.append(encode_scene(constants, subj, constants['verbs']['leave'], obj))
	relation_vectors.append(construct_relation_vector(variables['p1']['vector'], constants['placeholders']['null_property'], constants['verbs']['leave'], constants['nouns']['mall'], constants['placeholders']['null_property']))
	return None

# ----- event generator -----

def generate_fight(variables, constants):
	func = enter_coffeeshop
	seqs = []
	relation_vectors = []
	while (func != None):
		func = func(seqs, variables, constants, relation_vectors)
	return seqs, relation_vectors

def generate_mall(variables, constants):
	func = enter_mall
	seqs = []
	relation_vectors = []
	while (func != None):
		func = func(seqs, variables, constants, relation_vectors)
	return seqs, relation_vectors

def initialize_constants(constant_strings, dim):
	constants = {}
	for i in constant_strings:
		constants[i] = embed_2d(dim, None)
	return constants

def initialize_actors(num, dim):
	actors = []
	for i in range(num):
		a = {}
		a['thirsty'] = random.random() < 0.5
		a['impatient'] = random.random() < 0.5
		a['violent'] = random.random() < 0.5
		a['rich'] = random.random() < 0.5
		a['vector'] = embed_2d(dim, None)
		actors.append(a)
	return actors

def initialize_nouns(num, dim):
	nouns = []
	for i in range(num):
		n = {}
		n['value'] = embed_2d(dim, None)
		nouns.append(n)
	return actors

def generate_stories(train_num, test_num, dim):
	actors, constants = initialize_vectors(dim)

	# store sequences and generating events
	seqs_train = []
	events_train = []
	seqs_test = []
	events_test = []

	relations_list_train = []
	relations_list_test = []

	variables = {}
	# generate events
	generators = [generate_fight, generate_mall]
	for i in range(train_num):
		ind = np.random.permutation(len(actors))
		variables['p1'] = actors[ind[0]]
		variables['p2'] = actors[ind[1]]
		generate_story(variables, constants, generators, seqs_train, relations_list_train, events_train)
	for i in range(test_num):
		ind = np.random.permutation(len(actors))
		variables['p1'] = actors[ind[0]]
		variables['p2'] = actors[ind[1]]
		generate_story(variables, constants, generators, seqs_test, relations_list_test, events_test)
	return seqs_train, events_train, seqs_test, events_test, relations_list_train, relations_list_test

def generate_story(variables, constants, generators, seqs_list, relations_list, events_list):
	index = np.random.randint(len(generators))
	generator = generators[index]
	ret, relation_vectors = generator(variables, constants)
	events_list.extend([index] * len(ret))
	seqs_list.extend(ret)
	relations_list.extend(relation_vectors)

def initialize_vectors(dim):
	# initialize actors
	actors = initialize_actors(4, dim)

	# initialize constants
	placeholders = ['noun', 'property', 'null_property', 'subject', 'verb', 'object']
	properties = ['thirsty', 'violent', 'impatient', 'rich']
	nouns = ['coffeeshop', 'coffee', 'line', 'store', 'shirt', 'mall']
	verbs = ['enter', 'obey', 'cut', 'buy', 'confront', 'hit', 'apologize', 'leave', 'greet', 'try']
	constants = {}
	constants['placeholders'] = initialize_constants(placeholders, dim)
	constants['properties'] = initialize_constants(properties, dim)
	constants['nouns'] = initialize_constants(nouns, dim)
	constants['verbs'] = initialize_constants(verbs, dim)

	return actors, constants

# generate scenes with a single generator, separated for each ground truth event - currently used to test decoding
def generate_evaluation_events(actors, constants, num_events):
	training = []
	semantic_vectors = []
	variables = {}
	generators = [generate_fight]
	for i in range(num_events):
	    ind = np.random.permutation(len(actors))
	    variables['p1'] = actors[ind[0]]
	    variables['p2'] = actors[ind[1]]
	    event = []
	    semantic = []
	    generate_story(variables, constants, generators, event, semantic, [])
	    # normalize
	    training.append(list(map(lambda x: normalize(x), event)))
	    semantic_vectors.append(semantic)
	return training, semantic_vectors
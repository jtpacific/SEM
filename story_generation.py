from hrr import encode, decode, embed_2d
import numpy as np
import random

# ----- fight event schema -----

def enter_coffeeshop(seqs, v, c):
	if v['p1']['thirsty']:
		actor = encode(v['p1']['vector'], c['thirsty'])
	else:
		actor = v['p1']['vector']
	scene = encode(actor, c['actor']) + encode(c['enter'], c['verb']) + encode(c['coffeeshop'], c['subject'])
	seqs.append(np.asarray(scene))
	if v['p1']['thirsty']:
		threshold = 0.9
	else:
		threshold = 0.1
	if random.random() < threshold:
		return walk_to_front
	else:
		return walk_to_back

def walk_to_back(seqs, v, c):
	if v['p1']['impatient']:
		actor = encode(v['p1']['vector'], c['impatient'])
	else:
		actor = v['p1']['vector']
	scene = encode(actor, c['actor']) + encode(c['obey'], c['verb']) + encode(c['line'], c['subject'])
	seqs.append(np.asarray(scene))
	if v['p1']['impatient']:
		threshold = 0.9
	else:
		threshold = 0.1
	if random.random() < threshold:
		return walk_to_front
	else:
		return buy_coffee

def walk_to_front(seqs, v, c):
	if v['p2']['violent']:
		subject = encode(v['p2']['vector'], c['violent'])
	else:
		subject = v['p2']['vector']
	scene = encode(v['p1']['vector'], c['actor']) + encode(c['cut'], c['verb']) + encode(subject, c['subject'])
	seqs.append(np.asarray(scene))
	if v['p2']['violent']:
		threshold = 0.9
	else:
		threshold = 0.1
	if random.random() < threshold:
		return confront
	else:
		return buy_coffee

def confront(seqs, v, c):
	if v['p1']['violent']:
		subject = encode(v['p1']['vector'], c['violent'])
	else:
		subject = v['p1']['vector']
	scene = encode(v['p2']['vector'], c['actor']) + encode(c['confront'], c['verb']) + encode(subject, c['subject'])
	seqs.append(np.asarray(scene))
	if v['p1']['violent']:
		threshold = 0.9
	else:
		threshold = 0.1
	if random.random() < threshold:
		return hit
	else:
		return apologize

def apologize(seqs, v, c):
	scene = encode(v['p1']['vector'], c['actor']) + encode(c['apologize'], c['verb']) + encode(v['p2']['vector'], c['subject'])
	seqs.append(np.asarray(scene))
	return buy_coffee

def hit(seqs, v, c):
	scene = encode(v['p1']['vector'], c['actor']) + encode(c['hit'], c['verb']) + encode(v['p2']['vector'], c['subject'])
	seqs.append(np.asarray(scene))
	return leave_coffeeshop

def buy_coffee(seqs, v, c):
	scene = encode(v['p1']['vector'], c['actor']) + encode(c['buy'], c['verb']) + encode(c['coffee'], c['subject'])
	seqs.append(np.asarray(scene))
	return leave_coffeeshop

def leave_coffeeshop(seqs, v, c):
	scene = encode(v['p1']['vector'], c['actor']) + encode(c['leave'], c['verb']) + encode(c['coffeeshop'], c['subject'])
	seqs.append(np.asarray(scene))
	return None

# ----- mall event schema -----

def enter_mall(seqs, v, c):
	if v['p1']['thirsty']:
		actor = encode(v['p1']['vector'], c['thirsty'])
	else:
		actor = v['p1']['vector']
	scene = encode(actor, c['actor']) + encode(c['enter'], c['verb']) + encode(c['mall'], c['subject'])
	seqs.append(np.asarray(scene))
	if v['p1']['thirsty']:
		threshold = 0.9
	else:
		threshold = 0.1
	if random.random() < threshold:
		return go_to_cafe
	else:
		return go_to_store

def go_to_cafe(seqs, v, c):
	if v['p1']['rich']:
		actor = encode(v['p1']['vector'], c['rich'])
	else:
		actor = v['p1']['vector']
	scene = encode(actor, c['actor']) + encode(c['buy'], c['verb']) + encode(c['coffee'], c['subject'])
	seqs.append(np.asarray(scene))
	if v['p1']['rich']:
		threshold = 0.9
	else:
		threshold = 0.1
	if random.random() < threshold:
		return go_to_store
	else:
		return leave_mall

def go_to_store(seqs, v, c):
	if v['p1']['impatient']:
		actor = encode(v['p1']['vector'], c['impatient'])
	else:
		actor = v['p1']['vector']
	scene = encode(actor, c['actor']) + encode(c['enter'], c['verb']) + encode(c['store'], c['subject'])
	seqs.append(np.asarray(scene))
	if v['p1']['impatient']:
		threshold = 0.9
	else:
		threshold = 0.1
	if random.random() < threshold:
		return try_shirt
	else:
		return greet

def greet(seqs, v, c):
	if v['p2']['impatient']:
		subject = encode(v['p2']['vector'], c['impatient'])
	else:
		subject = v['p2']['vector']
	scene = encode(v['p1']['vector'], c['actor']) + encode(c['greet'], c['verb']) + encode(subject, c['subject'])
	seqs.append(np.asarray(scene))
	if v['p2']['impatient']:
		threshold = 0.9
	else:
		threshold = 0.1
	if random.random() < threshold:
		return leave_mall
	else:
		return try_shirt

def try_shirt(seqs, v, c):
	scene = encode(v['p1']['vector'], c['actor']) + encode(c['try on'], c['verb']) + encode(c['shirt'], c['subject'])
	seqs.append(np.asarray(scene))
	return buy_shirt

def buy_shirt(seqs, v, c):
	scene = encode(v['p1']['vector'], c['actor']) + encode(c['buy'], c['verb']) + encode(c['shirt'], c['subject'])
	seqs.append(np.asarray(scene))
	return leave_mall

def leave_mall(seqs, v, c):
	scene = encode(v['p1']['vector'], c['actor']) + encode(c['leave'], c['verb']) + encode(c['mall'], c['subject'])
	seqs.append(np.asarray(scene))
	return None

# ----- event generator -----

def generate_fight(variables, constants):
	func = enter_coffeeshop
	seqs = []
	while (func != None):
		func = func(seqs, variables, constants)
	return seqs

def generate_mall(variables, constants):
	func = enter_mall
	seqs = []
	while (func != None):
		func = func(seqs, variables, constants)
	return seqs

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

def generate_stories(train_num, test_num, dim):
	# initialize actors
	actors = initialize_actors(4, dim)

	# initialize constants
	constant_strings = ['actor', 'verb', 'subject', 'thirsty', 'violent', 'impatient', 'coffeeshop', 'coffee', 'line', 'enter', 'obey', 'cut', 'buy', 'confront', 'hit', 'apologize', 'buy', 'leave', 'store', 'shirt', 'greet', 'rich', 'mall', 'try on']
	constants = initialize_constants(constant_strings, dim)

	# store sequences and generating events
	seqs_train = []
	events_train = []
	seqs_test = []
	events_test = []

	variables = {}
	# generate events
	generators = [generate_fight, generate_mall]
	for i in range(train_num):
		ind = np.random.permutation(len(actors))
		variables['p1'] = actors[ind[0]]
		variables['p2'] = actors[ind[1]]
		generate_story(variables, constants, generators, seqs_train, events_train)
	for i in range(test_num):
		ind = np.random.permutation(len(actors))
		variables['p1'] = actors[ind[0]]
		variables['p2'] = actors[ind[1]]
		generate_story(variables, constants, generators, seqs_test, events_test)
	return seqs_train, events_train, seqs_test, events_test

def generate_story(variables, constants, generators, seqs_list, events_list):
	index = np.random.randint(len(generators))
	generator = generators[index]
	ret = generator(variables, constants)
	events_list.extend([index] * len(ret))
	seqs_list.extend(ret)
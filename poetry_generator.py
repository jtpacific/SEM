from encoding_schemes import encode_scene
import random
import numpy as np

def transition(probs, next_scene):
	return np.random.choice(next_scene, p = probs)

'''
each function in the generator takes in
	event: sequence of scenes that each function appends a scene to
	generating_fillers: corresponding sequence of fillers used to generate each scene, used for decoded comparison
	chosen_properties: properties of the filler to include (used for selective property inclusion classes)
	variables: dictionary mapping actor roles to actors
	context: structure representing the vectorized symbol space
	encoding: string representing the desired encoding scheme that the appended scene will be encoded under
the function appends an encoded scene to the event and appends a generating filler structure to the generating_fillers
the function transitions to the next function based on the properties of the actors involved

'''

def enter_coffeeshop_poetry(event, generating_fillers, chosen_properties, variables, context, encoding):
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p1'], context.verbs['enter'], context.nouns['coffeeshop'], agent_property = ['hungry'])
	event.append(scene)
	generating['id'] = 1
	generating_fillers.append(generating)
	if variables['p1']['properties']['hungry']:
		probabilities = [0.9, 0.1]
	else: 
		probabilities = [0.4, 0.6]
	return transition(probabilities, [order_drink, sit_down])

def order_drink(event, generating_fillers, chosen_properties, variables, context, encoding):
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p1'], context.verbs['order'], context.nouns[variables['p1']['properties']['drink']])
	event.append(scene)
	generating['id'] = 2
	generating_fillers.append(generating)
	if variables['p1']['properties']['drink'] == 'latte':
		probabilities = [0.8, 0.2]
	else: 
		probabilities = [0.0, 1.0]
	return transition(probabilities, [too_expensive, sit_down])

def too_expensive(event, generating_fillers, chosen_properties, variables, context, encoding):
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p1'], context.verbs['cancel'], context.nouns[variables['p1']['properties']['drink']])
	event.append(scene)
	generating['id'] = 3
	generating_fillers.append(generating)
	return transition([1.0], [sit_down])

def sit_down(event, generating_fillers, chosen_properties, variables, context, encoding):
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p1'], context.verbs['find'], variables['p2'])
	event.append(scene)
	generating['id'] = 4
	generating_fillers.append(generating)
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p2'], context.verbs['greet'], variables['p1'])
	event.append(scene)
	generating['id'] = 5
	generating_fillers.append(generating)
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p1'], context.verbs['greet'], variables['p2'])
	event.append(scene)
	generating['id'] = 6
	generating_fillers.append(generating)
	return transition([0.5, 0.5], [emcee_intro, poet_performs])

def emcee_intro(event, generating_fillers, chosen_properties, variables, context, encoding):
	scene, generating = encode_scene(context, encoding, chosen_properties, context.nouns['emcee'], context.verbs['introduce'], variables['p3'])
	event.append(scene)
	generating['id'] = 7
	generating_fillers.append(generating)
	return transition([1.0], [poet_performs])

def poet_performs(event, generating_fillers, chosen_properties, variables, context, encoding):
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p3'], context.verbs['perform'], context.nouns['poem'])
	event.append(scene)
	generating['id'] = 8
	generating_fillers.append(generating)
	if variables['p1']['properties']['nervous']:
		probabilities = [0.8, 0.2]
	elif variables['p1']['properties']['happy']: 
		probabilities = [0.1, 0.9]
	else:
		probabilities = [0.3, 0.7]
	return transition(probabilities, [agent_declines, agent_performs])

def agent_declines(event, generating_fillers, chosen_properties, variables, context, encoding):
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p1'], context.verbs['decline'], context.nouns['poem'])
	event.append(scene)
	generating['id'] = 9
	generating_fillers.append(generating)
	return transition([1.0], [say_goodbye])

def agent_performs(event, generating_fillers, chosen_properties, variables, context, encoding):
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p1'], context.verbs['perform'], context.nouns['poem'])
	event.append(scene)
	generating['id'] = 10
	generating_fillers.append(generating)
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p2'], context.verbs['congratulate'], variables['p1'])
	event.append(scene)
	generating['id'] = 11
	generating_fillers.append(generating)
	return transition([1.0], [say_goodbye])

def say_goodbye(event, generating_fillers, chosen_properties, variables, context, encoding):
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p1'], context.verbs['say_goodbye'], variables['p2'], agent_property = ['hungry'])
	event.append(scene)
	generating['id'] = 12
	generating_fillers.append(generating)
	if variables['p1']['properties']['hungry']:
		probabilities = [0.7, 0.3]
	else:
		probabilities = [0.1, 0.9]	
	return transition(probabilities, [order_dessert, None])

def order_dessert(event, generating_fillers, chosen_properties, variables, context, encoding):
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p1'], context.verbs['order'], context.nouns[variables['p1']['properties']['dessert']])
	event.append(scene)
	generating['id'] = 13
	generating_fillers.append(generating)
	return None

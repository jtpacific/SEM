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

def enter_coffeeshop_fight(event, generating_fillers, chosen_properties, variables, context, encoding):
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p1'], context.verbs['enter'], context.nouns['coffeeshop'], agent_property = ['thirsty'])
	event.append(scene)
	generating['id'] = 1
	generating_fillers.append(generating)
	if variables['p1']['properties']['thirsty']:
		probabilities = [0.8, 0.2]
	else: 
		probabilities = [0.4, 0.6]
	return transition(probabilities, [walk_to_front, walk_to_back])

def walk_to_front(event, generating_fillers, chosen_properties, variables, context, encoding):
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p1'], context.verbs['cut'], variables['p2'], patient_property = ['violent'])
	event.append(scene)
	generating['id'] = 2
	generating_fillers.append(generating)
	if variables['p2']['properties']['violent']:
		probabilities = [0.8, 0.2]
	else: 
		probabilities = [0.5, 0.5]
	return transition(probabilities, [step_in_front, say_excuse_me])

def walk_to_back(event, generating_fillers, chosen_properties, variables, context, encoding):
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p1'], context.verbs['obey'], context.nouns['line'])
	event.append(scene)
	generating['id'] = 3
	generating_fillers.append(generating)
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p1'], context.verbs['cut'], variables['p2'], patient_property = ['violent'])
	event.append(scene)
	generating['id'] = 4
	generating_fillers.append(generating)
	if variables['p2']['properties']['violent']:
		probabilities = [0.7, 0.3]
	else: 
		probabilities = [0.4, 0.6]
	return transition(probabilities, [step_in_front, say_excuse_me])

def step_in_front(event, generating_fillers, chosen_properties, variables, context, encoding):
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p2'], context.verbs['cut'], variables['p1'], patient_property = ['violent', 'drink'])
	event.append(scene)
	generating['id'] = 5
	generating_fillers.append(generating)
	if variables['p1']['properties']['violent']:
		probabilities = [0.6, 0.2, 0.2]
	else: 
		probabilities = [0.1, 0.4, 0.5]
	return transition(probabilities, [cutter_shove, cutter_ignore, cutter_glare])

def say_excuse_me(event, generating_fillers, chosen_properties, variables, context, encoding):
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p2'], context.verbs['confront'], variables['p1'], patient_property = ['violent', 'drink'])
	event.append(scene)
	generating['id'] = 6
	generating_fillers.append(generating)
	if variables['p1']['properties']['violent']:
		probabilities = [0.6, 0.2, 0.2]
	else: 
		probabilities = [0.1, 0.4, 0.5]
	return transition(probabilities, [cutter_shove, cutter_ignore, cutter_glare])

def cutter_shove(event, generating_fillers, chosen_properties, variables, context, encoding):
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p1'], context.verbs['shove'], variables['p2'], patient_property = ['violent'])
	event.append(scene)
	generating['id'] = 7
	generating_fillers.append(generating)
	if variables['p2']['properties']['violent']:
		probabilities = [0.6, 0.2, 0.2]
	else: 
		probabilities = [0.3, 0.4, 0.3]
	return transition(probabilities, [cutted_shove, turn_to_barista, cutted_glare])

def cutter_ignore(event, generating_fillers, chosen_properties, variables, context, encoding):
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p1'], context.verbs['ignore'], variables['p2'], patient_property = ['violent'])
	event.append(scene)
	generating['id'] = 8
	generating_fillers.append(generating)
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p1'], context.verbs['order'], context.nouns[variables['p1']['properties']['drink']])
	event.append(scene)
	generating['id'] = 9
	generating_fillers.append(generating)
	if variables['p2']['properties']['violent']:
		probabilities = [0.6, 0.2, 0.2]
	else: 
		probabilities = [0.3, 0.4, 0.3]
	return transition(probabilities, [cutted_shove, turn_to_barista, cutted_glare])

def cutter_glare(event, generating_fillers, chosen_properties, variables, context, encoding):
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p1'], context.verbs['glare'], variables['p2'], patient_property = ['violent'])
	event.append(scene)
	generating['id'] = 10
	generating_fillers.append(generating)
	if variables['p2']['properties']['violent']:
		probabilities = [0.8, 0.1, 0.1]
	else: 
		probabilities = [0.2, 0.7, 0.1]
	return transition(probabilities, [cutted_shove, turn_to_barista, cutted_glare])

def turn_to_barista(event, generating_fillers, chosen_properties, variables, context, encoding):
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p2'], context.verbs['complain'], context.nouns['barista'])
	event.append(scene)
	generating['id'] = 11
	generating_fillers.append(generating)
	probabilities = [0.6, 0.4]
	return transition(probabilities, [cream_splash, dessert_crumble])

def cutted_glare(event, generating_fillers, chosen_properties, variables, context, encoding):
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p2'], context.verbs['glare'], variables['p1'])
	event.append(scene)
	generating['id'] = 12
	generating_fillers.append(generating)
	probabilities = [0.4, 0.6]
	return transition(probabilities, [cream_splash, dessert_crumble])

def cutted_shove(event, generating_fillers, chosen_properties, variables, context, encoding):
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p2'], context.verbs['shove'], variables['p1'])
	event.append(scene)
	generating['id'] = 13
	generating_fillers.append(generating)
	probabilities = [0.3, 0.7]
	return transition(probabilities, [cream_splash, dessert_crumble])

def cream_splash(event, generating_fillers, chosen_properties, variables, context, encoding):
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p1'], context.verbs['splash'], variables['p2'], agent_property = ['violent'])
	event.append(scene)
	generating['id'] = 14
	generating_fillers.append(generating)
	if variables['p1']['properties']['violent']:
		probabilities = [0.8, 0.2]
	else: 
		probabilities = [0.5, 0.5]
	return transition(probabilities, [call_policeman, barista_orders])

def dessert_crumble(event, generating_fillers, chosen_properties, variables, context, encoding):
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p1'], context.verbs['crumble'], variables['p2'], agent_property = ['violent'])
	event.append(scene)
	generating['id'] = 15
	generating_fillers.append(generating)
	if variables['p1']['properties']['violent']:
		probabilities = [0.8, 0.2]
	else: 
		probabilities = [0.3, 0.7]
	return transition(probabilities, [call_policeman, barista_orders])

def call_policeman(event, generating_fillers, chosen_properties, variables, context, encoding):
	scene, generating = encode_scene(context, encoding, chosen_properties, context.nouns['barista'], context.verbs['call'], context.nouns['policeman'])
	event.append(scene)
	generating['id'] = 16
	generating_fillers.append(generating)
	scene, generating = encode_scene(context, encoding, chosen_properties, context.nouns['policeman'], context.verbs['expel'], variables['p1'])
	event.append(scene)
	generating['id'] = 17
	generating_fillers.append(generating)
	probabilities = [0.3, 0.7]
	return transition(probabilities, [love_juice, hate_coffee])

def barista_orders(event, generating_fillers, chosen_properties, variables, context, encoding):
	scene, generating = encode_scene(context, encoding, chosen_properties, context.nouns['barista'], context.verbs['expel'], variables['p1'])
	event.append(scene)
	generating['id'] = 18
	generating_fillers.append(generating)
	probabilities = [0.7, 0.3]
	return transition(probabilities, [love_juice, hate_coffee])

def barista_apologize(event, generating_fillers, chosen_properties, variables, context, encoding):
	scene, generating = encode_scene(context, encoding, chosen_properties, context.nouns['barista'], context.verbs['apologize'], variables['p2'])
	event.append(scene)
	generating['id'] = 19
	generating_fillers.append(generating)
	return None

def love_juice(event, generating_fillers, chosen_properties, variables, context, encoding):
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p1'], context.verbs['yell'], context.nouns['loves_juice'])
	event.append(scene)
	generating['id'] = 20
	generating_fillers.append(generating)
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p1'], context.verbs['leave'], context.nouns['coffeeshop'])
	event.append(scene)
	generating['id'] = 21
	generating_fillers.append(generating)
	return None

def hate_coffee(event, generating_fillers, chosen_properties, variables, context, encoding):
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p1'], context.verbs['yell'], context.nouns['hates_coffee'])
	event.append(scene)
	generating['id'] = 22
	generating_fillers.append(generating)
	scene, generating = encode_scene(context, encoding, chosen_properties, variables['p1'], context.verbs['leave'], context.nouns['coffeeshop'])
	event.append(scene)
	generating['id'] = 23
	generating_fillers.append(generating)
	return None
import event_generation
import numpy as np
import hrr
import encoding_schemes
import matplotlib.pyplot as plt

# train single event schema on list of events
def train_online(context, encoding, segment_size, num_segments, eval_size, transition_model, generator):
    errors = {}
    decoded_errors = []
    encoded_errors = []
    for segment in range(num_segments):
        # generate evaluation set and calculate evaluation error 
        testing, testing_fillers = event_generation.generate_evaluation_events(context, encoding, generator, num_events = eval_size, testing = True)
        testing_errors = evaluate_predictions(encoding, testing, testing_fillers, context, transition_model, testing = True)
        encoded_errors.append(testing_errors[0])
        decoded_errors.append(testing_errors[1])
        # generate training set and train online for each event
        training, training_fillers = event_generation.generate_evaluation_events(context, encoding, generator, num_events = segment_size, testing = False)
        for e in range(len(training)):
            X = np.asarray([training[e][0:len(training[e]) - 1]])
            Y = np.asarray([training[e][1:len(training[e])]])
            transition_model.model.fit(X, Y, verbose=0)
    errors['decoded_errors'] = np.asarray(decoded_errors)
    errors['encoded_errors'] = np.asarray(encoded_errors)
    return errors

# takes in set of events and returns encoded and decoded errors against generating fillers
def evaluate_predictions(encoding, events, fillers, context, transition_model, testing = False):
    encoded_errors = []
    decoded_errors = []
    # traverse all evaluation events
    for e in range(len(events)):
        seen = [events[e][0]]
        encoded_error = 0.0
        decoded_error = np.zeros(6)
        # traverse all scene excluding the first
        for s in range(1, len(events[e])):
            prediction = hrr.normalize(transition_model.predict(seen))
            encoded_error += np.linalg.norm(events[e][s] - prediction)/(len(events[e]) - 1)
            decoded_error += np.asarray(encoding_schemes.filler_errors(encoding, prediction, fillers[e][s], context, testing))/(len(events[e]) - 1)
            seen.append(events[e][s])
        encoded_errors.append(encoded_error)
        decoded_errors.append(decoded_error)
    return [encoded_errors, decoded_errors]

# plot encoded and decoded errors
def plot_encoded_errors(encoding, errors):
    num = len(errors['encoded_errors'])
    plt.subplot(2, 1, 1)
    
    y = list(map(lambda x: np.mean(x), errors['encoded_errors']))
    y_err = list(map(lambda x: np.std(x)/np.sqrt(len(x)), errors['encoded_errors']))
    plt.errorbar(range(num), y, yerr=y_err, fmt='', capsize=1)
   
    plt.title(encoding + " encoded error")
    plt.xlabel("training segment")
    plt.ylabel("average event rmse")
    plt.subplot(2, 1, 2)
    decoded_errors_agent = errors['decoded_errors'][:,3]
   
    y_agent = list(map(lambda x: np.mean(x[:,3]), errors['decoded_errors']))
    y_agent_err = list(map(lambda x: np.std(x[:,3])/np.sqrt(len(x)), errors['decoded_errors']))    
    plt.errorbar(range(num), y_agent, yerr=y_agent_err, fmt='', capsize=1, label="agent")

    y_action = list(map(lambda x: np.mean(x[:,4]), errors['decoded_errors']))
    y_action_err = list(map(lambda x: np.std(x[:,4])/np.sqrt(len(x)), errors['decoded_errors']))    
    plt.errorbar(range(num), y_action, yerr=y_action_err, fmt='', capsize=1, label="action")

    y_patient = list(map(lambda x: np.mean(x[:,5]), errors['decoded_errors']))
    y_patient_err = list(map(lambda x: np.std(x[:,5])/np.sqrt(len(x)), errors['decoded_errors']))    
    plt.errorbar(range(num), y_patient, yerr=y_patient_err, fmt='', capsize=1, label="patient")

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(encoding + " decoded error")
    plt.xlabel("training segment")
    plt.ylabel("average event filler error")

def plot_decoded_errors(encoding, errors):
    num = len(errors['encoded_errors'])
    
    y_agent = list(map(lambda x: np.mean(x[:,0]), errors['decoded_errors']))
    y_agent_err = list(map(lambda x: np.std(x[:,0])/np.sqrt(len(x)), errors['decoded_errors']))    
    plt.errorbar(range(num), y_agent, yerr=y_agent_err, fmt='', capsize=1, label="agent")

    y_action = list(map(lambda x: np.mean(x[:,1]), errors['decoded_errors']))
    y_action_err = list(map(lambda x: np.std(x[:,1])/np.sqrt(len(x)), errors['decoded_errors']))    
    plt.errorbar(range(num), y_action, yerr=y_action_err, fmt='', capsize=1, label="action")

    y_patient = list(map(lambda x: np.mean(x[:,2]), errors['decoded_errors']))
    y_patient_err = list(map(lambda x: np.std(x[:,2])/np.sqrt(len(x)), errors['decoded_errors']))    
    plt.errorbar(range(num), y_patient, yerr=y_patient_err, fmt='', capsize=1, label="patient")

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(encoding + " decoded error")
    plt.xlabel("training segment")
    plt.ylabel("average event filler error")
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.show()
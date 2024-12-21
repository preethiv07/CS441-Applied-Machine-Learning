%matplotlib inline
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
from hmmlearn.hmm import MultinomialHMM
from matplotlib import pyplot as plt
import seaborn as sns

from aml_utils import test_case_checker, perform_computation


# Data

# 0.2 Loading the Data
# Let's extract the data

data = pd.read_csv('../HMMInference-lib/data.csv')
V = data['Markings'].values

# 0.3 Creating a default set of transition and emision probability matrices and training the HMM

# Transition Probability Matrix
transition_p = np.ones((2, 2))
transition_p = transition_p / np.sum(transition_p, axis=1)
 
# Emission Probability Matrix
emission_p = np.array(((1, 3, 5), (2, 4, 6)))
emission_p = emission_p / np.sum(emission_p, axis=1).reshape((-1, 1))
 
initial_distribution = np.array((0.5, 0.5))
 

print(transition_p)
print(emission_p)
print(initial_distribution)

np.random.seed(37)
model = MultinomialHMM(n_components=2,  n_iter=10000).fit(np.reshape(V,[len(V),1]))


print(model.n_features)
print(model.emissionprob_)
print(model.transmat_)

# Task 1

#
'''
Create a python function to compute the cost of the best path leaving each node at a given iteration of the dynamic programming process (for a given column of the trellis), and also the path. For the sake of simplicity you are given the dynamic programming procedure. You only need to program the computation of the cost function and the identification of the best cost and the index of the best cost. Thus, for the function currentWeightBestPath, you should report (a) the cost (sum of log probabilities) and (b) the index (argument) that optimizes the cost.
For the sake of simplicity you can maximize the sum of log probabilities using the formula:
𝐶𝑤(𝑗)=max𝑢[log𝑃(𝑋𝑤+1=𝑢|𝑋𝑤=𝑗)+log𝑃(𝑌𝑤+1|𝑋𝑤+1=𝑢)+𝐶𝑤+1(𝑢)]  𝐵𝑤(𝑗)=argmax𝑢[log𝑃(𝑋𝑤+1=𝑢|𝑋𝑤=𝑗)+log𝑃(𝑌𝑤+1|𝑋𝑤+1=𝑢)+𝐶𝑤+1(𝑢)] 
Suggestion: build first the sum of log probabilities and use the resulting vector to identify the maximal value of the log probability and the index associated to it:
log𝑃(𝑋𝑤+1|𝑋𝑤)+log𝑃(𝑌𝑤+1|𝑋𝑤+1)+𝐶𝑤+1 
The dimension of  log𝑃(𝑋𝑤+1|𝑋𝑤)  and  𝐶𝑤+1  is equal to the number of states (in this assignment the numver is 2). log𝑃(𝑌𝑤+1|𝑋𝑤+1=𝑢)  is a single value.
'''

def currentWeightBestPath(logP_X_given_X, logP_Y_given_X, Cw1):
    """
    Performs computation of log-probabilities to derive Cw(j) and Bw(j).
    for a specific iteration j

        Parameters:
                logP_X_given_X (np array): Dimension = S (number of states) x 1
                logP_Y_given_X (np array): dimension 1x1.  
                Cw1 (np array): Dimension= number of states x 1. The value of the previous Cw

        Returns:
                u (int): the index with maximum sum of log probabilities among S options
                maxlogprob (double): the maximum sum of log probabilities among S options
    """       
    assert logP_X_given_X.ndim == 1
    assert Cw1.ndim == 1

    assert logP_X_given_X.shape[0]==Cw1.shape[0] #== S

    # your code here
    # raise NotImplementedError
    
    return u,maxplogrob

def viterbi_Modular(V, a, b, initial_distribution):
    """ 
    This function simply applies your cost function to traverse the trellis for identification of the optimal path
    This implementation uses a forward exploration and then backtracks the computed values to find the optimal path.
    """
    T = V.shape[0]
    M = a.shape[0] # or [1]
 
    omega = np.zeros((T, M))
    omega[0, :] = np.log(initial_distribution * b[:, V[0]])
 
    prev = np.zeros((T - 1, M))
 
    for t in range(1, T):
        for j in range(M):
            #transition and emision (transition is associated to X|X since the book uses X to represent the hidden states and emision to Y|X)
            u,maxprob = currentWeightBestPath( np.log(a[:, j]),  np.log(b[j, V[t]]) , omega[t - 1]) 
 
            # The most probable state given previous state for time t:
            prev[t - 1, j] = u #np.argmax(probability)
 
            # The probability of the most probable state at time t:
            omega[t, j] = maxprob #np.max(probability)

 
    # Path initialization
    S = np.zeros(T,dtype=int)
 
    # Find the most probable last hidden state
    last_state = np.argmax(omega[T - 1, :])
 
    S[0] = last_state
 
    backtrack_index = 1
    for i in range(T - 2, -1, -1):
        S[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1
 
    # backtrack
    result = np.flip(S, axis=0)
 
    return result


# Let's evaluate the Viterbi algorithm with your cost function implementation using X1
np.random.seed(1)
X1 = np.random.randint(low=0, high=3, size=(15,))

model2=MultinomialHMM(n_components=2,  n_iter=10000).fit(np.reshape(X1,[len(X1),1]))

predicted_hidden_sates = viterbi_Modular(X1, model2.transmat_, model2.emissionprob_, initial_distribution)

assert np.array_equal(predicted_hidden_sates,  np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1]))

print(predicted_hidden_sates)



predicted_sequence = viterbi_Modular(V, model.transmat_, model.emissionprob_, initial_distribution)

print(predicted_sequence)


# Task 2
'''
Implement the HMM sequence using the predict funtion of hmmlearn. For this you need to make sure the sequence (which is stored in the variable V) is formatted as a column vector (one column).
The function predict is a property of the model's object. Your implementation uses as parameters the model trained in the previous cell and the array of observed values.
'''

def predictSequenceReference(observed_values, model):
    """
    Performs prediction of most likely sequence using the hmmlearn library.

        Parameters:
                observed_values (np array): Observed values
                model (hmmlearn.hmm.MultinomialHMM object): the trained HMM  

        Returns:
                predicted_sequence (np array): the predicted sequence of hidden states
    """       

    # your code here
    # Ensure observed_values is reshaped into a column vector
    observed_values = observed_values.reshape(-1, 1)
    
    # Use the predict method to compute the most likely sequence
    predicted_sequence = model.predict(observed_values)
    
#     raise NotImplementedError
    
    return predicted_sequence


# Task 3
'''
Use the ground truth sequence to format the noisy ground truth, i.e., to replace a value of 0 instead of "centered" and a value of 1 instead of "out"
'''

def formatNoisySequence(noisy_sequence):
    """
    uses a numeric formatting of the original noisy sequence.

        Parameters:
                noisy_sequence (np array): Sequence with elements in {'centered', 'out'}
        Returns:
                formatted_noisy_sequence (np array): the numeric version of the original sequence
    """       

    # your code here
    # Define the mapping: "centered" -> 0, "out" -> 1
    mapping = {"centered": 0, "out": 1}
    
    # Vectorized mapping using NumPy
    formatted_noisy_sequence = np.vectorize(mapping.get)(noisy_sequence)
#     raise NotImplementedError
    
    return formatted_noisy_sequence


# Task 4
# Compute the MSE and MAE of the predicted label. While it is not always straighforward to see, an objective of using HMM for data denoising is also minimizing the difference from the noisy states with the objective of maintaining a close resemblance to the expected sequence.
# To evaluate your implementation of HMM with respect to a previously assign "ground truth" you will implement the mean squared error and mean absolute error of the predicted sequence vs. the "real" but noisy sequence.

def computeErrors(predictions, realvalues):
    """
    Performs computation of MSE and MAE of the predicted sequence vs. the real sequence

        Parameters:
                predictions (np array): Dimension = N x 1
                realvalues (np array): Dimension = Nx1  

        Returns:
                MSE (double): the Mean Squared Error
                MAE (double): the Mean Absolute Error
    """       
    assert predictions.shape[0]==realvalues.shape[0] #== S

    # your code here
    # Calculate the differences
    differences = predictions - realvalues
    
    # Compute MSE
    mse = np.mean(np.square(differences))
    
    # Compute MAE
    mae = np.mean(np.abs(differences))
#     raise NotImplementedError
    
    return mse,mae


# Errors for your solution: mse: 0.196000, mae: 0.196000
# Errors for reference solution: mse: 0.200000, mae: 0.200000
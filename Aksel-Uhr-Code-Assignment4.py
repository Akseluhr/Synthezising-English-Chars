#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 11:10:34 2022

@author: akseluhr
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

#########################################

# Read in data

#########################################

def read_data():
    file = open('/Users/akseluhr/Documents/GitHub/RNN-Synthezising-English-Chars/goblet_book.txt', 'r')
    book_data = file.read()
    return book_data, set(book_data)

# Creating maps (dicts) for quick check ups in one-hot
def GetKeyValueStore(data):
    char_to_index = {}
    index_to_char = {}
    key_val = 0
    for c in set_book_chars:
        char_to_index[c] = key_val
        index_to_char[key_val] = c
        key_val += 1
    return char_to_index, index_to_char

# Not used currently
def OneHotNo(X_chars, Y_chars, c_t_i, rnn):
   # print(rnn.seq_length)
    #print(len(Y_chars), 'ychar')
    X = np.zeros((rnn.k, len(X_chars)))
    Y = np.zeros((rnn.k, len(Y_chars)))
    vector = '.'
    C_T_I = np.zeros((len(c_t_i), len(vector)))

   # print("X input shape: ", X.shape)
   # print("Y output shape: ", Y.shape)
    for i in range(len(X_chars)):
        X[c_t_i[X_chars[i]], i] = 1
       # print(X.shape)
        
    for i in range(len(Y_chars)):
        Y[c_t_i[Y_chars[i]], i] = 1
       # print(Y.shape)
        
    for i in range(len(vector)):
        C_T_I[c_t_i[vector[i]], i] = 1
    return X, Y, C_T_I 

# Testing implementations
book_data, set_book_chars = read_data()
out_in_dimension = len(set_book_chars)
print("Output Dimension: ", out_in_dimension)
char_to_index, index_to_char = GetKeyValueStore(book_data)
dict_items_ci = char_to_index.items()
dict_items_ic = index_to_char.items()
print(list(dict_items_ci)[:5])
print(list(dict_items_ic)[:5])

def OneHot(character_index, number_distinct_characters):
    character_one_hot = np.zeros(shape=(number_distinct_characters,1))
    character_one_hot[character_index,0] = 1
    
    return character_one_hot

class RNN(object):
    
    # Standard hyperparams and params for this RNN
    def __init__(self, k=1, m=100, eta=0.1, seq_length=25, sig=0.001):
        
        self.m = m # Hidden state
        self.k = k # Dimensionality of output and input
        self.eta = eta # Learning rate
        self.seq_length = seq_length # Length of input (training) sequence
        
        # b = bias for equation @ at (hidden-to-output)
        self.b = np.zeros((m, 1))
        # c = bias for equation @ ot (before prediction)
        self.c = np.zeros((k, 1))
        np.random.seed(1)
        # U = applied to x (input-to hidden connection)
        self.U = np.random.normal(size=(m, k), loc=0, scale=sig)
        # W = applied to ht-1 (hidden to hidden connection)
        self.W = np.random.normal(size=(m, m), loc=0, scale=sig)
        # V = applied to at (hidden-to-output connection)
        self.V = np.random.normal(size=(k, m), loc=0, scale=sig)

def tanh(a):
    a = np.clip(a, -700, 700)
    return (np.exp(a)-np.exp(-a))/(np.exp(a)+np.exp(-a))

def softmax(x):
    x = np.clip(x, -700, 700)
    exp_x = np.exp(x)
    P = exp_x/exp_x.sum(axis=0)

    return P

def SynthesizeText(rnn, x0, h0, l, stop_char=None):
    
    # Initial shapes and vals
    Y = np.zeros(shape=(rnn.k,l))
    x = x0
    h_prev = h0
    for t in range(l):
        
        # Make predictions
        at = rnn.W@h_prev+rnn.U@x+rnn.b
        ht = tanh(at)
        ot = rnn.V@ht+rnn.c
        p = softmax(ot)

        # Randomly create next input sequence w.r.t. predicted output
        x = np.random.multinomial(1, np.squeeze(p))[:,np.newaxis]

        Y[:,[t]] = x[:,[0]]

        # Terminate if stop char is matched
        if all(x==stop_char):
            Y = Y[:,0:(t+1)]
            break

        # Update previous hidden state for next sequence iteration
        h_prev = ht

    return Y

def ForwardPass(rnn, X, Y, h0):

    # Create empty lists for storing the final and intermediary vectors (by sequence iterations)
    seq_length = X.shape[1]
    pt, ot, ht, at = [None]*seq_length, [None]*seq_length, [None]*seq_length, [None]*seq_length

    # Iterate the input sequence of one hot encoded characters
    loss = 0
    for t in range(seq_length):
        if t==0:
            # First forward pass form input to first hidden state
            at[t] = rnn.W@h0+rnn.U@X[:,[t]]+rnn.b
        else:
            # Remaining forward passes. We multiply with previous hidden state
            at[t] = rnn.W@ht[t-1]+rnn.U@X[:,[t]]+rnn.b
        
        # activation
        ht[t] = tanh(at[t])
        # Output vector (of unnormalized log probas for each class) at time t
        ot[t] = rnn.V@ht[t]+rnn.c
        # Output proba vector at time t
        pt[t] = softmax(ot[t])
        # Cross entropy loss
        loss -= np.log(Y[:,[t]].T@pt[t])[0,0]

    return loss,[h0]+ht, at, pt

def BackPropagation(rnn, X, Y, pt, ht, at):

    # Extract h0
    h0 = ht[0]
    # Extract rest of hiden states
    ht = ht[1:]

    # Initialize the gradients 
    all_grads = dict()
    for parameter in ['b','c','W','U','V']:
        all_grads[parameter] = np.zeros_like(vars(rnn)[parameter])

    # Iterate inversively the input sequence of one hot encoded characters
    seq_length = X.shape[1]
    grad_a = [None]*seq_length
    for t in range((seq_length-1), -1, -1):
        # Computation of the last gradient of o
        g = -(Y[:,[t]]-pt[t]).T
        # Computation of V and c
        all_grads['V'] += g.T@ht[t].T
        all_grads['c'] += g.T
        # Computation of grad h
        if t<(seq_length-1):
            grad_h = g@rnn.V+grad_a[t+1]@rnn.W
        else:
            grad_h = g@rnn.V
        # Grad a, using h
        grad_a[t] = grad_h@np.diag(1-ht[t][:,0]**2)
        
        # First layer (time t =0 ) for W
        if t==0:
            all_grads['W'] += grad_a[t].T@h0.T
        else:
            all_grads['W'] += grad_a[t].T@ht[t-1].T
        # Computation of grad U and b
        all_grads['U'] += grad_a[t].T@X[:,[t]].T
        all_grads['b'] += grad_a[t].T

    # Clipping gradients to reduce chances of exploding grads.
    for parameter in ['b','c','U','W','V']:
        all_grads[parameter] = np.clip(all_grads[parameter], -5, 5)

    return all_grads

 # Computing gradients numerically
def ComputeGradsNum(rnn, X, Y, h0, h=1e-4):
    all_grads = dict()
    for param in ['b','c','U','W','V']:
        all_grads[param] = np.zeros_like(vars(rnn)[param])
        for i in range(vars(rnn)[param].shape[0]):
            for j in range(vars(rnn)[param].shape[1]):
                rnn_try = copy.deepcopy(rnn)
                vars(rnn_try)[param][i,j] += h
                l2, _, _, _ = ForwardPass(rnn_try, X, Y, h0)
                vars(rnn_try)[param][i,j] -= 2*h
                l1, _, _, _ = ForwardPass(rnn_try, X, Y, h0)
                all_grads[param][i,j] = (l1-l2)/(2*h)
    
    return all_grads

# Comparing analytical with numerical gradients.
def CheckGradients(grads_n, grads_a):

    for param in ['b','c','U','W','V']:
        print('-'*50)
        rel_error = abs(grads_n[param]-grads_a[param])
        mean_error = np.mean(rel_error<1e-6)
        max_error = rel_error.max()
        print('Percentage of absolute error smaller than 1e-6 for parameter: '+param+', is '+str(mean_error*100)+ \
              '%, and the maximum error is '+str(max_error))

# Optimizing gradient descent with ADAGRAD   
def AdaGrad(m_old, g, param_old, eta):
    m_new = m_old + np.power(g, 2)
    param = param_old - (eta / np.sqrt(m_new + np.finfo('float').eps)) * g

    return param, m_new

#########################################

# Testing implementations

#########################################

book_data, book_chars = read_data()
K = len(book_chars)
rnn = RNN(k=K, m=100)
print(rnn.b.shape, rnn.U.shape, rnn.W.shape, rnn.V.shape, rnn.c.shape)
print("Default sequence length: ", rnn.seq_length)
print("Default hidden state: ", rnn.m)
print("Input layer: ", K)


char_to_ind = {}
ind_to_char = {}
for idx, x in enumerate(book_chars):  # Create the enconding conversors
    char_to_ind[x] = idx
    ind_to_char[idx] = x
'''
print(type(char_to_ind))

# Input book data or unique book chars?
X_one_hot, Y_one_hot, _ = OneHot(X_chars, Y_chars, char_to_ind, rnn)

# Checking dims and one-hot output (x and y should be equal here)
print("X and Y one-hot shape: ", X_one_hot.shape)
unique, counts = np.unique(Y_one_hot, return_counts=True)
print("X and Y unique value counts: ", dict(zip(unique, counts)))
'''

# Set h0 to zero vector (?)
h0 = np.zeros(shape=(rnn.m,1) )
print(h0.shape)
sq_len = rnn.seq_length
# First sequence length chars of book data
X_chars = book_data[0:sq_len]
Y_chars = book_data[1:sq_len+1]
X_chars, Y_chars

print("Input chars: ", X_chars)
print("Output chars: ", Y_chars)

X_one_hot = np.zeros(shape=(K,sq_len))
Y_one_hot = np.zeros(shape=(K,sq_len))
for t in range(sq_len):
    X_one_hot[:,[t]] = OneHot(character_index=char_to_ind[X_chars[t]], number_distinct_characters=K)
    Y_one_hot[:,[t]] = OneHot(character_index=char_to_ind[Y_chars[t]], number_distinct_characters=K)
    
# Test forward pass
loss, ht, at, prob = ForwardPass(rnn, X_one_hot, Y_one_hot, h0)
print(loss)
print(prob[0].shape, 'Probas shape')

# Test backward pass
grad_a = BackPropagation(rnn, X_one_hot, Y_one_hot, prob, ht, at)
#grad_n = ComputeGradsNum(rnn, X_one_hot, Y_one_hot, h0)
#print(grad_n['U'], '3')
#print(type(grad_a))
#print(type(grad_n))
#print(grad_a['U'])
#print(grad_n)
#print(grad_n)
#CheckGradients(grad_a, grad_n)
print()

# Trains the network, stores the best model.
def main(rnn,book_data,ind_to_char,char_to_ind,seq_length =25,number_updates=100000,max_epochs=np.inf,best_rnn=True,prev_train=False,
              output=False,output_loss_freq=1000,output_sample_freq=10000,output_sample_len=200,
              output_stop_char=None):
        
        prev_u_l = True
        try:
           # current_update
           # smooth_loss
           print("")
        except:
            prev_u_l = False
        
        if not prev_train or not prev_u_l:
            
            # Init current update and smooth loss list
            current_update = 1
            smooth_loss = []
            
            # Create AdaGrad memory parameters matrix
            for param in ['b','c','U','W','V']:
                vars(rnn)[param+'_history'] = np.zeros_like(vars(rnn)[param])
            
            # Define minimum loss and initialize best network parameters matrix (if required)
            if best_rnn:
                smooth_loss_min = np.inf
                smooth_loss_min_update = 1
                for param in ['b','c','U','W','V']:
                    vars(rnn)[param+'_best'] = vars(rnn)[param]

        # Iterate updates
        current_epoch = 1
        while current_update<=number_updates:
            
            # Define the initial previous hidden state for next epoch (full text iteration)
            h_prev = np.zeros(shape=(rnn.m,1))

            # Iterate input text by blocks of seq_length characters
            for e in range(0, len(book_data)-1, seq_length):
                
                if e>len(book_data)-seq_length-1:
                    break

                # Generate the sequence data 
                X_chars, Y_chars = book_data[e:(e+seq_length)], book_data[(e+1):(e+1+seq_length)]
                X = np.zeros(shape=(rnn.k,seq_length))
                Y = np.zeros(shape=(rnn.k,seq_length))
                
                # One hot encode sequence data
                for t in range(rnn.seq_length):
                    X[:,[t]] = OneHot(char_to_ind[X_chars[t]], rnn.k)
                    Y[:,[t]] = OneHot(char_to_ind[Y_chars[t]], rnn.k)
                    
                   # X[:,[t]] = one_hot(char_to_ind[X_chars[t]], 80)
                   # Y[:,[t]] = one_hot(char_to_ind[Y_chars[t]], 80)

                # Forward and backward pass
                loss, h, a, p = ForwardPass(rnn, X, Y, h_prev)
                newGRADS = BackPropagation(rnn, X, Y, p, h, a)
              #  print('hönder')

                # Store smooth loss
                if current_update==1:
                    smooth_loss.append(loss)
                else:
                    smooth_loss.append(0.999*smooth_loss[-1]+0.001*loss)

                # AdaGrad will update params weights w.r.t. frequencies in sequence data. 
                for param in ['b','c','U','W','V']:
                    vars(rnn)[param+'_history'] += newGRADS[param]**2
                    vars(rnn)[param] += -rnn.eta*newGRADS[param]/ \
                        np.sqrt(vars(rnn)[param+'_history']+np.spacing(1))
                
                # Check if best loss improved for current RNN 
                # If so, update min loss and its corresponding update step
                if best_rnn and smooth_loss[-1]< smooth_loss_min:
                    smooth_loss_min = smooth_loss[-1]
                    smooth_loss_min_update = current_update
                
                # Outputs to the console.
                if output:
                    
                    # Show loss
                    shown_loss = False
                    if current_update%output_loss_freq==0 or current_update==1:
                        shown_loss = True
                        print('Update '+str(current_update)+' with loss: '+ \
                              str(smooth_loss[-1]))
                        
                    # Show synthesized sample
                    if current_update % output_sample_freq==0 or current_update==1:
                        synthesized = \
                            SynthesizeText(rnn, x0=X[:,[0]], h0=h_prev, l=output_sample_len,
                                            stop_char=output_stop_char)
                        synthesized_chars = []
                        for i in range(synthesized.shape[1]):
                            char = ind_to_char[np.where(synthesized[:,i]>0)[0][0]]
                            synthesized_chars.append(char)
                        if shown_loss:
                            print('Synthesized sample:\n'+''.join(synthesized_chars)+'\n')
                        else:
                            print('Update '+str(current_update)+' with loss: '+ \
                                  str(smooth_loss[-1])+'\nSynthesized sample:\n'+ \
                                  ''.join(synthesized_chars)+'\n')
                    
                current_update += 1
                if current_update>number_updates:
                    return smooth_loss
                    break

                # Update the previous hidden state for next iteration
                h_prev = h[seq_length]

            current_epoch += 1
            if current_epoch>max_epochs:
                return smooth_loss
                break
        
        # Update the final training parameters with the best stored network (if required)
        if best_rnn:
            for param in ['b','c','U','W','V']:
                vars(rnn)[param] = vars(rnn)[param+'_best']
                
      
 
#########################################

# 1st experiment, K=80, m = 100, updates = 100000

#########################################               
    
book_data, book_chars = read_data()
K = len(book_chars)
rnn = RNN(k=K, m=100)
print(rnn.b.shape, rnn.U.shape, rnn.W.shape, rnn.V.shape, rnn.c.shape)

smooth_loss = main(rnn, book_data, ind_to_char,char_to_ind, seq_length=25, number_updates=100000, max_epochs=np.inf, best_rnn=True,
    prev_train=False, output=True, output_loss_freq=10000, output_sample_freq=10000, output_sample_len=200,
    output_stop_char=None)

# Plots learning curve
def plot(smooth_loss, title='', length_text=None, seq_length=None):
    
    _, ax = plt.subplots(1, 1, figsize=(15,5))
    plt.title('Learning curve '+title)
    ax.plot(range(1, len(smooth_loss)+1), smooth_loss, color='r')
        
    optimal_update = np.argmin(smooth_loss)
    optimal_loss = np.round(smooth_loss[optimal_update], 4)
    label = 'Optimal training loss: '+str(optimal_loss)+' at update '+str(optimal_update+1)
    ax.axvline(optimal_update, c='b', linestyle='--', linewidth=1, label=label)
        
    if length_text is not None and seq_length is not None:
        updates_per_epoch = len([e for e in range(0, length_text-1, seq_length) \
                                 if e<=length_text-seq_length-1])
        for e in range(updates_per_epoch, len(smooth_loss)+1, updates_per_epoch):
            label = 'Epoch' if e==updates_per_epoch else ''
            ax.axvline(e, c='g', linestyle='--', linewidth=1, label=label)
        
    ax.set_xlabel('Update step')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(False)
    
plot(smooth_loss, title='for 100000 update steps. Eta = 0.1, sequence length = 25')

synthesized = SynthesizeText(rnn, x0=X_one_hot[:,[0]], h0=h0, l=1000)
synthesized_chars = []
for i in range(synthesized.shape[1]):
    chars = ind_to_char[np.where(synthesized[:,i]>0)[0][0]]
    synthesized_chars.append(chars)
print(''.join(synthesized_chars))

#########################################

# 2nd experiment, K=80, m = 100, updates = 300000

#########################################  

K = len(book_chars)
rnn = RNN(k=K, m=100)
#print(rnn.b.shape, rnn.U.shape, rnn.W.shape, rnn.V.shape, rnn.c.shape)
print()
smooth_loss = main(rnn, book_data, ind_to_char,char_to_ind, seq_length=25, number_updates=300000, max_epochs=np.inf, best_rnn=True,
    prev_train=False, output=True, output_loss_freq=10000, output_sample_freq=10000, output_sample_len=200,
    output_stop_char=None)

plot(smooth_loss, title='for 300000 update steps. Eta = 0.1, sequence length = 25')

synthesized = SynthesizeText(rnn, x0=X_one_hot[:,[0]], h0=h0, l=1000)
synthesized_chars = []
for i in range(synthesized.shape[1]):
    chars = ind_to_char[np.where(synthesized[:,i]>0)[0][0]]
    synthesized_chars.append(chars)
print(''.join(synthesized_chars))
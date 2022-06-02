#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 11:10:34 2022

@author: akseluhr
"""

import numpy as np
import pandas as pd
import copy

#########################################

# Read in data and preprocess

#########################################

def read_data():
    file = open('/Users/akseluhr/Documents/GitHub/RNN-Synthezising-English-Chars/goblet_book.txt', 'r')
    book_data = file.read()
    return book_data, set(book_data)

book_data, set_book_chars = read_data()
out_in_dimension = len(set_book_chars)

print(out_in_dimension)
      
# Creating maps (dicts) for quick check ups in one-hot

def GetKeyValueStore(data):
    char_to_index = {}
    index_to_char = {}
    key_val = 0
    for c in set_book_chars:
        char_to_index[c] = key_val
        index_to_char[key_val] = c
        key_val += 1
    print(char_to_index)
    return char_to_index, index_to_char

def OneHot(X_chars, Y_chars, c_t_i, rnn):
   # print(rnn.seq_length)
    X = np.zeros((rnn.k, len(X_chars)))
    Y = np.zeros((rnn.k, len(Y_chars)))
   # print("X input shape: ", X.shape)
   # print("Y output shape: ", Y.shape)
    for i in range(len(X_chars)):
        X[c_t_i[X_chars[i]], i] = 1
        Y[c_t_i[Y_chars[i]], i] = 1
        
    return X, Y

# Testing implementation
char_to_index, index_to_char = GetKeyValueStore(book_data)
dict_items_ci = char_to_index.items()
dict_items_ic = index_to_char.items()
#print(list(dict_items_ci)[:5])
#print(list(dict_items_ic)[:5])


#########################################

# Creating the model as a class

#########################################

class RNN:
    
    # Standard hyperparams and params for this RNN
    def __init__(self, k=1, m=5, eta=0.1, seq_length=25, sig=0.01):
        
        self.m = m # Hidden state
        self.k = k # Dimensionality of output and input
        self.eta = eta # Learning rate
        self.seq_length = seq_length # Length of input (training) sequence
        
        np.random.seed(1)
        # b = bias for equation @ at (hidden-to-output)
        self.b = np.zeros((m, 1))
        # c = bias for equation @ ot (before prediction)
        self.c = np.zeros((k, 1))

        # U = applied to x (input-to hidden connection)
        self.U = np.random.normal(size=(m, k), loc=0, scale=sig)
        # W = applied to ht-1 (hidden to hidden connection)
        self.W = np.random.normal(size=(m, m), loc=0, scale=sig)
        # V = applied to at (hidden-to-output connection)
        self.V = np.random.normal(size=(k, m), loc=0, scale=sig)

#########################################

# Network functions

#########################################

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    
def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0)

# rnn = rnn object
# h0 = hidden state at time 0
# x0 dim = d x 1, represents the first dummy input vector of the RNN
# n = lenth of the sequence to genreate
def SynthesizeText(rnn, h0, x0, n):
    
    # x will be next input vector
    xt = np.copy(x0)
    print(np.copy(h0))
    # New axis basically adds a new dimension.
    print(np.copy(h0)[:, np.newaxis])
    ht = np.copy(h0)[:, np.newaxis]
    samples = np.zeros((x0.shape[0], n))

    # Iterating over the length of sequence we want to generate
    # 
    for t in range(n):
        at = np.dot(rnn.W, ht) + np.dot(rnn.U, xt) + rnn.b
        ht = tanh(at)
        ot = np.dot(rnn.V. ht) + rnn.c
        p = softmax(ot)
        
        # randomly select character based on the output probability socres p:
        # cumsum returns for column c: xi + xi-1 
       # cp = np.cumsum(p)
        
        # flatten () projects the nd array to 1d
        rnd_char = np.random.choice(range(xt.shape[0]), 1, p=p.flatten())
        xt = np.zeros(xt.shape)
        xt[rnd_char] = 1
        # Fill samples up to t with rnd chars
        samples[:, t] = xt.flatten()
        
    return samples

def ForwardPass(rnn, h0, X_data, Y_data):
    
    # Hidden state at time t of m x 1
    ht = np.zeros((h0.shape[0], X_data.shape[1]))
    # Hidden state at time t before non-linearity
    at = np.zeros((h0.shape[0], X_data.shape[1]))
   # X_chars = list(set_book_chars[1:rnn.seq_length])
    #Y_chars = list(set_book_chars[2:rnn.seq_length+1])
    probas = np.zeros(X_data.shape)
    n = X_data.shape[1] # input dimension
    loss = 0
    for t in range(n):
        if t == 0:
            # First forward pass form input to first hidden state
            at[:, t] = (np.dot(rnn.W, h0[:, np.newaxis]) + np.dot(rnn.U, X_data[:, t][:, np.newaxis]) + rnn.b).flatten()  
        else:
            # Remaining forward passes. We multiply with previous hidden state
            # ' From all '
            at[:, t] = (np.dot(rnn.W, ht[:, t - 1][:, np.newaxis]) + np.dot(rnn.U, X_data[:, t][:, np.newaxis]) + rnn.b).flatten()

        ht[:, t] = tanh(at[:, t]) # activation

        # Output vector (of unnormalized log probas for each class) at time t
        ot = np.dot(rnn.V, ht[:, t][:, np.newaxis]) + rnn.c # ?
        p = softmax(ot)

        # Output proba vector at time t
        probas[:, t] = p.flatten()
        loss = -np.sum(np.log(np.sum(Y_data[:, t] * probas[:, t], axis=0)))

    return loss, ht, at, probas


"""
def CrossEntropyLoss(probas, Y_data):
    #print(Y_data, "y")
    probas = 10**-10
    loss = -np.sum(np.log(np.sum(Y_data * probas, axis=0)))
   # print(loss)
    return loss


def ComputeCost(rnn, h0, X, Y):
    
    if(batch_norm):
        P, S_batch_Norm, S, H, mean, var = EvaluateClassifier(X, W, b, k, gamma, beta, batch_norm=True)
    else:
        P, H = ForwardPass(X, W, b, k)
    
    sqrt_W = 0 
    for i in range(len(W)):
        sqrt_W += (W[i]**2).sum()
        
    lcr = -np.sum(np.multiply(Y, np.log(P)))
    Reg_term = lambd*sqrt_W
    J = lcr/X.shape[1]+Reg_term
    
    return J
"""

def BackPropagation(rnn, h0, ht, at, probas, X_data, Y_data):
    
    #n = X_data.shape[1] # input dimension
    output_dim = Y_data[1].shape
    ht_grad = [None]*output_dim[0]
    at_grad = [None]*output_dim[0]
    #print(len(ht_grad))
   # print(len(at_grad), "at min")
    ot_grad = -(Y_data - probas).T
    
    ht_grad[0] = np.dot(ot_grad[0][np.newaxis, :], rnn.V)
    at_grad[0] = np.dot(ht_grad[0], np.diag(1 - np.power(np.tanh(at[:, -1]), 2)))
       
   # print("hal", ot_grad)
   # print(at_grad)

    for t in range(1, output_dim[0]):
        ht_grad[t] = np.dot(ot_grad[t][np.newaxis, :], rnn.V + np.dot(at_grad[t-1], rnn.W))
        at_grad[t] = np.dot(ht_grad[t-1], np.diag(1 - np.power(np.tanh(at[:, t]), 2)))

    store_grads = RNN()
    store_grads.V = np.dot(ot_grad.T, ht.T)
    print(ht.shape)
    print(at.shape)
    
    # Might be a problem here
    h_aux = np.zeros(ht.shape)  # Auxiliar h matrix that includes h_prev
    h_aux[:, 0] = h0
    h_aux[:, 1:] = ht[:, 0:-1]
   # print(type(at_grad))
    at_grad = np.asarray(at_grad) # For transposing
   # print(type(at_grad))


    store_grads.W = np.dot(at_grad.T, h_aux.T)
    store_grads.U = np.dot(at_grad.T, X_data.T)
    store_grads.b = np.sum(at_grad, axis=0)[:, np.newaxis]
    store_grads.c = np.sum(ot_grad, axis=0)[:, np.newaxis]
    return store_grads
    


def ComputeGradsNum(rnn, X, Y, h0, h=1e-4):
   
    # Iterate parameters and compute gradients numerically
    # Stored in a dict
    grads = dict()
    for param in ['V','W','U','b','c']:

        grads[param] = np.zeros_like(vars(rnn)[param])
 
        for i in range(vars(rnn)[param].shape[0]):
            for j in range(vars(rnn)[param].shape[1]):
                rnn_try = copy.deepcopy(rnn)
                #print(rnn.U, 'RNN.U')
                #print(vars(rnn_try)[param], 'Kopia')#[i,j])
                vars(rnn_try)[param][i,j] += h # takes same weights that were used for analytical grads
                loss1, ht, at, probas = ForwardPass(rnn_try, h0, X, Y)
                #print(loss1, "LOSS1")
                #print(probas, "Probas Num")
               # loss1 = CrossEntropyLoss(probas, Y)
                #print(loss1)
                vars(rnn_try)[param][i,j] -= 2*h
                loss2, ht, at, probas = ForwardPass(rnn_try, h0, X, Y)
               # print(loss2, "LOSS2")
                #loss2 = CrossEntropyLoss(probas, Y)
                #print(loss2)
                grads[param][i,j] = (loss2-loss1)/(2*h)
                #print("final", grads[param][i,j])

        #print(grads[param], "HÄRVAREVILT")
    return grads


def CheckGradients(grads_a, grads_n):
    
    for param in ['V','W','U','b','c']:
        #print('-'*50)
        #print(grads_n[param].shape, "NUM")
        #print(getattr(grads_a, param).shape, "ANALYTIC")
        print(param)
        rel_error = abs(grads_n[param]-getattr(grads_a, param))
        #print(rel_error, 'rel err')
        mean_error = np.mean(rel_error<1e-6)
        print(mean_error, ' mean err')
        max_error = rel_error.max()
        print('Percentage of absolute error smaller than 1e-6 for parameter: '+param+', is '+str(mean_error*100)+ \
              '%, and the maximum error is '+str(max_error))
    
#########################################

# Testing network functions

#########################################

#########################################

# Testing implementations

#########################################

book_data, book_chars = read_data()
dimension = len(book_chars)
rnn = RNN(k=dimension)
print("Default sequence length: ", rnn.seq_length)
print("Default hidden state: ", rnn.m)


# First sequence length chars of book data
X_chars = book_data[0:rnn.seq_length]
Y_chars = book_data[1:rnn.seq_length+1]

print("Input chars: ", X_chars)
print("Output chars: ", Y_chars)

char_to_ind = {}
ind_to_char = {}
for idx, x in enumerate(book_chars):  # Create the enconding conversors
    char_to_ind[x] = idx
    ind_to_char[idx] = x
    
print(type(char_to_ind))

# Input book data or unique book chars?
X_one_hot, Y_one_hot = OneHot(X_chars, Y_chars, char_to_ind, rnn)

# Checking dims and one-hot output (x and y should be equal here)
print("X and Y one-hot shape: ", X_one_hot.shape)
unique, counts = np.unique(Y_one_hot, return_counts=True)
print("X and Y unique value counts: ", dict(zip(unique, counts)))


# Set h0 to zero vector (?)
h0 = np.zeros(rnn.m)  


# Test forward pass
loss, ht, at, prob = ForwardPass(rnn, h0, X_one_hot, Y_one_hot)
#print(ht)
#print(at)
#print(prob)
print("LOSS ANALYTICAL: ", loss)


# Test backward pass
grad_a = BackPropagation(rnn, h0, ht, at, prob, X_one_hot, Y_one_hot)
grad_n = ComputeGradsNum(rnn, X_one_hot, Y_one_hot, h0)
print(type(grad_a))
print(type(grad_n))
#print(grad_a['U'])
#print(grad_n)
#print(grad_n)
CheckGradients(grad_a, grad_n)
    
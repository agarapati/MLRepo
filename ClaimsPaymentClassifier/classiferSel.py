# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 15:34:11 2020
HyperGrid top
@author: garapati

COde leveraged from David

The class where the classifier is invoked is call createHyperGrid
These objects are stored in list (list of dictonary)
To plot,  I iterate throgh these objects, collect the hyper parameters
and accuracy to plot

"""

from createHyperGrid import *;
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from sklearn import svm
clfDictGoodExample = {RandomForestClassifier: {"min_samples_split": [2,3,4],
                                      "n_jobs": [1,2,3]},
                     LogisticRegression: {"tol": [0.001,0.01,0.1],"penalty" : ['l1', 'l2']},svm.SVC:{"gamma":[1,0.1,0.001],"kernel":['rbf']}}
                     

X = np.array([[1, 2], [3, 4], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5]])
#X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
#y = np.array([1, 2, 3, 4])
y= np.random.choice([0, 1], size=(X.shape[0],), p=[1./3, 2./3])

#clfsList = [RandomForestClassifier, LogisticRegression]

#clfDict = {RandomForestClassifier: {"min_samples_split": [2], "n_jobs": [1,2,3]} , LogisticRegression: {"tol": [0.001,0.01,0.1]}}
##clfDict = {RandomForestClassifier: {} ,LogisticRegression: {"tol": 0.1,"random_state":1} }
#clfDict = {'RandomForestClassifier': 2 }
val=[]  
clfsAccuracyDict = {}
         
#id=1
#for key , value in clfDict.items() :
#    print("clf -> " , key)
#    print("param -> ",value)
#    z=myGridSearch.createHyperGrid(X,y,value,key,4) ###creating objects for each classifier and collecting the dictionaries in the list
#    val.append(z)
###iterate through length of dictionary and extract the values to plot 
#for i in range(len(val)):


for k1, v1 in clfDictGoodExample.items(): # go through the inner dictionary of hyper parameters
    #Nothing to do here, we need to get into the inner nested dictionary.

    try:
        k2,v2 = zip(*v1.items()) 
        for values in product(*v2): #for the values in the inner dictionary, get their unique combinations from product()
            hyperSet = dict(zip(k2, values)) # create a dictionary from their values
            z=myGridSearch.createHyperGrid(X,y,hyperSet,k1,4) ##calling the object 
            val.append(z)
            # print out the results in a dictionary that can be used to feed into the ** operator in run()
    except AttributeError:
        print("no k2 and v2 found")

    
for dic in val :
    for key in dic:
        #print("Hello" , dic[key]['accuracy'])
        k1 = dic[key]['clf']
        v1 = dic[key]['accuracy']
        k1Test = str(k1) #Since we have a number of k-folds for each classifier...
                         #We want to prevent unique k1 values due to different "key" values
                         #when we actually have the same classifer and hyper parameter settings.
                         #So, we convert to a string

        #String formatting
        k1Test = k1Test.replace('            ',' ') # remove large spaces from string
        k1Test = k1Test.replace('          ',' ')

        #Then check if the string value 'k1Test' exists as a key in the dictionary
        if k1Test in clfsAccuracyDict:
            clfsAccuracyDict[k1Test].append(v1) #append the values to create an array (techically a list) of values
        else:
            clfsAccuracyDict[k1Test] = [v1] #create a new key (k1Test) in clfsAccuracyDict with a new value, (v1)


print("Hello:::",clfsAccuracyDict)
# for determining maximum frequency (# of kfolds) for histogram y-axis
n = max(len(v1) for k1, v1 in clfsAccuracyDict.items())

# for naming the plots
filename_prefix = 'clf_Histograms_'

# initialize the plot_num counter for incrementing in the loop below
plot_num = 1

# Adjust matplotlib subplots for easy terminal window viewing
left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.6      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for space between subplots,
               # expressed as a fraction of the average axis width
hspace = 0.2   # the amount of height reserved for space between subplots,
               # expressed as a fraction of the average axis height

#create the histograms
#matplotlib is used to create the histograms: https://matplotlib.org/index.html
for k1, v1 in clfsAccuracyDict.items():
    # for each key in our clfsAccuracyDict, create a new histogram with a given key's values
    fig = plt.figure(figsize =(10,10)) # This dictates the size of our histograms
    ax  = fig.add_subplot(1, 1, 1) # As the ax subplot numbers increase here, the plot gets smaller
    plt.hist(v1, facecolor='green', alpha=0.75) # create the histogram with the values
    ax.set_title(k1, fontsize=25) # increase title fontsize for readability
    ax.set_xlabel('Classifer Accuracy (By K-Fold)', fontsize=25) # increase x-axis label fontsize for readability
    ax.set_ylabel('Frequency', fontsize=25) # increase y-axis label fontsize for readability
    ax.xaxis.set_ticks(np.arange(0, 1.1, 0.1)) # The accuracy can only be from 0 to 1 (e.g. 0 or 100%)
    ax.yaxis.set_ticks(np.arange(0, n+1, 1)) # n represents the number of k-folds
    ax.xaxis.set_tick_params(labelsize=20) # increase x-axis tick fontsize for readability
    ax.yaxis.set_tick_params(labelsize=20) # increase y-axis tick fontsize for readability
    #ax.grid(True) # you can turn this on for a grid, but I think it looks messy here.

    # pass in subplot adjustments from above.
    plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
    plot_num_str = str(plot_num) #convert plot number to string
    filename = filename_prefix + plot_num_str # concatenate the filename prefix and the plot_num_str
    plt.savefig(filename, bbox_inches = 'tight') # save the plot to the user's working directory
    plot_num = plot_num+1 # increment the plot_num counter by 1
plt.show()

        
    
    
    
    
    



   
    
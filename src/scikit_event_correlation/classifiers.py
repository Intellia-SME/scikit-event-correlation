import time
from math import exp
import pandas as pd
import numpy as np


from sklearn.neighbors import KNeighborsClassifier

def powerset(lst):
    # the power set of the empty set has one element, the empty set
	result = lst
	res = list(map(list, result))
	return res

# k = ageing factor, i = how many steps is the prob active
def linearAgeing(i):
	age = 1 - 0.05*i
	return age

def exponentialAgeing(i):
	k = 0.05
	age = exp(-k*i)
	return age


def updateFreq(event, freq):
	if event not in freq:
		freq[event] = 0
	freq[event] += 1


def updateGraph(prevState, event, graph):
	if event not in graph:
		graph[event] = {}

	if event not in graph[prevState]:
		graph[prevState][event] = 0
	graph[prevState][event] += 1

class EventCorrelationClassifier(KNeighborsClassifier):
    pass
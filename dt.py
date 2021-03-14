import sklearn
import pandas as pd
from collections import Counter
import math

from sklearn.datasets import load_iris

### compute entropy for a set of classification, provided as a pandas Series
def entropy(classes) :
    vals = set(classes)
    counts = Counter(classes)
    ent = 0.0
    for val in vals :
        frequency = counts[val] / len(classes)
        ent += -1 * frequency * math.log(frequency, 2)
    return ent

### Assume that both attribute and classes are pandas Series
### For each value of attribute, compute the entropy. Then return the weighted sum
def remainder(attribute, classes) :
   pass


### assume that data is a pandas dataframe, and classes is a pandas series
### For each column in the dataframe, compute the remainder and select the column with the lowest
### remainder

def selectAttribute(data, classes)
   pass

### Now we're ready to build a Decision Tree.
### A tree consists of one or more Nodes.
### A Node is either a leaf node, which has a value and no children
### Or it is a non-leaf, in which case it has an attribute that it tests and a set of children.

class Node :
    def __init__(self, attribute=None, value=None):
        self.attribute = attribute
        self.value=value
        self.children={}

    def isLeaf(self):
        return len(self.children) == 0

    ### you'll implement this
    def classify(self, instance):
       pass

    def __repr__(self) :
        return "%s %s" % (self.attribute, self.value)

##
class DecisionTree :
    def __init__(self, root) :
        self.root = root

    ### assume instance is a pandas dataframe - use node.classify as a helper.
    def classify(self, instance):
        pass


### construct a decision tree. Inputs are a pandas dataframe containing a dataset,
### and an attributeDict that maps each attribute to the possible values it can take on.

### We make the tree recursively. There are three base cases:
### 1. All the data is of the same class.
###   In this case, we are at a leaf node. set the value to be the classification.
### 2. We are out of attributes to test.
###   In this case, apply ZeroR.
### 3 We are out of data
###   In this case, apply ZeroR.
### Return the node
### Otherwise :
###  1. Use selectAttribute to find the attribute with the largest information gain.
###  2. Break the data into subsets according to each value of that attribute.
###  3. For each subset, call makeNode

def makeNode(df, attributeDict) :
    pass




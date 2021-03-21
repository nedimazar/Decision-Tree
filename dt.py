import sklearn
import pandas as pd
import numpy as np
from collections import Counter
import math
from restaurant import getARFFData

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
def remainder(attribute, classes):
    remainder = 0
    for val in attribute.unique():
        subset = classes[attribute == val]
        remainder = remainder + entropy(subset) * len(subset) / len(classes)
    return remainder


### assume that data is a pandas dataframe, and classes is a pandas series
### For each column in the dataframe, compute the remainder and select the column with the lowest
### remainder

def selectAttribute(data, classes):
    minRemainder = 10e9
    selected = None
    
    for attribute in data:
        r = remainder(data[attribute], classes)
        if r < minRemainder:
            minRemainder = r
            selected = attribute
    return selected

### Now we're ready to build a Decision Tree.
### A tree consists of one or more Nodes.
### A Node is either a leaf node, which has a value and no children
### Or it is a non-leaf, in which case it has an attribute that it tests and a set of children.

class Node :
    def __init__(self, attribute=None, value=None, children = []):
        self.attribute = attribute
        self.value=value
        self.children=children

    def isLeaf(self):
        return len(self.children) == 0

    ### you'll implement this
    def classify(self, instance):
       pass

    def __repr__(self) :
        return "%s %s" % (self.attribute, self.value)

##
class DecisionTree :
    def __init__(self, root = None) :
        self.root = root

    ### assume instance is a pandas dataframe - use node.classify as a helper.
    def classify(self, instance):
        pass
    
    def fit(self, X, y):
        self.root = self.makeNode(X, y)

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
    
    ### Inputs are:
    ### X: The features
    ### y: The classifications
    ### parentVal: The classification that the parent would give in case we run out of data in a subset
    def makeNode(self, X, y, parentVal = None):
        if(len(y.unique()) == 1):
            return Node(attribute = None, value = y.unique()[0])
        elif X.empty:
            return Node(attribute = None, value = parentVal)
        else:
            attribute = selectAttribute(X, y)
            children = []
            
            for val in X[attribute].unique():
                mask = X[attribute] == val
                
                subsetX = X[mask]
                subsetY = y[mask]
                
                children.append(self.makeNode(subsetX, subsetY, parentVal = val))
                
            return(Node(attribute = attribute, value = parentVal, children = children))

    def predict(self, row):
        pass


def main():
    a, b = getARFFData()
    X = a.drop(['WillWait'], axis = 1)
    y = a['WillWait']
    dt = DecisionTree()
    dt.fit(X, y)

    print(dt.root.children)


main()
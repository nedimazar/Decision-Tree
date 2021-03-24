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
    def __init__(self, attribute=None, value=None, children = {}):
        self.attribute = attribute
        self.value=value
        self.children=children

    def isLeaf(self):
        return len(self.children) == 0

    ### you'll implement this
    def classify(self, instance):
       pass

    def __repr__(self) :
        return "Attribute: %s | Value: %s" % (self.attribute, self.value)

##
class DecisionTree :
    def __init__(self, root = None) :
        self.root = root

    ### assume instance is a pandas dataframe - use node.classify as a helper.
    def classify(self, instance):
        pass
    
    def fit(self, X, y):
        zeroR = y.mode()[0]
        self.root = self.makeNode(X, y, zeroR)
    
    def makeNode(self, X, y, parentZeroR):
        if len(y.unique()) == 1:
            return Node(attribute = None, value = parentZeroR)
        elif (X.shape[1] == 0) or X.empty:
            mode = y.mode()[0]
            return Node(attribute = None, value = mode)
        
        attribute = selectAttribute(X, y)
        zeroR = y.mode()[0]
        
        children = {}
        
        for val in X[attribute].unique():
            mask = X[attribute] == val
            
            subsetX = X[mask]
            subsetY = y[mask]
            
            children[val] =  self.makeNode(subsetX, subsetY, zeroR)
            
        return Node(attribute = attribute, value = None, children = children)
            
    def predict(self, rows):
        predictions = []
        for index, data in rows.iterrows():
            predictions.append(self.classify(data, self.root))
        return predictions
    
    def classify(self, row, node):
        attribute = node.attribute
        
        if len(node.children) == 0 or attribute == None:
            return node.value
        
        trueValue = row[attribute]

        for x in node.children:
            if x == trueValue:
                node = node.children[x]
                return self.classify(row, node)


def main():
    data = pd.read_csv('tennis.arff')
    X = data.drop(['play'], axis=1)
    y = data['play']

    d = DecisionTree()
    d.fit(X, y)

    predictions = d.predict(X)

    print(predictions)

main()
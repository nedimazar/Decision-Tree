### Assignment 3: Knowledge Representation and Decision Trees

#### Due Tuesday, March 30 at the start of class.

#### 100 points. (plus 10 extra for cs686 students)


Question 1. (written - please submit your answers as a PDF):

Part 1 (5 points). Encode the English language sentences below as propositional logic. 
Please use the following abbreviations:

A: it's a great day <br>
X: it's sunny out<br>
Y: It's the weekend<br>
Z: There's no chores to do<br>
F: it's not raining<br>
G: It's Saturday<br>
B: The dishes are done<br>
C: The laundry is done<br>

1. If it's sunny, and it's the weekend, and there's no chores, then it's a great day.

2. It it's not raining, then it's sunny.

3. If it's Saturday, then it's the weekend

4. If the dishes are done and the laundry is done, then the chores are done.

5. Either the dishes are not done, or it's not raining.

6. The dishes are done.

7. If the laundry is not done, then the dishes are not done.

8. If the dishes are done and it's not raining, then it must be Saturday.

Part 2: (5 points) Trace the execution of forward chaining to prove that it's a great day. 

Part 3: (5 points) Trace the execution of backward chaining to prove that it's a great day.

Part 4. (5 points) Convert the sentences to CNF and use resolution with refutation to prove that it's a great day.


Question 2: Implement the ID3 decision tree algorithm.

In this section, you'll implement the basic ID3 decision tree algorithm from scratch. For this part, please do not use sklearn or any other ML libraries. (pandas and numpy are fine.)
The purpose of the exercise is to give you experience implementing these techniques yourself.

I've provided some skeleton code to get you started. You're welcome to adapt or modify it if you like.

In particular, I've provided code that will read in the restaurant dataset, populating both a pandas dataframe and a dictionary mapping attributes to possible values. 

I've also provided the basic structure for a decision tree. Again, please feel free to add methods to test or display the tree.

I've provided three datasets, all in the ARFF format. The first is the tennis data, the second the restaurant data, and the third is the breast cancer data.
The first two are mostly useful for testing.

I strongly recommend unit testing your code as you go, both with the restaurant and the tennis dataset. 

Part 1. (10 points) I've implemented entropy for you. Use that to implement remainder - it should take as input a Series representing one column of the dataset, and the corresponding classifications, and return the remainder contained in these classifications.

Part 2. (10 points) Next, implement selectAttribute. It should consider all attributes and choose the one that maximizes gain (or minimizes remainder)

Part 3. (10 points) Now you're ready to build your tree. A tree is really just a wrapper class for a Node.
A Node is either a leaf node, which just has a value, or it's not, in which case it has an attribute and children.
   makeNode should recursively construct your tree based on the pseudocode in the comments.
   
Also add a wrapper to the Tree class that initially calls makeNode and sets the root to the result of this call.

Part 4. (10 points) Now you're ready to implement classify. Classify should take as input a Series representing an instance and return the value predicted for this instance.

Part 5. (10 points) Now, let's test your tree. Using the cancer dataset, measure the tree's accuracy. Use 5-fold cross validation. Note - you should implement this yourself - do not use the sklearn module for this part. (it's good practice!) Please provide a script that allows us to easily run and test your code.

Question 3:
(15 points) In this part you'll compare the performance of your tree to the one included in sklearn. 

 To begin, implement the same setup as above using sklearn's [Decision Tree Classifier](https://scikit-learn.org/stable/modules/tree.html#classification). You can load the cancer dataset using sklearn's load_breast_cancer() function.
Use their [Cross validation helper](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-evaluating-estimator-performance) to evaluate the performance of the sklearn tree on the breast cancer data with both entropy and gini as splitting criteria. Prepare a PDF document containing a chart that shows your results.

Question 4: (15 points) 
A weakness of decision trees is their tendency to overfit to the training data. One solution to this is to use a *random forest*, which is a set of pruned decision trees, each trained on a subset of the data.
sklearn also contains [random forests](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees). 
Extend your analysis in question 3 to test the accuracy of random forests on the breast cancer data. Vary both the number of estimators (2,5, and 10) and the number of samples (25%, 50%, 100%). Add a chart comparing these results to your PDF.

Question 5: (686 students only). (10 points) Rule-based inference is a long-standing approach to a wide variety of problems, including system verification, 
code analysis, and medical diagnoses. One of the first systems to demonstrate the effectiveness of this approach was MYCIN. [This book](https://aitopics.org/misc/rule-based-expert-systems-mycin-experiments-stanford-heuristic-programming-project) provides a historical overview of MYCIN. Please read Chapter 1 (describing MYCIN) , Chapter 2.1-2.3 (describing rule-based systems), and Chapter 36 through section 36.2.3.

Prepare a report summarizing the contributions of MYCIN, spefically addressing these questions:
- What was MYCIN? What problem was it designed to solve from a medical perspective? Why was this important?
- What problem was MYCIN designed to address from an AI perspective? Why was this important?
- MYCIN is an example of a production system. What does that mean?
- MYCIN was one of the first systems to separate the knowledge base from the inference engine. What does this mean? Why is this approach important?
- MYCIN is referred to as an example of the "evidence gathering" paradigm. This is closely related to the fact that it uses backward chaining. What does it mean to say that MYCIN uses evidence gathering, and how does that make the inference process more efficient? 


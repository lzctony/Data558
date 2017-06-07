# Data 558 Project
### Statistical Machine Learning For Data Scientists

### Logistic Regression using L2-Regularized Paramter

We are going to use both gradient Descent and gast gradient desent algorithms
to solve the binary classfication problem.

### Organization of the  project

The project has the following structure:

   	Data558/
      |- src/
      	|- simulated.py
        	|- spam.py
        	|- comparison.py
        	|- __init__.py
      |- README.md

### Simple Simulated Dataset
Consider the iris dataset from the sklearn library

### Real World Dataset
Consider the Spam dataset from The Elements of Statistical Learning
```
https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.data

```

The simulated.py uses the iris dataset from sklearn library. We pick two classes and use the logistic regression with
L2-regularized paramter to do the binary classification.

The spam.py uses the real world data (spam) from the above url. The data comes from The Elements of Statistical Learning. We use the logistic regression with L2-regularized paramter to do the binary classification.

The comparison.py runs an experimental comparison between our own implementation and scikit-learn's by using the spam data

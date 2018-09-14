# DecisionTree
Decision Tree Classifier based on Information Gain, that works with attributes having 2 distinct values

This is the first assignment for my Introduction to Machine Learning (10-601) course. 
We had to implement a decision tree that takes data where attributes have only 2 distinct values. 

The repo contains two files inspect.py and decisionTree.py

1) Inspect.py:

To run it "python inspect.py #training_file_name# #output_file_name#"
It will calculate the entropy of the label and the error rate working as a majority vote classifier (i.e. a depth 0 decision tree).

2) decisionTree.py

To run it "python decisionTree.py politicians_train.csv politicians_test.csv 2 pol_2_train.labels pol_2_test.labels pol_2_metrics.txt"

Format is argv[1] is training data
          argv[2] is testing data
          argv[3] is maximumg depth of the tree required
          argv[4] is the labels predicted by decision tree on training data
          argv[5] is the labels predicted by decision tree on testing data
          argv[6] is the error rate for both training and testing data
          
 decisionTree.py also prints out the decision Tree it makes on the training data. So you can visualize what exactly the decisionTree is formed.
   



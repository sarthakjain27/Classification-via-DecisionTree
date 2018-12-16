# DecisionTree
Decision Tree Classifier based on Information Gain, that works with attributes having 2 distinct values

The following code is used to predict the party of a politician and grade of a high school student. 
Greedy policy (Information Gain) is used to build our decision tree. The hyperparameter which decided the depth of the tree is given as an input from command line.

The repo contains two files inspect.py and decisionTree.py

1) Inspect.py:

To run it "python inspect.py #training_file_name# #output_file_name#"
It will calculate the entropy of the label and the error rate working as a majority vote classifier (i.e. a depth 0 decision tree).

2) decisionTree.py

To run it "python decisionTree.py politicians_train.csv politicians_test.csv 2 pol_2_train.labels pol_2_test.labels pol_2_metrics.txt"

Format is argv[1] is training data
          argv[2] is testing data
          argv[3] is maximum depth of the tree required
          argv[4] is the labels predicted by decision tree on training data
          argv[5] is the labels predicted by decision tree on testing data
          argv[6] is the error rate for both training and testing data
          
decisionTree.py also prints out the decision Tree it makes on the training data. So you can visualize what exactly the decisionTree is formed.
   



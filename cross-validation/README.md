Things  Learned:
1. Before implementing any model it is important to know about cross validation
2. Cross Validation is done before building any machine learning model
3. It ensures that our model fits the data set accurately
4. It prevents that data do not over fit
5. There are various cross validation techniques, namely 
     i.  k-fold cross validation
    ii.  stratified k-fold cross validation
    iii. hold out based cross validation 
    iv.  leave-one-out cross validation
    v. group k-fold cross validation
6. Choosing best cross validation depends on dataset. 

In the project we used a Decision Tree Classifiers. As we increased the depth of the decision tree, 
accuracy for taring data increased, but we could not improve accuracy for testing data. 
Form the graph we can conclude that the model over fitted the training data.
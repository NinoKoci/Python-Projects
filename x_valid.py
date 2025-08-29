    #part 2:
    #perform cross-validation to observe the accuracy of dfferent k values
    k_vals = [1, 3, 5, 7, 11, 21, 23] # list of k values to test
    K = 5 # Number of folds for cross-validation
    pred_dic,accuracy_dic = cross_valid(k_vals, K, x_train, y_train) # calculate the predictions and accuracies for each k value
    plt.figure(figsize=(10, 5)) #set the figure size
    plt.plot(k_vals, accuracy_dic.values(), marker='o') #plot the accuracies vs. k values
    plt.xlabel('k_neighbors Value') #label the x and y axis
    plt.ylabel('Accuracy') 
    plt.title('Accuracy vs. k_Number Neighbors') #add the title 
    plt.grid() #add grid lines
    plt.savefig('knn_accuracies.png') # get the png image file of the plot
    
    # perform the accuracy test for the best k value
    best_k = max(accuracy_dic, key=accuracy_dic.get) #get the best performing k value from  cross validation 
    TP, TN, FP, FN = calc_TP_TN_FP_FN(y_train, pred_dic[best_k]) #calculate the true positive, true negative, false positive and false negative counts
    print("Accuracy results for the knn classifier for the best k value:",)
    print("Best k value: ", best_k) #return the best k value
    print("True Positives: ", TP) #print the true positive count
    print("True Negatives: ", TN) #print the true negative count
    print("False Positives: ", FP) #print the false positive count
    print("False Negatives: ", FN) #print the false negative count
    print("Total accuracy of the knn classifier: ", accuracy(y_train, pred_dic[best_k])) #print the total accuracy of the knn classifier's best performing k value
    
    #part 3:
    #initiaze the agglomerative clustering model
    model = AgglomerativeClustering(n_clusters=2, linkage='average') #set up the model with the required parameters
    labels = model.fit_predict(x_train) #get the binary labels from the clustering 
    assigned_labels = cluster_labeler(labels) #use 0 and 1 to denote CurrentSmoker and NeverSmoker respectively
    
    #perform the accuracy test for the agglomerative clustering
    TP, TN, FP, FN = calc_TP_TN_FP_FN(y_train, assigned_labels) #calculate the true positive, true negative, false positive and false negative counts
    print("Accuracy results for the knn classifier for Agglomerative Clustering:",)
    print("True Positives: ", TP) #print the true positive count
    print("True Negatives: ", TN) #print the true negative count
    print("False Positives: ", FP) #print the false positive count
    print("False Negatives: ", FN) #print the false negative count
    print("Accuracy of Agglomerative Clustering: ", accuracy(assigned_labels, y_train)) #print the accuracy of the clustering results
    
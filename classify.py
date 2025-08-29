#Define the KNN class
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import AgglomerativeClustering


# Load the datasets
def train_set(train_set): 
    x_train = np.array(train_set.iloc[:-1,:].T) #take the columns and transpose them to rows
    y_train = np.array(train_set.iloc[-1]) #return only the last row
    return x_train.astype(float), y_train #ensure that the numbers are returneds as floats

def test_set(test_set):
    x_test = np.array(test_set.iloc[:-1,:].T)
    return x_test.astype(float)


class KNNClassifier:
    def __init__(self, k, x_train, y_train):
        self.k = k
        self.x_train = x_train
        self.y_train = y_train
        
    def euclidean_distance(self, x1, x2): #define Euclidian distance function
                return np.sqrt(np.sum((x1 - x2) ** 2)) # calculate the Euclidean distance between two points
    
    def k_neighbors(self, distances): #find the k nearest neighbors
        neighbors = sorted(distances)[:self.k] #sort the distances from longest to shortestm and return the k nearest neighbors(the top k distances)
        return neighbors
    
    def tie_break(self, neighbors): #break the tie in case of equal votes
        labels =  [label for _, label in neighbors] #get the class labels of the nweeighbors
        votes = Counter(labels).most_common(1)[0][0] #get the most common class label
        return votes
    

    def predict(self, x_test):  # predict the  labels for the test set
        y_pred = []  # list to store the predictions
        for x in x_test:
            distances = [] #list for storing the distances in the form of tuples (distance, label)
            for i in range(len(self.x_train)):
                distance = self.euclidean_distance(x, self.x_train[i])  #get the euclidian distances for each point in the test set for all the points in the training set
                distances.append((distance, self.y_train[i])) #append the distaes to the list 
            neighbors = self.k_neighbors(distances) #return the k number of the greatest distance/s 
            votes = self.tie_break(neighbors)  # get the most common label for that pateient in the training set
            y_pred.append(votes)  #append the most common label to the list of predictions
        return np.array(y_pred)  #return the predictions for the test set
    
    
def accuracy(y_true, y_pred): #calculate the accuracy of the predictions
    correct = 0 #initialize the true positive and false positive counts
    total_count = len(y_true) #total number of predicions 
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    accuracy = correct / total_count #calculate the accuracy of y_predicted
    return accuracy 


def cross_valid(k_vals, K, x_train, y_train):
    folds = np.array_split(x_train, K) #split the training set into 5 folds
    y_train_split = np.array_split(y_train, K) #split the labels into 5 folds
    pred_dic = {} #initialize the dictionary to store the predictions for each k value
    accuracy_dic = {} #initialize the dictionary to store the accuracies for each k value
    for k in k_vals:
        pred_list = []
        acc_list = []
        for i in range(K): #for each fold
            fold = folds[i] #get the current fold
            x_train_conc = np.concatenate((folds[:i] + folds[i+1:])) #concatenate the other folds to get the rest of the training set
            y_train_conc = np.concatenate((y_train_split[:i] + y_train_split[i+1:])) #concatenate the other folds to get the rest of the labels
            knn_classifier = KNNClassifier(k, x_train_conc, y_train_conc) #create an instance of the KNNClassifier class with the rest of the training set
            y_pred_by_k = knn_classifier.predict(fold)
            pred_list.append(y_pred_by_k) #append the predictions for the current fold to the list
            acc_list.append(accuracy(y_train_split[i], y_pred_by_k))#append the accuracy for the current fold to the list
        pred_dic[k] = np.concatenate(pred_list) #concatate the predictions per k then store them in the dictionary
        accuracy_dic[k] = np.mean(acc_list) #calculate the mean accuracy for each k value and store it in the dictionary
    return pred_dic, accuracy_dic #return the predictions and accuracies dictionaries for each k value

def calc_TP_TN_FP_FN(y_true, y_pred): #calculate the true positive, true negative, false positive and false negative counts
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    i = 0
    while i < len(y_true):
        if y_true[i] == 'CurrentSmoker' and y_pred[i] == 'CurrentSmoker':
            TP += 1
        elif y_true[i] == 'NeverSmoker' and y_pred[i] == 'NeverSmoker':
            TN += 1
        elif y_true[i] == 'NeverSmoker' and y_pred[i] == 'CurrentSmoker':
            FP += 1
        elif y_true[i] == 'CurrentSmoker' and y_pred[i] == 'NeverSmoker':
            FN += 1
        i += 1
    return TP, TN, FP, FN 


def cluster_labeler(labels): #label the clusters as CurrentSmoker and NeverSmoker based on the clustering results
    return ['CurrentSmoker' if label == 1 else 'NeverSmoker' for label in labels]

    
def main():
    # part 1:
    # predict the labels for the test set using the knn classifier
    train_data = pd.read_table(sys.argv[1]) # Load the training data from the first command line argument
    test_data = pd.read_table(sys.argv[2])
    x_train, y_train = train_set(train_data) # Split the training set into features and labels
    x_test = test_set(test_data) # Split the test set into features
    k = int(sys.argv[3]) # Set the number of neighbors
    knn_classifier = KNNClassifier(k, x_train, y_train) # Create an instance of the KNNClassifier class
    y_pred  = knn_classifier.predict(x_test) # Predict the class labels for the test set
    test_data.iloc[-1] = y_pred #replace the last row of the test set with the predicted labels
    print("Predicted labels for the test set:")
    print(test_data.iloc[-1])
    
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
    

        
if __name__ == '__main__': #run the main function in comand console
    main()
    

                
   

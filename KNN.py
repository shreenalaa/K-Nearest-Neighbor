import numpy as np 
import pandas as pd 
import os 
os.system("cls")

class KNN_classifier :   # for classification
    def _init_ (self ,k=3): # k=3 as default value
        self.k =k
        self.x_train = None
        self.y_train = None

    def distance (self , x1 ,x2): #x1 : point 1 , x2 : point 2 
        return np.sqrt(np.sum((x2-x1)**2))
    def fit (self,x,y):
        self.x_train = x
        self.y_train = y

    def predict (self ,x_new): # x : is a vector of featuers ... (list)
        predictions =[]
        for x in x_new:
            distances = [self.euclidean_distance(x,i) for i in self.X_train]
            k_indices = np.argsort(distances)[:self.k] # need only first k values 
            k_labels = [self.y_train[i] for i in k_indices] # determine class 
            common = np.bincount(k_labels).argmax() # get maximum repeated class 
            predictions.append(common) # append prediction for each point
        return np.array(predictions)
    

class KNNRegressor: # for regression
    def __init__ (self, k = 3):
        self.k = k
        self.X_train = None
        self .y_train = None

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum( (x2 - x1) ** 2))
    
    def fit (self,X,y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_new):
        predictions = []
        for x in X_new:
            distances = [self.euclidean_distance(x,i) for i in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_values = [self.y_train[i] for i in k_indices]
            result = np.mean(k_values)
            predictions.append(result)
        return np.array(predictions)



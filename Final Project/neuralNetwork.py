import util
import classificationMethod
import numpy as np
import random

class NeuralNetworkClassifier(classificationMethod.ClassificationMethod):
    def __init__(self, legalLabels):

        self.legalLabels = legalLabels
        self.type = "neuralnetwork"
        

        # size of first hidden layer
        self.hidden1_size = 100
        # size of second hidden layer    
        self.hidden2_size = 100
        # learning rate
        self.learning_rate = 0.005
        # number of training epochs  
        self.epochs = 50 
        # size of mini batches 
        self.batch_size = 100
        # Lambda strength (helps overfitting)
        self.l2_lambda = 0.001     
        
        # layer weights
        self.W1 = None  
        self.W2 = None  
        self.W3 = None  

        # layer biases
        self.b1 = None  
        self.b2 = None  
        self.b3 = None  


    def forward(self, X):
        # first hidden layer
        z1 = self.b1+np.dot(X, self.W1)  # a weighted sum of inputs plus a bias
        a1 = np.maximum(z1,0)                # introduce curvature
        
        # second hidden layer
        z2 = self.b2+np.dot(a1, self.W2)
        a2 = np.maximum(z2,0)       
        
        # output layer
        z3 = self.b3+np.dot(a2, self.W3)
        exp_x = np.exp(z3 - np.max(z3, axis=1, keepdims=True))
        a3 =  exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        return z1, a1, z2, a2, a3

    def backward(self, X, y, z1, a1, z2, a2, a3):

        # hidden layer errors
        d3 = a3 - y  
        d2 = np.dot(d3, self.W3.T) * (z2 > 0).astype(float)
        d1 = np.dot(d2, self.W2.T) * (z1 > 0).astype(float)
        
        # helps with overfitting
        batch_size = X.shape[0]
        temp1 = np.dot(X.T, d1)/ batch_size
        temp2 = self.l2_lambda * self.W1
        dW1 = temp1+temp2
        db1 = np.sum(d1, axis=0) / batch_size

        temp1 = np.dot(a1.T, d2) / batch_size
        temp2 = self.l2_lambda * self.W2
        dW2 = temp1+temp2
        db2 = np.sum(d2, axis=0) / batch_size

        temp1 = np.dot(a2.T, d3) / batch_size
        temp2 = self.l2_lambda * self.W3
        dW3 = temp1+temp2
        db3 = np.sum(d3, axis=0) / batch_size
    

        return dW1, db1, dW2, db2, dW3, db3

    def update_weights(self, dW1, db1, dW2, db2, dW3, db3):
        # Update weights and biases 
        self.W1 = self.W1-self.learning_rate * dW1
        self.b1 =self.b1-self.learning_rate * db1
        self.W2 =self.W2-self.learning_rate * dW2
        self.b2 =self.b2-self.learning_rate * db2
        self.W3 = self.W3-self.learning_rate * dW3
        self.b3 = self.b3-self.learning_rate * db3

    def convert1(self, labels):
        #converts to a encoded matrix size samples x classes
        class_size = len(self.legalLabels)
        sample_size = len(labels)
        onh = np.zeros((sample_size, class_size))
        for i, j in enumerate(labels):
            onh[i, self.legalLabels.index(j)] = 1

        return onh

    def convert2(self, features_list):
        feature_size = len(self.features)
        sample_size = len(features_list)
        X = np.zeros((sample_size, feature_size))
        for i, j in enumerate(features_list):
            for k, l in enumerate(self.features):
                X[i, k] = j.get(l, 0)
        return X

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        # unique feature names
        unique = set([f for datum in trainingData for f in datum.keys()])
        self.features = list(unique)
        
        # map labels and indices
        self.label_to_idx = {}
        self.idx_to_label = {}
        for i, label in enumerate(self.legalLabels):
            self.label_to_idx[label] = i
            self.idx_to_label[i] = label
        
        # data to matrices
        X_train = self.convert2(trainingData)
        X_val = self.convert2(validationData)
        y_train = self.convert1(trainingLabels)
        y_val = self.convert1(validationLabels)
        
        #Network dimensions
        output_size = len(self.legalLabels)
        input_size = len(self.features)
        
        # initialize weights
        temp1 = np.random.randn(input_size, self.hidden1_size)
        temp2 = np.sqrt(2 / input_size)
        self.W1 = temp1 * temp2
        self.b1 = np.zeros((1, self.hidden1_size))

        temp1 = np.random.randn(self.hidden1_size, self.hidden2_size)
        temp2 = np.sqrt(2 / self.hidden1_size)
        self.W2 = temp1 * temp2
        self.b2 = np.zeros((1, self.hidden2_size))

        temp1 = np.random.randn(self.hidden2_size, output_size)
        temp2 = np.sqrt(2 / self.hidden2_size)
        self.W3 = temp1 * temp2
        self.b3 = np.zeros((1, output_size))
        
        # training 
        samples = X_train.shape[0]
        for epoch in range(self.epochs):
            # shuffle data
            indices = np.random.permutation(samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            for i in range(0, samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                z1, a1, z2, a2, a3 = self.forward(X_batch)
                dW1, db1, dW2, db2, dW3, db3 = self.backward(X_batch, y_batch, z1, a1, z2, a2, a3)
                self.update_weights(dW1, db1, dW2, db2, dW3, db3)
            
            # validating
            if validationData and epoch % 5 == 0:
                i, j, k, l, val_output = self.forward(X_val)
                val_predictions = np.argmax(val_output, axis=1)
                val_actual = np.argmax(y_val, axis=1)
                val_accuracy = np.mean(val_predictions == val_actual)
                print(f"Epoch {epoch}: validation accuracy = {val_accuracy}")

    def classify(self, testData):

        X_test = self.convert2(testData)
        i, j, k, l, output = self.forward(X_test)
        predictions = np.argmax(output, axis=1)
        guesses=[]
        for i in predictions:
            guesses.append(self.legalLabels[i])

        return guesses
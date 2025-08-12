import util
import classificationMethod
import numpy as np
import torch
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
from torch.utils.data import DataLoader, TensorDataset  # For batch processing

class NeuralNetworkClassifierPytorch(classificationMethod.ClassificationMethod):
    def __init__(self, legalLabels):

        self.legalLabels = legalLabels
        self.type = "neuralNetwork_pytorch"
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
    
    def create_model(self, input_size, output_size):
        model = nn.Sequential(
            # first hidden layer
            nn.Linear(input_size, self.hidden1_size),
            nn.ReLU(), 
            
            # second hidden layer
            nn.Linear(self.hidden1_size, self.hidden2_size),
            nn.ReLU(),
            
            # output layer
            nn.Linear(self.hidden2_size, output_size)
        )
        
        return model
    
    def convert_to_tensor(self, features_list):
        feature_size = len(self.features)
        sample_size = len(features_list)
        X = np.zeros((sample_size, feature_size))
        for i, j in enumerate(features_list):
            for k, l in enumerate(self.features):
                X[i, k] = j.get(l, 0)
                
        # convert to PyTorch tensor
        return torch.FloatTensor(X)
    
    def convert_labels(self, labels):

        indices = []
        for label in labels:
            index = self.legalLabels.index(label)
            indices.append(index)

        #return idices for each label
        return torch.LongTensor(indices)
    
    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Train the neural network model on the provided data.
        
        Args:
            trainingData: List of training samples (each sample is a feature dictionary)
            trainingLabels: List of corresponding training labels
            validationData: Optional validation data
            validationLabels: Optional validation labels
        """
        # unique feature names
        unique = set([f for datum in trainingData for f in datum.keys()])
        self.features = list(unique)
        
        # map labels and indices
        self.label_to_idx = {}
        self.idx_to_label = {}
        for i, label in enumerate(self.legalLabels):
            self.label_to_idx[label] = i
            self.idx_to_label[i] = label
        
        # convert pytorch tensors
        X_train = self.convert_to_tensor(trainingData)
        y_train = self.convert_labels(trainingLabels)

        X_val = None
        y_val = None
        if validationData:
             X_val = self.convert_to_tensor(validationData)
        if validationLabels:
            y_val = self.convert_labels(validationLabels)
        
        # create the neural network 
        input_size = len(self.features)
        output_size = len(self.legalLabels)
        self.model = self.create_model(input_size, output_size)
        
        # setup loss function and optimizer
        criterion = nn.CrossEntropyLoss()  
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_lambda)  
        
        # creating dataloader for batch 
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True) 
        
        # training loop
        for epoch in range(self.epochs):
            self.model.train()  
            
            # process data in batches
            for i, j in train_loader:
                optimizer.zero_grad()
                
                # forward pass: compute predictions
                outputs = self.model(i)
                
                # compute loss
                loss = criterion(outputs, j)
                
                # backward pass: compute gradients
                loss.backward()

                # updating weights
                optimizer.step()
                
            # validation
            if validationData and epoch % 5 == 0:
                self.model.eval() 
                with torch.no_grad():  
                    # compute predictions
                    val_outputs = self.model(X_val)
                    i, predicted = torch.max(val_outputs, 1)
                    
                    # accuracy
                    correct = (predicted == y_val).sum().item()
                    val_accuracy = correct / len(y_val)
                    
                    print(f"Epoch {epoch}: validation accuracy = {val_accuracy:.4f}")
    
    def classify(self, testData):
        # if not self.model:
        #     raise RuntimeError("Model must be trained before classification!")
            
        #test data to tensor
        X_test = self.convert_to_tensor(testData)
        
        self.model.eval()
        with torch.no_grad():
            # model predictions
            outputs = self.model(X_test)
            i, predictions = torch.max(outputs, 1) 
        
        # convert  indices to original labels
        guesses=[]
        for i in predictions:
            guesses.append(self.legalLabels[i.item()])

        return guesses

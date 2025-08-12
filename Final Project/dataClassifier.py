# dataClassifier.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# This file contains feature extraction methods and harness 
# code for data classification

# # import mostFrequent
# import naiveBayes
import perceptron
import neuralNetwork
import neuralNetwork_pytorch  # Import the PyTorch implementation
# import math
import samples
from random import sample
import sys
import util
import time
import numpy as np

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH = 28
DIGIT_DATUM_HEIGHT = 28
FACE_DATUM_WIDTH = 60
FACE_DATUM_HEIGHT = 70


def basicFeatureExtractorDigit(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is white (0) or gray/black (1)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(DIGIT_DATUM_WIDTH):
    for y in range(DIGIT_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def basicFeatureExtractorFace(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is an edge (1) or no edge (0)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(FACE_DATUM_WIDTH):
    for y in range(FACE_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def enhancedFeatureExtractorDigit(datum):
  """
  Your feature extraction playground.
  
  You should return a util.Counter() of features
  for this datum (datum is of type samples.Datum).
  
  ## DESCRIBE YOUR ENHANCED FEATURES HERE...
  
  ##
  """
  features =  basicFeatureExtractorDigit(datum)

  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()

  return features


def basicFeatureExtractorPacman(state):
    """
    A basic feature extraction function.

    You should return a util.Counter() of features
    for each (state, action) pair along with a list of the legal actions

    ##
    """
    features = util.Counter()
    for action in state.getLegalActions():
        successor = state.generateSuccessor(0, action)
        foodCount = successor.getFood().count()
        featureCounter = util.Counter()
        featureCounter['foodCount'] = foodCount
        features[action] = featureCounter
    return features, state.getLegalActions()

def enhancedFeatureExtractorPacman(state):
    """
    Your feature extraction playground.

    You should return a util.Counter() of features
    for each (state, action) pair along with a list of the legal actions

    ##
    """

    features = basicFeatureExtractorPacman(state)[0]
    for action in state.getLegalActions():
        features[action] = util.Counter(features[action], **enhancedPacmanFeatures(state, action))
    return features, state.getLegalActions()

def enhancedPacmanFeatures(state, action):
    """
    For each state, this function is called with each legal action.
    It should return a counter with { <feature name> : <feature value>, ... }
    """
    features = util.Counter()
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
    return features


def contestFeatureExtractorDigit(datum):
  """
  Specify features to use for the minicontest
  """
  features = basicFeatureExtractorDigit(datum)
  return features

def enhancedFeatureExtractorFace(datum):
  """
  Your feature extraction playground for faces.
  It is your choice to modify this.
  """
  features = basicFeatureExtractorFace(datum)
  return features

def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
  """
  This function is called after learning.
  Include any code that you want here to help you analyze your results.
  
  Use the printImage(<list of pixels>) function to visualize features.
  
  An example of use has been given to you.
  
  - classifier is the trained classifier
  - guesses is the list of labels predicted by your classifier on the test set
  - testLabels is the list of true labels
  - testData is the list of training datapoints (as util.Counter of features)
  - rawTestData is the list of training datapoints (as samples.Datum)
  - printImage is a method to visualize the features 
  (see its use in the odds ratio part in runClassifier method)
  
  This code won't be evaluated. It is for your own optional use
  (and you can modify the signature if you want).
  """
  
  # Put any code here...
  # Example of use:
  # for i in range(len(guesses)):
  #     prediction = guesses[i]
  #     truth = testLabels[i]
  #     if (prediction != truth):
  #         print "==================================="
  #         print "Mistake on example %d" % i
  #         print "Predicted %d; truth is %d" % (prediction, truth)
  #         print "Image: "
  #         print rawTestData[i]
  #         break
  for i in range(len(guesses)):
      prediction = guesses[i]
      truth = testLabels[i]
      if (prediction == truth):
          print("===================================")
          print(f"True with example {i}")
          print(f"Predicted {prediction}; truth is {truth}")
          break

## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
      self.width = width
      self.height = height

    def printImage(self, pixels):
      """
      Prints a Datum object that contains all pixels in the 
      provided list of pixels.  This will serve as a helper function
      to the analysis function you write.
      
      Pixels should take the form 
      [(2,2), (2, 3), ...] 
      where each tuple represents a pixel.
      """
      image = samples.Datum(None, self.width, self.height)
      for pix in pixels:
        try:
            # This is so that new features that you could define which 
            # which are not of the form of (x,y) will not break
            # this image printer...
            x, y = pix
            image.pixels[x][y] = 2
        except:
            print("new features:", pix)
            continue
      print(image)  

def default(str):
  return str + ' [Default: %default]'

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """


def readCommand(argv):
  "Processes the command used to run from the command line."
  from optparse import OptionParser  
  parser = OptionParser(USAGE_STRING)
  # python dataClassifier.py -c perceptron -d faces
  # python dataClassifier.py -c neuralNetwork -d faces
  # python dataClassifier.py -c neuralNetwork_pytorch -d faces
  parser.add_option('-c', '--classifier', help=default('The type of classifier'), 
                   choices=['perceptron', 'neuralNetwork', 'neuralNetwork_pytorch'], 
                   default='neuralNetwork_pytorch')
  parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces'], default='faces')
  parser.add_option('-t', '--training', help=default('The size of the training set'), default=450, type="int")
  parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False, action="store_true")
  parser.add_option('-o', '--odds', help=default('Whether to compute odds ratios'), default=False, action="store_true")
  parser.add_option('-1', '--label1', help=default("First label in an odds ratio comparison"), default=0, type="int")
  parser.add_option('-2', '--label2', help=default("Second label in an odds ratio comparison"), default=1, type="int")
  parser.add_option('-w', '--weights', help=default('Whether to print weights'), default=False, action="store_true")
  parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"), type="float", default=2.0)
  parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False, action="store_true")
  parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
  parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")
  parser.add_option('-g', '--agentToClone', help=default("Pacman agent to copy"), default=None, type="str")
  parser.add_option('--hidden1', help=default("Size of first hidden layer in neural network"), default=100, type="int")
  parser.add_option('--hidden2', help=default("Size of second hidden layer in neural network"), default=100, type="int")
  parser.add_option('--learningrate', help=default("Learning rate for neural network"), default=0.005, type="float")
  parser.add_option('--epochs', help=default("Number of epochs for neural network training"), default=50, type="int")
  parser.add_option('--batchsize', help=default("Batch size for neural network training"), default=100, type="int")
  parser.add_option('--l2lambda', help=default("L2 regularization strength for neural network"), default=0.001, type="float")
  
  options, otherjunk = parser.parse_args(argv)
  if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
  args = {}
  
  # Set up variables according to the command line input.
  print("Doing classification")
  print("--------------------")
  print(f"data:\t\t{options.data}")
  print(f"classifier:\t\t{options.classifier}")
  if(options.data == "digits"):
    printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
    if (options.features):
      featureFunction = enhancedFeatureExtractorDigit
    else:
      featureFunction = basicFeatureExtractorDigit
  elif(options.data == "faces"):
    printImage = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
    if (options.features):
      featureFunction = enhancedFeatureExtractorFace
    else:
      featureFunction = basicFeatureExtractorFace      
  else:
    print("Unknown dataset", options.data)
    print(USAGE_STRING)
    sys.exit(2)
    
  if(options.data == "digits"):
    legalLabels = list(range(10))
  else:
    legalLabels = list(range(2))
    
  if options.training <= 0:
    print(f"Training set size should be a positive integer (you provided: {options.training})")
    print(USAGE_STRING)
    sys.exit(2)
    
  if options.smoothing <= 0:
    print(f"Please provide a positive number for smoothing (you provided: {options.smoothing})")
    print(USAGE_STRING)
    sys.exit(2)
    
  if options.odds:
    if options.label1 not in legalLabels or options.label2 not in legalLabels:
      print(f"Didn't provide a legal labels for the odds ratio: ({options.label1},{options.label2})")
      print(USAGE_STRING)
      sys.exit(2)

  if(options.classifier == "perceptron"):
    classifier = perceptron.PerceptronClassifier(legalLabels, options.iterations)
  elif(options.classifier == 'neuralNetwork'):
    classifier = neuralNetwork.NeuralNetworkClassifier(legalLabels)
    classifier.hidden1_size = options.hidden1
    classifier.hidden2_size = options.hidden2
    classifier.learning_rate = options.learningrate
    classifier.epochs = options.epochs
    
    print(f"Neural Network configuration:")
    print(f"- First hidden layer size: {classifier.hidden1_size}")
    print(f"- Second hidden layer size: {classifier.hidden2_size}")
    print(f"- Learning rate: {classifier.learning_rate}")
    print(f"- Training epochs: {classifier.epochs}")
  elif(options.classifier == 'neuralNetwork_pytorch'):
    # Use the PyTorch neural network implementation
    classifier = neuralNetwork_pytorch.NeuralNetworkClassifierPytorch(legalLabels)
    # Set hyperparameters from command line options
    classifier.hidden1_size = options.hidden1
    classifier.hidden2_size = options.hidden2
    classifier.learning_rate = options.learningrate
    classifier.epochs = options.epochs
    classifier.batch_size = options.batchsize
    classifier.l2_lambda = options.l2lambda
    
    print(f"PyTorch Neural Network configuration:")
    print(f"- First hidden layer size: {classifier.hidden1_size}")
    print(f"- Second hidden layer size: {classifier.hidden2_size}")
    print(f"- Learning rate: {classifier.learning_rate}")
    print(f"- Training epochs: {classifier.epochs}")
    print(f"- Batch size: {classifier.batch_size}")
    print(f"- L2 regularization: {classifier.l2_lambda}")
  else:
    print(USAGE_STRING)
    print("Unknown classifier:", options.classifier)
    
    sys.exit(2)
  
  args['agentToClone'] = options.agentToClone

  args['classifier'] = classifier
  args['featureFunction'] = featureFunction
  args['printImage'] = printImage
  
  return args, options

# Dictionary containing full path to .pkl file that contains the agent's training, validation, and testing data.
MAP_AGENT_TO_PATH_OF_SAVED_GAMES = {
    'FoodAgent': ('pacmandata/food_training.pkl','pacmandata/food_validation.pkl','pacmandata/food_test.pkl' ),
    'StopAgent': ('pacmandata/stop_training.pkl','pacmandata/stop_validation.pkl','pacmandata/stop_test.pkl' ),
    'SuicideAgent': ('pacmandata/suicide_training.pkl','pacmandata/suicide_validation.pkl','pacmandata/suicide_test.pkl' ),
    'GoodReflexAgent': ('pacmandata/good_reflex_training.pkl','pacmandata/good_reflex_validation.pkl','pacmandata/good_reflex_test.pkl' ),
    'ContestAgent': ('pacmandata/contest_training.pkl','pacmandata/contest_validation.pkl', 'pacmandata/contest_test.pkl' )
}
# Main harness code

def runClassifier(args, options):
    featureFunction = args['featureFunction']
    classifier = args['classifier']
    printImage = args['printImage']

   # Load data
    if options.data == "faces":
        rawAllTrainingData = samples.loadDataFile("facedata/facedatatrain", 500, FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
        allTrainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", 500)
        rawTestData = samples.loadDataFile("facedata/facedatatest", options.test, FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", options.test)
    else:
        rawAllTrainingData = samples.loadDataFile("digitdata/trainingimages", 5000, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
        allTrainingLabels = samples.loadLabelsFile("digitdata/traininglabels", 5000)
        rawTestData = samples.loadDataFile("digitdata/testimages", options.test, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("digitdata/testlabels", options.test)

    # Extract features for all data
    allTrainingData = list(map(featureFunction, rawAllTrainingData))
    testData = list(map(featureFunction, rawTestData))

    # Define percentages and iterations
    percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    results = {}

    for percent in percentages:
        numTraining = int(len(allTrainingData) * percent / 100)
        accuracies = []
        training_times = []

        for i in range(5):
            # randoml sample data
            indices = list(range(len(allTrainingData)))
            sampledIndices = sample(indices, numTraining)
            
            trainingData = []
            for j in sampledIndices:
              trainingData.append(allTrainingData[j])
            
            trainingLabels = []
            for k in sampledIndices:
              trainingLabels.append(allTrainingLabels[k])

            #time it takes to run
            start_time = time.time()
            classifier.train(trainingData, trainingLabels, [], [])
            training_time = time.time() - start_time
            training_times.append(training_time)

            # calculate accuracy  and prediction error
            guesses = classifier.classify(testData)
            correct = sum(guesses[i] == testLabels[i] for i in range(len(testLabels)))
            accuracy = 100.0 * correct / len(testLabels)
            accuracies.append(accuracy)

            print(f"Iteration {i+1} for {percent}%: Accuracy = {accuracy:.2f}%, Prediction Error = {100-accuracy:.2f}%, Time = {training_time:.2f}s")

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        mean_time = np.mean(training_times)
        results[percent] = {
            'training_size': numTraining,
            'mean_accuracy': mean_acc,
            'mean_pred_error':100-mean_acc,
            'std_accuracy': std_acc,
            'mean_time': mean_time
        }

    print("\n=== Results ===")
    print("Percentage | Training Size | Mean Accuracy (%) | Mean Prediction Error (%) | Std Dev (%) | Mean Time (s)")
    for percent, stats in results.items():
        print(
            f"{percent:>9}% | {stats['training_size']:>13} | {stats['mean_accuracy']:>16.2f}% | {stats['mean_pred_error']:>24.2f}% |"
            f"{stats['std_accuracy']:>11.2f}% | {stats['mean_time']:>11.2f}"
        )
        
if __name__ == '__main__':
    # Read input
    args, options = readCommand( sys.argv[1:] )
    # Run classifier
    runClassifier(args, options)

  

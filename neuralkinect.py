#!/usr/bin/env python
''' 
    @title ->          neuralkinect.py
    @author ->         Michael Kepple
    @date ->           15 Dec 2013
    @description ->    neuralkinect.py -> Achieves Kinect gesture recognition
                       through training a neural network for classification.
    @note ->           Utilizes SciPy && PyBrain libraries:
                       http://www.scipy.org/install.html
                       https://github.com/pybrain/pybrain
    @python_version -> Anaconda 32-bit (2.7)
    @usage ->          python neuralkinect.py
'''
import time
import csv
import glob
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.structure import TanhLayer
EPOCHS = 6000

class NeuralKinect():
    def __init__(self):
        # Softmax layer -> great for classification networks
        #self.neuralNet = buildNetwork(60, 60, 5, outclass=SoftmaxLayer)
        #self.neuralNet = buildNetwork(60, 60, 5, hiddenclass=TanhLayer)
        #self.neuralNet = buildNetwork(60, 60, 5, bias=True)
        self.neuralNet = buildNetwork(60, 60, 5)
        self.dataSet = SupervisedDataSet(60, 5)

    def trainBackProp(self):
        trainer = BackpropTrainer(self.neuralNet, self.dataSet)
        start = time.time()
        trainer.trainEpochs(EPOCHS)
        end = time.time()
        print("Training time -> " + repr(end-start))
        print(repr(trainer.train()))

    def loadDataSet(self):
        points = []
        for csvFile in glob.iglob("TrainData/*.csv"):
            with open(csvFile, 'rt') as letterSet:
                reader = csv.reader(letterSet)
                header = str(reader.next())
                letter = header[2:3]
                targetStr = header[4:9]
                print("Processing Dataset for letter -> " + letter)
                target = []
                for digit in targetStr:
                    target.append(digit)
                rows = 1
                for row in reader:              
                    for col in row:
                        points.append(col)
                    if rows % 20 == 0:
                        self.dataSet.addSample(points, target)
                        points = []
                    rows += 1
                    
    def processResults(self, output):
        result = ""
        for digit in output:
            if digit > 0.5:
                result += "1"
            else:
                result += "0"
        print("Network result -> " + chr(64+int(result,2)))
                    
    def testNetwork(self):
        points = []
        for csvFile in glob.iglob("TestData/*.csv"):
            with open(csvFile, 'rt') as testPose:
                reader = csv.reader(testPose)
                rows = 1
                for row in reader:
                    for col in row:
                        points.append(col)
                    if rows % 20 == 0:
                        self.processResults(self.neuralNet.activate(points))
                        points = []
                    rows += 1

def main():
    nk = NeuralKinect()
    nk.loadDataSet()
    nk.trainBackProp()
    nk.testNetwork()
    
if __name__ == '__main__':
    main()

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
    @python_version -> Anaconda (2.7)
    @usage ->          python neuralkinect.py
'''
import csv
import glob
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

class NeuralKinect():
    def __init__(self):
        # Softmax layer -> great for classification networks
        self.neuralNet = buildNetwork(60, 20, 5, outclass=SoftmaxLayer)
        self.dataSet = SupervisedDataSet(60, 5)

    def trainBackProp(self):
        trainer = BackpropTrainer(self.neuralNet, self.dataSet)
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

def main():
    nk = NeuralKinect()
    nk.loadDataSet()
    nk.trainBackProp()
    
if __name__ == '__main__':
    main()

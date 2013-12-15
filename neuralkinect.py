'''
Created on Dec 14, 2013

@author: Kep
'''
import csv
import glob
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet

class NeuralKinect():
    
    def __init__(self):
        self.neuralNet = buildNetwork(60, 20, 5)
        self.dataSet = SupervisedDataSet(60, 5)
        
    def initWeights(self):
        pass
    
    def trainNetwork(self):
        points = []
        for csvFile in glob.iglob("TrainData/*.csv"):
            with open(csvFile, 'rt') as letterSet:
                reader = csv.reader(letterSet)
                header = str(reader.next())
                letter = header[2:3]
                target = header[4:9]
                print("Training DataSet: " + letter)
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
    nk.trainNetwork()
        
if __name__ == '__main__':
    main()

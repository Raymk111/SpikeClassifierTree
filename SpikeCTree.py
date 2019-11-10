import operator
from math import log
import pandas as pd
import pydot



class DTreeNode:

    def __init__(self, xtrain = pd.DataFrame(), attributeToTest = "", thresholdValue = 0, isleaf = False, entries = pd.DataFrame(), infoGain = 0.0):
        #Node can apply a threshold on the entry or classify (leaf)
        #data needed in the case of a "node"
        if(not isleaf):
            #data needed for building and setting entries for majority class in leaf cases
            self.leftX = pd.DataFrame(columns=xtrain.columns)
            self.leftY = []
            self.rightX = pd.DataFrame(columns=xtrain.columns)
            self.rightY = []

            #left and rigth child nodes themselves
            self.left = None
            self.right = None

            #threshold to apply and data needed for drawing the tree later for analysis
            self.attributeToTest = attributeToTest
            self.thresholdValue = thresholdValue
            self.infoGain = infoGain

        #data needed for leaf
        self.isLeaf = isleaf
        self.entries = entries
        self.label = ""
        best = 0
        i = 0
        if(isleaf):
            self.ytrainDist = countTargetInstances(entries)
            for case in self.ytrainDist:
                if case > best:
                    best = case
                    self.label = i
                i += 1

    #apply predefined threshold values of node to xtrain passed in
    def splitBranchesCalcNodeValues(self, xtrain, ytrain):
        if(len(xtrain) != len(ytrain) or len(xtrain) == 0):
            return

        xtrainList = list(zip(xtrain[self.attributeToTest], ytrain))
        index = 0
        xtrainArr = xtrain.values
        for case, target in xtrainList:
            if case <= self.thresholdValue:
                self.leftX.loc[len(self.leftX), :] = xtrainArr[index]
                self.leftY.append(target)
            else:
                self.rightX.loc[len(self.leftX), :] = xtrainArr[index]
                self.rightY.append(target)
            index += 1

    #recur on each node of tree until reaching a leaf return label defined
    def predict(self, entry):
        if(not self.isLeaf and len(self.attributeToTest) > 0):
            if(entry[self.attributeToTest] <= self.thresholdValue):
                return self.left.predict(entry)
            else:
                return self.right.predict(entry)

        return self.label

class SpikeCTree:

    def __init__(self, depthLimit = 0, balancerSelected = ""):
        self.depthLimit = depthLimit
        self.root = DTreeNode()

        #rules to apply for information gain / where to split split calculation
        self.balancerSelected = balancerSelected

    #same api as decision tree classifier
    def fit(self, x, y):
        self.root = self.buildtree(x, y)
        return

    #contained recursive method returning root node once all leaves have been defined
    def buildtree(self, xtrain, ytrain, depthCount = 0):
        attrstotest = xtrain.keys()

        infogain = {}
        threshold = {}
        bestInfoGainAttrName = ""
        infogain[bestInfoGainAttrName] = 0
        for attribute in attrstotest:
            infogainattr, thresholdattr = self.getInfoGain(xtrain[attribute], ytrain)
            infogain[attribute] = infogainattr
            threshold[attribute] = thresholdattr
            bestInfoGainAttrName = attribute if infogain[attribute] > infogain[bestInfoGainAttrName] else bestInfoGainAttrName

        if(len(bestInfoGainAttrName) == 0):
            return self.makeLeaf(xtrain, ytrain)

        root = DTreeNode(xtrain, bestInfoGainAttrName, threshold[bestInfoGainAttrName], infoGain=infogain[bestInfoGainAttrName])
        root.splitBranchesCalcNodeValues(xtrain.copy(), ytrain)

        depthCount += 1
        if(self.shouldMakeLeaf(root.leftY, xtrain, depthCount)):
            if(len(root.leftY)):
                root.left = self.makeLeaf(root.leftX, root.leftY)
            else:
                root.left = self.makeLeaf(xtrain, ytrain)
        else:
            root.left = self.buildtree(root.leftX, root.leftY, depthCount)

        if(self.shouldMakeLeaf(root.rightY, xtrain, depthCount)):
            if (len(root.rightY)):
                root.right = self.makeLeaf(root.rightX, root.rightY)
            else:
                root.right = self.makeLeaf(xtrain, ytrain)
        else:
            root.right = self.buildtree(root.rightX, root.rightY, depthCount)

        return root

    #calculate the best threshold to split on for attribute passed in that yields the best information gain,
    # or other splitting metric chosen
    def getInfoGain(self, xtrain, ytrain):

        bestInfoGain = 0
        bestThreshold = 0
        xSortedByAttr, ySortedByAttr = list(zip(*sorted(zip(xtrain, ytrain), key=operator.itemgetter(0))))

        totalEntropy = self.calculateEntropyForyTrain(ytrain)

        for i in range(len(xSortedByAttr) - 1):
            if xSortedByAttr[i] == xSortedByAttr[i + 1]:
                continue

            midVal = xSortedByAttr[i] + (xSortedByAttr[i + 1] - xSortedByAttr[i]) / 2
            newInfoGain = self.getInfoGainForThreshold(zip(xSortedByAttr, ySortedByAttr), midVal, totalEntropy)

            if newInfoGain > bestInfoGain:
                bestInfoGain = newInfoGain
                bestThreshold = midVal

        return bestInfoGain, bestThreshold

    #calculates entropy for a particular set of the data
    #ytrain is the set of classified samples as part of the dataset of the node
    def calculateEntropyForyTrain(self, ytrain):

        yTrainTargetDistribution = countTargetInstances(ytrain)

        totalCases = len(ytrain)
        totalEntropy = 0
        for caseProp in yTrainTargetDistribution:
            if caseProp == 0:
                continue
            totalEntropy += -caseProp / totalCases * log(caseProp / totalCases, 2)
        return totalEntropy

    #calculate splitting metric for a threshold on the node
    #options:
    # 0 -> information gain alone
    # 1 -> information gain ratio (tends to over fit)
    # 2 -> information gain adjusted by how even a split of data there is (avoids harsh early splits to classify small proportion of the data early on)
    def getInfoGainForThreshold(self, xtrain_ytrainSortedByX, testThreshold, totalEntropy):
        xtrain, ytrain = list(zip(*xtrain_ytrainSortedByX))
        cases = xtrain
        lookup = dict(zip(xtrain, ytrain))
        totalCases = len(xtrain)
        lefty = []
        righty = []

        for case in cases:
            target = lookup[case]
            if case <= testThreshold:
                lefty.append(target)
            else:
                righty.append(target)

        balancer = 1

        if self.balancerSelected == "gainratio":
            balancer = -len(lefty)/totalCases * log(len(lefty)/totalCases, 2) - len(righty)/totalCases * log(len(righty)/totalCases, 2)
        elif self.balancerSelected == "setsplit":
            balancer = max(len(lefty)/len(righty), len(righty)/len(lefty))

        infoGain = totalEntropy - len(lefty) / totalCases * self.calculateEntropyForyTrain(lefty) - len(
            righty) / totalCases * self.calculateEntropyForyTrain(righty)

        return infoGain / balancer

    #should make leaf applies base cases for leaf production
    # 1. If the split has resulted in an empty or one entry leaf
    # 2. If only one type of Data entry
    # 3. If reached depth limit
    def shouldMakeLeaf(self, ytrain, xtrainParent, depthCount):
        if(len(xtrainParent.index._data) <= 1 or len(ytrain) <= 1):
            return True

        yTrainDist = countTargetInstances(ytrain)
        numNonZero = 0
        for caseCount in yTrainDist:
            if caseCount > 0:
                numNonZero += 1

        if(numNonZero == 1):
            return True

        if(depthCount >= self.depthLimit):
            return True

        return False

    def makeLeaf(self, xtrain, ytrain):
        leafNode = DTreeNode(xtrain, isleaf=True, entries=ytrain)
        return leafNode

    #uses pydot to recursively produce a dot graph of the tree fit similar to export graphvis
    def makeDotDiagram(self, node = None, dotGraph = None, parentDotNode = None):
        if(dotGraph == None and node == None):
            dotGraph = pydot.Dot(graph_type='digraph')
            node = self.root
            parentDotNode = pydot.Node(str.format("Attribute\t{}\nThreshold\t{:.2f}\nInfo Gain\t{:.2f}", node.attributeToTest, node.thresholdValue, node.infoGain, shape="square"))
            dotGraph.add_node(parentDotNode)

        nodeLeft = None
        nodeRight = None
        if(not node.left.isLeaf):
            nodeLeft = pydot.Node(str.format("Attribute\t{}\nThreshold\t{:.2f}\nInfo Gain\t{:.2f}", node.left.attributeToTest, node.left.thresholdValue, node.left.infoGain, shape="square"))
            self.makeDotDiagram(node.left, dotGraph, nodeLeft)
        else:
            nodeLeft = pydot.Node(str.format("Left\nLabel\t{}\nValues\t{}", node.left.label, node.left.ytrainDist, shape="square"))

        if(not node.right.isLeaf):
            nodeRight = pydot.Node(str.format("Attribute\t{}\nThreshold\t{:.2f}\nInfo Gain\t{:.2f}", node.right.attributeToTest, node.right.thresholdValue, node.right.infoGain, shape="square"))
            self.makeDotDiagram(node.right, dotGraph, nodeRight)
        else:
            nodeRight = pydot.Node(str.format("Right\nLabel\t{}\nValues\t{}", node.right.label, node.right.ytrainDist, shape="square"))

        dotGraph.add_node(nodeLeft)
        dotGraph.add_node(nodeRight)
        dotGraph.add_edge(pydot.Edge(parentDotNode, nodeLeft))
        dotGraph.add_edge(pydot.Edge(parentDotNode, nodeRight))
        return dotGraph

    #score for testing against a set of samples
    def score(self, xTest, yTest):
        wins = 0
        yTestArr = yTest.values;
        for i in range(len(yTest)):

            if yTestArr[i] == self.root.predict(xTest.iloc[i, :]):
                wins += 1
            i += 1
        return wins / len(yTest)

    #returns the predictions
    def predict(self, xTest):
        wins = 0
        yTestArr = []
        for i in range(len(xTest)):
            yTestArr.append(int(self.root.predict(xTest.iloc[i, :])))

        return yTestArr

#static method used for entropy or leaf label calculation returns occurence of each label in a set of data
def countTargetInstances(ytrain):
    yTrainTargetDistribution = [0] * (max(ytrain) + 1)
    for case in ytrain:
        yTrainTargetDistribution[int(case)] += 1
    return yTrainTargetDistribution

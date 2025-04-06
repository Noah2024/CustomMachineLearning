import os
import lzma
import numpy as np

def verifyFile(location):#Yoinked from chatGpt
    try:
        with open(location, "x") as f:
            print("File not found, creating a new one")
    except FileExistsError:
        # Handle the case where the file already exists
        print("File already exists -- Overwriting --") 

class neuronStrucutre:
    def __init__(Self, arch, setWeights):
        Self.arch = arch
        Self.scriptDir = os.path.dirname(os.path.abspath(__file__))
        Self.parentDir = os.path.dirname(Self.scriptDir)
        Self.learningRate = 1
        Self.activeFuncs = {0: Self.relu, 1: Self.relu, 2: Self.relu}
        Self.setUpNNW(setWeights)
        Self.activeDervs = {0: Self.relu, 1: Self.reluDerivative, 2: Self.reluDerivative}
    def identity(Self, X):
        print("SHOULD NEVER BE CALLED")
        return X
    def relu(Self, X):#Yoinked from chatgpt
        """
        ReLU function that works with both scalars and NumPy arrays.
        If input is a scalar, it will return a scalar.
        If input is a NumPy array, it will return an array with the same shape.
        """
        return np.maximum(0, X)
    def reluDerivative(Self, x):
        return np.where(x > 0, 1, 0)
    def setUpNNW(Self, setWeights):
        length = 0
        numNodes = 0
        for I in range(len(Self.arch)-1):
            V = Self.arch[I]
            length += (V*Self.arch[I+1])+V
            numNodes += V
        length += Self.arch[len(Self.arch)-1]#To account for the bias in the output neurons
        numNodes += Self.arch[len(Self.arch)-1]
        Self.neuronInputOutput = np.zeros(numNodes, dtype="float")
        Self.weightsAndBias = np.zeros(length-Self.arch[0], dtype="float")
        print("Num numNodes: ", numNodes)
        print("Num Weights: ", length - numNodes)
        print("Len of wight&bias: ", length-Self.arch[0])
        if setWeights:
            #array[:] = np.random.rand(*array.shape)
            Self.weightsAndBias = np.random.rand(*Self.weightsAndBias.shape)
        return np.zeros(length, dtype="float")
    
    def forwaradPropogate(Self, inpt):
        if len(inpt) != Self.arch[0]:
            print("input and expected input differ in length")
            print(len(inpt), Self.arch[0])
        Self.lastKnownInput = inpt#MUST RESET ALL OF THESE EVERY PREDICT FUNCTION CALL
        lastLayerInput = inpt
        weightIndex = (0,Self.arch[0]*Self.arch[1])
        biasIndex = (weightIndex[1], weightIndex[1]+Self.arch[1])
        #weightIndex = (biasIndex[1],biasIndex[1]+(Self.arch[0]*Self.arch[1]))
        weightShape = (Self.arch[0], Self.arch[1])
        neuronTrack = 0
        Self.preActiveSumms = np.zeros(len(Self.arch))
        for I, V in enumerate(Self.arch):
            if I != len(Self.arch)-1 and I != 0: weightIndex = (biasIndex[1], biasIndex[1]+(Self.arch[I]*Self.arch[I+1]))
            if I != len(Self.arch)-1 and I != 0: biasIndex = (weightIndex[1],weightIndex[1]+Self.arch[I+1] )
            Self.preActiveSumms[I] = lastLayerInput##NEED TO MAKE THIS WORK, NEED THE SUMMED COMPONENETS FOR THE DERVIERTIVE CALC
            lastLayerOutput = Self.activeFuncs[I](lastLayerInput) ##!!!!!!!!!!!!MUST ADDD THE SKIPPUING FUNCTION HEREoeugurboQ W68TFqfodtq wgbdvuk
            
            Self.neuronInputOutput[neuronTrack:neuronTrack+Self.arch[I]] = lastLayerOutput#MIGHT NEED TO MOVE THIS
            if I != len(Self.arch)-1:
                subset = Self.weightsAndBias[weightIndex[0]: weightIndex[1]]
                weightShape = (Self.arch[I], Self.arch[I+1])
                subset = subset.reshape(weightShape)
                #biasIndex = (weightIndex[1], weightIndex[1] +Self.arch[I+1])
                currentBias = Self.weightsAndBias[biasIndex[0]:biasIndex[1]]
                #print(lastLayerOutput[:, np.newaxis])
                #print(subset)
                #print(currentBias)
                curtSummed = np.sum(lastLayerOutput[:, np.newaxis] * subset, axis=0) + currentBias#[:, np.newaxis]
                #Self.preActiveSumms[I]
                lastLayerInput = curtSummed
                neuronTrack += Self.arch[I]
                #weightIndex = (biasIndex[1], biasIndex[1]+(Self.arch[I]*Self.arch[I+1]))
            else:
                print("FINAL OUTPUT", lastLayerOutput)
                return lastLayerOutput
    def backPropogate(Self, actualOutput, expectedOutput):
        if len(actualOutput) != len(expectedOutput):
            print("actualOutput and expectedOutput input differ in length")
            print(len(actualOutput), len(expectedOutput))
        print("Full Weights and Biases", Self.weightsAndBias)
        print("NeuronInputOutput", Self.neuronInputOutput)
        print("DIFF", actualOutput - expectedOutput)
        totalCost = np.square((actualOutput - expectedOutput))
        rollingCost = totalCost
        #lastInput = Self.neuronInputOutput[-Self.arch[len(Self.arch)-1]:]
        outputIndex = (len(Self.neuronInputOutput)-Self.arch[len(Self.arch)-1],len(Self.neuronInputOutput))
        inputIndex = (outputIndex[0]-Self.arch[len(Self.arch)-2], outputIndex[0])#Requires at least two layers
        biasIndex = (len(Self.weightsAndBias)-Self.arch[len(Self.arch)-1],len(Self.weightsAndBias) )
        print("inputIndex", inputIndex)
        print("weightBiasIndex", biasIndex)
        I = len(Self.arch)-1
        #print("last Known input", lastInput)
        for fkeI in range(len(Self.arch)-1):
            weightIndex = biasIndex[0]-(Self.arch[I-1]*Self.arch[I]), biasIndex[0]
            curtOutput = Self.neuronInputOutput[outputIndex[0]:outputIndex[1]]
            lastOutput = Self.neuronInputOutput[inputIndex[0]:inputIndex[1]]
            curtWeights = Self.weightsAndBias[weightIndex[0]:weightIndex[1]]
            curtBias = Self.weightsAndBias[biasIndex[0]:biasIndex[1]]
            print(lastOutput)
            print(curtWeights.reshape())
            print(curtBias)
            curtInput = ((lastOutput*curtWeights)+curtBias)
            print("Last Active Input", lastActiveOutput)
            print("Last Active Input", lastInput)
            dervActiveFunc = Self.activeDervs[I]
            cost = 2*(curtOutput-rollingCost)
            dCdF = cost/curtOutput#dCost/dactivation function\
            dAFdFI = curtOutput/curtInput#dActivationFunction/dFunctionInput
            dFIdw = curtInput/curtWeights#dFunctionInput/dWeight
            #print(dCdF)
            #print(dAFdFI)
            #print(dFIdw)
            dCdW = dCdF*dAFdFI*dFIdw#Need to do some matrix reshaping, cause shapes are off

            dCdB = 1
            dCdB *= dAFdFI * dFIdw
            finalChange = np.append(dCdW, dCdB)
            print(Self.weightsAndBias)
            print(finalChange)
            breakpoint()
            I -= 1
            outputIndex = (start, outputIndex[1]-Self.arch[I])
        print("COST", totalCost)
        pass
    def saveModel(Self, **kwargs):
        location = kwargs.get("location", Self.scriptDir+"\savedNNW.bin")
        verifyFile(location)
        with lzma.open(location, "wb") as f:
            #f_out.write(f_in.read())
            pickle.dump(Self.allWeights, f)            
            
    def loadModel(Self, **kwargs):
        location = kwargs.get("location", Self.scriptDir+"\savedNNW.bin")
        with lzma.open(location, "rb") as f:
                #f_out.write(f_in.read())
                Self.allWeights = pickle.load(f)  
        

NNW = neuronStrucutre([2,3,2], False)
#NNW.weightsAndBias = np.asarray([.15, 0, .25, 0])#111
#NNW.weightsAndBias = np.asarray([.25, .35,.45, .55, 0, 0, .65, .75, 0,])#221
#NNW.weightsAndBias = np.asarray([.25, .35,.45, .55,.65,.75,0, 0, 0, .85, .95, .1, 0])#231
NNW.weightsAndBias = np.asarray([.15, .25, .35, .45, .55, .65, 0, 0, 0, .75, .85, .95, .15, .25, .35, 0, 0])#232

prediction = NNW.forwaradPropogate([2, 1])
NNW.backPropogate(prediction, [1, 2])

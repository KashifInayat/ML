/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package MLP;

import java.util.List;
import java.util.Random;
import java.util.Vector;

/**
 *
 * @author Hp
 */
public class HiddenLayer implements Layer {

    List<HiddenNode> hiddenNodes;
    InputLayer inputValues;

    int dinput;
    double learningRate;
    
    int mhidden;
//Constructor
    public HiddenLayer(int numberOfHiddenNodes, int inputLength, double hiddenLayerLearingRate, InputLayer inputLayerValue) {
        dinput = inputLength;
        inputValues = inputLayerValue;
        learningRate = hiddenLayerLearingRate;
        hiddenNodes = new Vector<HiddenNode>();
        mhidden = numberOfHiddenNodes;

        
    }
/////////////Weight and nodes Initilization///////
    public void initilize() {

        for (int j = 0; j < mhidden; j++) {
            for (int i = 0; i < dinput; i++) {
                Random r = new Random();
                MultiLayerPerceptron.hiddenLayerWeights[j][i] = ((r.nextDouble() * (-0.1)) + 0.1);
            }
        }

        for (int i = 0; i < mhidden; i++) {
            hiddenNodes.add(new HiddenNode(dinput, mhidden, this));
        }

    }
//1st Step ofLearning Process of Weight Hidden Layer and Input Layer////////
    void learn(double[] expectedOutputValues,OutputVector outputValues) {
        InputVector vectorFromInputLayer = (InputVector) inputValues.getOutputValuesOfLayer();
        int nodeIndex = 0;
        double[] hiddenLayerOutput = new double[mhidden];
        
        for (HiddenNode hiddenNode : hiddenNodes) {
            hiddenNode.calculateNetInput(vectorFromInputLayer, nodeIndex);
            hiddenNode.learn(expectedOutputValues, outputValues,vectorFromInputLayer,nodeIndex);
            nodeIndex++;
        }
    }

    @Override
    public void setInputNodesValues(NeuralVector inputVector) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
////////////// Method to set values of Hidden node using the Neural Vector //////////
    @Override
    public NeuralVector getOutputValuesOfLayer() {
        InputVector vectorFromInputLayer = (InputVector) inputValues.getOutputValuesOfLayer();
        int nodeIndex = 0;
        double[] hiddenLayerOutput = new double[mhidden];
        for (HiddenNode hiddenNode : hiddenNodes) {
            hiddenNode.calculateNetInput(vectorFromInputLayer, nodeIndex);
            hiddenLayerOutput[nodeIndex] = hiddenNode.getOutput();
            nodeIndex++;
        }
        return new HiddenLayerVector(hiddenLayerOutput);

    }

}

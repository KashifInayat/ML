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
public class OutputLayer extends LearnableLayer {

    List<OutputNode> outputNodes;
    HiddenLayer hiddenLayer;
    int mhidden;
    int coutput;
    //double[][] weights;
    double learningRate;

    public OutputLayer(int outputLength, int hiddenLength, double hiddenLayerLearningRate, HiddenLayer hiddenLayerValue) {
        mhidden = hiddenLength;
        coutput = outputLength;
        hiddenLayer = hiddenLayerValue;
        learningRate = hiddenLayerLearningRate;
        outputNodes = new Vector<OutputNode>();

    }
//Initializing the Weight vector between Output and Hidden node
    public void initilize() {


        for (int j = 0; j < coutput; j++) {
            for (int i = 0; i < mhidden; i++) {
                Random r = new Random();
                MultiLayerPerceptron.outputLayerWeight[j][i] = ((r.nextDouble() * (-0.1)) + 0.1);
            }
        }

        for (int i = 0; i < coutput; i++) {
            outputNodes.add(new OutputNode(/*hiddenLayerValue,*/mhidden, coutput, this,0.52));
        }
    }
//1st Step of Learning Process of Weight between Output and Hidden Layer ////////
    public void learn(double[] expectedValues) {
        HiddenLayerVector vectorFromHiddenLayer = (HiddenLayerVector) hiddenLayer.getOutputValuesOfLayer();
        int nodeIndex = 0;
        double[] finalOutput = new double[coutput];
        for (OutputNode outputNode : outputNodes) {
            outputNode.calculateNetInput(vectorFromHiddenLayer, nodeIndex);
            outputNode.learn(expectedValues[expectedValues.length - coutput +  nodeIndex],nodeIndex,(NeuralVector) vectorFromHiddenLayer);
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
        HiddenLayerVector vectorFromHiddenLayer = (HiddenLayerVector) hiddenLayer.getOutputValuesOfLayer();
        int nodeIndex = 0;
        double[] finalOutput = new double[coutput];
        for (OutputNode outputNode : outputNodes) {
            outputNode.calculateNetInput(vectorFromHiddenLayer, nodeIndex);
            finalOutput[nodeIndex] = outputNode.getActivationFunctionOutput();
            nodeIndex++;
        }
        return new OutputVector(finalOutput);
    }
}

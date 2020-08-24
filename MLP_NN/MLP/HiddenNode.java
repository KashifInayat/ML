/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package MLP;

import java.util.Random;

/**
 *
 * @author Hp
 */
public class HiddenNode extends LearnableNode {

//    NeuralVector hiddenValue_node;
    private NeuralVector inputvalues;
    private int inputLayerSize;
    private int hiddenLayerSize;
    private double outPutOfCurrentNode;
    HiddenLayer hiddenlayer;
    private double netInput;
//Contructor
    public HiddenNode(int dinput, int mhidden, HiddenLayer hLayer) {
        inputLayerSize = dinput;
        hiddenLayerSize = mhidden;
        hiddenlayer = hLayer;

    }


    //2nd Step of Learning Process of Weight Hidden Layer and Input Layer////////
    void learn(double[] expectedOutputValues, OutputVector outputValues, NeuralVector input, int nodeSubscript) {
        getOutput();
        for (int i = 0; i < inputLayerSize; i++) {
            double multipleOfLearningRate  = 0;
            
            for(int outputIndex = expectedOutputValues.length - outputValues.length();
                    outputIndex < expectedOutputValues.length; outputIndex++){
                int c=outputIndex-inputLayerSize;
                double outputValue = outputValues.get(c);
                
                double expectedVale = expectedOutputValues[outputIndex];
              
                multipleOfLearningRate += (expectedVale - outputValue) * outputValue * (1 - outputValue) * MultiLayerPerceptron.outputLayerWeight[c][nodeSubscript] * (1 - outPutOfCurrentNode)*outPutOfCurrentNode * input.get (i);
            }
            
            
          MultiLayerPerceptron.hiddenLayerWeights[nodeSubscript][i] +=  hiddenlayer.learningRate * multipleOfLearningRate;
            
        }
    }
//Method to calculate the Netinput of Hidden Node
    @Override
    void calculateNetInput(NeuralVector input, int nodeSubscript) {
        netInput = 0;
        for (int i = 0; i < inputLayerSize; i++) {
            netInput += (input.get(i) * MultiLayerPerceptron.hiddenLayerWeights[nodeSubscript][i]);
        }
    }

    @Override
    void updateWeight() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public double getOutput() {
        outPutOfCurrentNode = (1 / (1 + Math.exp(-(netInput))));
        return outPutOfCurrentNode;
    }

}
/*@Override
    public double getOutput() {
//        calculateNetInput();
//        performActivationFunction();
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.

    }

    
    @Override
    public void calculateNetInput() {
    for(int j=0;j<=mhidden;j++)
    {
        for(int i=0;i<=dinput;i++)
        {
        hiddenNodes.set(j)= inputNodes.get(i)*weights[j][i];
        }
    }

    
    public void updateWeight() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    void learn() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
 */

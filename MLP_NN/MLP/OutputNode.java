/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package MLP;

/**
 *
 * @author Hp
 */
public class OutputNode extends LearnableNode {

    NeuralVector outputValue_node;
    NeuralVector hiddenvalues;
    int hiddenLayerSize;
    int outputLayerSize;
    OutputLayer outputLayer;
    private double netInput;
    private double outputOfCurrentNode;
    private double activationFunctionOutput;

    double bias;

    public OutputNode(int mhidden, int coutput, OutputLayer oLayer, double nodeeBias) {
//        hiddenvalues=hiddenLayerValue;
        hiddenLayerSize = mhidden;
        outputLayerSize = coutput;
        bias = nodeeBias;
        outputLayer = oLayer;

    }

    public OutputNode(int mhidden, int coutput, OutputLayer oLayer) {
//        hiddenvalues=hiddenLayerValue;
        hiddenLayerSize = mhidden;
        outputLayerSize = coutput;
        outputLayer = oLayer;

    }

    /*public void netInput() {
     //        for (int j = 0; j <= outputLayerSize; j++) {
     //            for (int i = 0; i <= hiddenLayerSize; i++) {
     //                outputValue_node.setNthValue(j, (outputValue_node.getNthValue(j) + (hiddenvalues.get(i) * weights[j][i])));
     //            }
     //
     //            outputValue_node.setNthValue(j, ());
     //        }

     }*/
    //2nd Step of Learning Process of between Output and Hidden Layer////////
    void learn(double expectedValue, int nodeSubscript, NeuralVector hiddenLayerOutput) {
        getOutput();
        for (int i = 0; i < hiddenLayerSize; i++) {

            MultiLayerPerceptron.outputLayerWeight[nodeSubscript][i] += outputLayer.learningRate * (expectedValue - activationFunctionOutput) * activationFunctionOutput * (1 - activationFunctionOutput) * hiddenLayerOutput.get(i);
            // System.out.println("Output Weight" + MultiLayerPerceptron.outputLayerWeight[nodeSubscript][i]);
        }

    }
//Mthod to calculate the net input of the output node
    @Override
    void calculateNetInput(NeuralVector input, int nodeSubscript) {
        netInput = 0;
        for (int i = 0; i < hiddenLayerSize; i++) {
            netInput += (input.get(i) * MultiLayerPerceptron.outputLayerWeight[nodeSubscript][i]);
        }
    }

    @Override
    void updateWeight() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

//Mthod to to apply the activation function on net input of the output node
    public double getActivationFunctionOutput() {

        activationFunctionOutput = (1 / (1 + Math.exp(-(netInput))));
        //System.out.println("Finaloutput: " +outputOfCurrentNode );
        return activationFunctionOutput;
    }

    @Override
    public double getOutput() {
        //getActivationFunctionOutput();
        //Here you can set bias(CSV: categorical Score Value) value according to your own training set
        outputOfCurrentNode = (activationFunctionOutput > bias ? 1 : 0);
        return outputOfCurrentNode;
    }

}

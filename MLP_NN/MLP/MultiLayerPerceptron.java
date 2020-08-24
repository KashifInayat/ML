/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package MLP;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Vector;

/**
 *
 * @author Hp
 */
public class MultiLayerPerceptron {

    InputLayer inputLayer;
    OutputLayer outputLayer;
    HiddenLayer hiddenLayer;
    public static double[][] hiddenLayerWeights;//Weight vector between Hidden and input Layer
    public static double[][] outputLayerWeight;//Weight vector between Output and Hidden Layer
    int numberOfInputNodes;
    int numberOfhiddenNodes;
    int numberOfOutputNodes;
//Initializing the number of node, and weights sizes
    public MultiLayerPerceptron(int d, int m, int c) {
        numberOfInputNodes = d;
        numberOfhiddenNodes = m;
        numberOfOutputNodes = c;
        hiddenLayerWeights = new double[m][d];
        outputLayerWeight = new double[c][m];

    }
//Initializing the complete network 
    public void initilize() {
        inputLayer = new InputLayer(numberOfInputNodes);
        hiddenLayer = new HiddenLayer(numberOfhiddenNodes, numberOfInputNodes, 0.3, inputLayer);
        outputLayer = new OutputLayer(numberOfOutputNodes, numberOfhiddenNodes, 0.3, hiddenLayer);
        hiddenLayer.initilize();
        outputLayer.initilize();

    }
//////////////////////Interactive Learning////////

    public void train(ArrayList<double[]> trainingExamples, int trainingIterations) throws Exception {
        if (trainingExamples.size() < 1) {
            throw new Exception("training examples must have some values.");
        }
        if (trainingExamples.get(0).length != numberOfInputNodes + numberOfOutputNodes) {
            throw new Exception("Length of training Example is not consistant with dimensions of MLP.");
        }
        //why this running for 1 time
        for (int i = 0; i < trainingIterations; i++) {//How many time inner loop will this run?29
            for (int j = 0; j < trainingExamples.size(); j++) {
                InputVector iv = new InputVector(trainingExamples.get(j), numberOfInputNodes);
                iv.printScreen();// To print Input Values
                inputLayer.setInputNodesValues(iv);
                OutputVector ov = (OutputVector) outputLayer.getOutputValuesOfLayer();
                outputLayer.learn(trainingExamples.get(j));
                hiddenLayer.learn(trainingExamples.get(j),ov);
                ov.printScreen();//To print Output Values

            }
        }

    }
}

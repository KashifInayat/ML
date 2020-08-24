/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package MLP;

import java.util.List;
import java.util.Vector;

/**
 *
 * @author Hp
 */
public class InputLayer implements Layer{
    int dinput;
    NeuralVector inputLayerValues;
    public InputLayer(int inputLength){
        
        dinput=inputLength;
    }
//Method so set the Input Vector
    @Override
    public void setInputNodesValues(NeuralVector inputVector) {
        inputLayerValues = inputVector;
    }
//Method to get the input Vector
    public NeuralVector getOutputValuesOfLayer() {
        return inputLayerValues;
    }
   
}

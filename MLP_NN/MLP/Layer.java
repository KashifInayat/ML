/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package MLP;

import java.util.List;

/** 
 *
 * @author Hp
 */
//Every Layer using this interface
public interface Layer {
    void setInputNodesValues(NeuralVector inputVector);
    NeuralVector getOutputValuesOfLayer();
    
    
}

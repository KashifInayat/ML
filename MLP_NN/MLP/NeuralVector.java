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

//Every Layer use this vector as List/Array to store the values
public interface NeuralVector {
    void setValues(List<Double> value);
    List<Double> getValues();
    void setNthValue(int n, double value);
    
    double getNthValue(int n);

    public double get(int i);
    
}

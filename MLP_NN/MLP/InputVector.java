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
public class InputVector implements NeuralVector {

    Vector<Double> input = new Vector();

    InputVector(double[] inputArray, int numberOfInputNodes) {
        for(int i = 0; i< numberOfInputNodes;i++){
            input.add((double)inputArray[i] );
        }
    }
//Formatting the Input values to print
    void printScreen() {
                String s1= "";
        for(Double inputValue: input){
            s1 +=  inputValue + " " ;
        }
        System.out.println("input: " + s1 );
    }
    
    @Override
    public void setValues(List<Double> value) {
        input = (Vector)value ;
    }

    @Override
    public List<Double> getValues() {
        return input;
    }

    

    @Override
    public double get(int i) {
        return input.elementAt(i);
    }

    
    @Override
    public void setNthValue(int n, double value) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public double getNthValue(int n) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    

}

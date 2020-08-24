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
 * @author Asif
 */
public class OutputVector implements NeuralVector {

    Vector<Double> finalOutputVector = new Vector();

    public OutputVector(double[] outputVector) {
        for (int i = 0; i < outputVector.length; i++) {
            finalOutputVector.add((double) outputVector[i]);
        }
    }

    public int length(){
        return finalOutputVector.size();
    }
    
    void printScreen() {
        
        String s1= "";
        for(Double outputValue: finalOutputVector){
            s1 +=  outputValue + " " ;
        }
        System.out.println("output: " + s1 );
    }
    
    @Override
    public void setValues(List<Double> value) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public List<Double> getValues() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void setNthValue(int n, double value) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public double getNthValue(int n) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public double get(int i) {
        return finalOutputVector.elementAt(i);
    }

    
    
}

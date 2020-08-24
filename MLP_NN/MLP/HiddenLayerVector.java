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

//Vector for Hidden Node
public class HiddenLayerVector implements NeuralVector {

    Vector<Double> hiddenVector = new Vector();

    HiddenLayerVector(double[] outPutValuesOfHiddenLayer) {
        for (int i = 0; i < outPutValuesOfHiddenLayer.length; i++) {
            hiddenVector.add((double) outPutValuesOfHiddenLayer[i]);
        }
    }

    @Override
    public void setValues(List<Double> value) {
        hiddenVector = (Vector)value;
    }

    @Override
    public List<Double> getValues() {
        return hiddenVector;
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
        return hiddenVector.elementAt(i);
    }

}

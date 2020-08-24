/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package mlpclass;

import MLP.MultiLayerPerceptron;
import java.util.ArrayList;

/**
 *
 * @author Hp
 */
public class MLPImplementation {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        try {
            ////////////Training Set/////////////
            ArrayList<double[]> traningExamples = new ArrayList<double[]>();
            traningExamples.add(new double[]{96, 95, -3, 1, 0, 0});
            traningExamples.add(new double[]{102, 102, 3, 1, 0, 0});
            traningExamples.add(new double[]{101, 100, -4, 1, 0, 0});
            traningExamples.add(new double[]{97, 99, 2, 1, 0, 0});
            traningExamples.add(new double[]{102, 96, -2, 1, 0, 0});
            traningExamples.add(new double[]{3, 97, 105, 0, 1, 0});
            traningExamples.add(new double[]{-5, 99, 97, 0, 1, 0});
            traningExamples.add(new double[]{5, 105, 102, 0, 1, 0});
            traningExamples.add(new double[]{0, 96, 101, 0, 1, 0});
            traningExamples.add(new double[]{5, 101, 103, 0, 1, 0});
            traningExamples.add(new double[]{0, 103, 98, 0, 1, 0});
            traningExamples.add(new double[]{-1, 99, 95, 0, 1, 0});
            traningExamples.add(new double[]{-4, 98, 96, 0, 1, 0});
            traningExamples.add(new double[]{0, 101, 104, 0, 1, 0});
            traningExamples.add(new double[]{5, 103, 104, 0, 1, 0});
            traningExamples.add(new double[]{-5, 4, 4, 0, 0, 1});
            traningExamples.add(new double[]{1, 4, 2, 0, 0, 1});
            traningExamples.add(new double[]{0, -4, 2, 0, 0, 1});
            traningExamples.add(new double[]{-5, -5, 1, 0, 0, 1});
            traningExamples.add(new double[]{-5, -2, 1, 0, 0, 1});
            traningExamples.add(new double[]{1, 2, -1, 0, 0, 1});
///////////////////////Initializing the MLP by giving values of Number of Nodes of each layer
            MultiLayerPerceptron Mlp = new MultiLayerPerceptron(3, 5, 3);
            Mlp.initilize();
//////Here i calling train Method to to train it on each training set and repeating it 10 time
            Mlp.train(traningExamples, 10);
        }catch (Exception e) {
            System.out.println("Exception: " + e.getMessage());
        
            }
        
    }

}

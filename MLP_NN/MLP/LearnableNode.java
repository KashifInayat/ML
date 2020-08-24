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
public abstract class LearnableNode implements Node{
    //abstract void learn();
    abstract void calculateNetInput(NeuralVector input,int nodeSubscript);
    abstract void updateWeight();
}

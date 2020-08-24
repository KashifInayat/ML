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
public class InputNode implements Node {
double XValue_inputnode;
//Method to set the input node value

public void setInput(double input) {
        XValue_inputnode=input;
    }
//Method to get the input node value

    @Override
    public double getOutput( ) {
      return XValue_inputnode;
    }
  
}

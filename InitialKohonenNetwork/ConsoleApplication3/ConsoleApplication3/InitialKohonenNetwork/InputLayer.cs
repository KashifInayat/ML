using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Initial.KohonenNetwork
{
    public class InputLayer
    {
		
        private int numberOfInputNodes;
        public static InputNode[] inputNodes;

		// creates Input layer as
        public InputLayer(int numberOfInputNodes)
        {
            this.numberOfInputNodes = numberOfInputNodes;
            inputNodes = new InputNode[numberOfInputNodes];
            for (int j = 0; j < numberOfInputNodes; j++)
            {
                inputNodes[j] = new InputNode();
            }
        }

		//takes as input double array and passes it's coresponding elements to in put nodes, 
        public double[] Input
        {
            set
            {
                if (value.Length != numberOfInputNodes)
                    throw new Exception("length of input vector is not same as lenght of input layer");

                for (int j = 0; j < numberOfInputNodes; j++)
                {
                    inputNodes[j].Input = value[j];
                }
            }
        }

    }
}

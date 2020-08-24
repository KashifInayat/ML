using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Initial.KohonenNetwork
{
    public class CometitiveNode
    {
        private int inputLayerLength;
        private int nodeSubscript;

		// calculate net input by calculating sum of products of the weight vector and input vector of currnet competative node 
        public double NetInput
        {
            get
            {
                double netInput = 0;
                for(int j=0;j< inputLayerLength;j++){
                    netInput += InitialKohonenNetwork.weights[nodeSubscript, j] * InputLayer.inputNodes[j].Output;
                }

                return netInput;
            }
        }

        public int InputLayerLength
        {
            set { inputLayerLength = value; }
            get { return inputLayerLength; }
        }

        public int NodeSubscript
        {
            set { nodeSubscript = value; }
            get { return nodeSubscript; }
        }

        public CometitiveNode()
        {
            this.inputLayerLength = 0;
        }

        public CometitiveNode(int inputLayerLength )
        {
            this.inputLayerLength = inputLayerLength;
        }


    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Initial.KohonenNetwork
{
    class CometitiveLayer
    {
        private int numberOfCompetitveNodes;
        private int numberOfInputNodes;

       
        public CometitiveNode[] CompetativeNodes
        {
            set;
            get;
        }

        public CometitiveLayer(int numberOfCompetitveNodes, int numberOfInputNodes)
        {
            this.numberOfCompetitveNodes = numberOfCompetitveNodes;
            this.numberOfInputNodes = numberOfInputNodes;
            this.CompetativeNodes = new CometitiveNode[numberOfCompetitveNodes];

        }

		// this method initilizes the Array of competative nodes, depenmding on size of input layer and assign index to each node in competative layer
        public void Initialize()
        {
            int nodeIndex = 0;
            for (int i = 0; i < numberOfCompetitveNodes; i++)
            {
                this.CompetativeNodes[i] = new CometitiveNode();
                this.CompetativeNodes[i].InputLayerLength = numberOfInputNodes;
                this.CompetativeNodes[i].NodeSubscript = nodeIndex;
                nodeIndex++;
            }

            
        }
    }
}

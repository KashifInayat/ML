using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Initial.KohonenNetwork
{
    class InitialKohonenNetwork
    {
        private int numberOfInputNodes;
        private int numberOfCompetitveNodes;
        private InputLayer inputLayer;
        private CometitiveLayer competitiveLayer;
        public static double[,] weights;
        Random rand = new Random();

		
        public double LearningRate
        {
            set;
            get;
        }

        public InitialKohonenNetwork(int numberOfInputNodes, int numberOfCompetitveNodes)
        {
            this.numberOfInputNodes = numberOfInputNodes;
            this.numberOfCompetitveNodes = numberOfCompetitveNodes;
            this.inputLayer = new InputLayer(numberOfInputNodes);
            this.competitiveLayer = new CometitiveLayer(numberOfCompetitveNodes, numberOfInputNodes);
            weights = new double[numberOfCompetitveNodes, numberOfInputNodes];
            LearningRate = 0.3;
        }

		// this method initilizes the competitiveLayer and initilizes the weight matrix by random value between (-0.1 to 0.1 )
        public void initilize()
        {
            competitiveLayer.Initialize();
            for (int i = 0; i < numberOfCompetitveNodes; i++)
            {
                for (int j = 0; j < numberOfInputNodes; j++)
                {
                    weights[i, j] = (rand.NextDouble() / 0.5) - 0.1;//Produce weight between -0.1 and 0.1
                }
            }

        }
		
		//Used to print node value, subscript and trainingexampel index for training examples
        public void PrintCompetativeNodeStatus(double value, int subscript,int trainingIndex)
        {

            Console.WriteLine("Training Example (" + trainingIndex + ") (" + subscript + ")th net input :" + value.ToString("F"));
        }

		//Used to print dotted line in results
        public void PrintEndLine()
        {
            Console.WriteLine("------------------------------------------");
        }

		//Used to print weigt matrix in results
        public void PrintWeightMatrix(string text = "")
        {
            
            string matrix = "weight matrix: \n";
            if (!String.IsNullOrEmpty(text))
                matrix += text + "\n";

            for (int j = 0; j < numberOfInputNodes; j++)
            {
                matrix += weights[0,j].ToString("F");
                for (int i = 1; i < numberOfCompetitveNodes; i++)
                {
                    matrix += ", " + weights[i, j].ToString("F");

                }
                matrix += "\n";
            }

            Console.WriteLine("------------------------------------------");
            Console.WriteLine(matrix);

            Console.WriteLine("------------------------------------------");

        }

		// this function ilustrates learning process
        public void Learn()
        {
            
			// cheks for validity of training examples
                if (TrainingExamples == null || TrainingExamples.Count == 0)
                {
                    throw new Exception("please set training examples to start learning procss of the network");
                }
				
				// print initial weight matrix
                PrintWeightMatrix("start of training Iteration");

				
                WinnerNode winner = new WinnerNode();
                // repeat for all training examples
				for (int trainingIndx = 0; trainingIndx < TrainingExamples.Count; trainingIndx++)
                {
					
					double[] trainingExample = TrainingExamples[trainingIndx];
                    inputLayer.Input = trainingExample;

					// set first node as winner and its net input as highest net input
                    
                    winner.subscript = 0;
                    winner.netInput = competitiveLayer.CompetativeNodes[0].NetInput;
                    // print net input and subscript of node  
					PrintCompetativeNodeStatus(winner.netInput, winner.subscript, trainingIndx);
                    //List<CometitiveNode> competativeNodes = new List<CometitiveNode>(competitiveLayer.CompetativeNodes);
                    double ithNetInput = 0;
                    // itrate through this loop to check if any other competative node has net input higher than first node.
					for (int i = 1; i < numberOfCompetitveNodes; i++)
                    {
						
                        ithNetInput = competitiveLayer.CompetativeNodes[i].NetInput;
                        PrintCompetativeNodeStatus(ithNetInput, i, trainingIndx);
                        if (winner.netInput < ithNetInput)
                        {
                            winner.subscript = i;
                            winner.netInput = ithNetInput;

                        }
                    }

					// update weight matrix for winner competative node.
                    for (int j = 0; j < numberOfInputNodes; j++)
                    {
                        weights[winner.subscript, j] += LearningRate * (trainingExample[j] - weights[winner.subscript, j]);

                        
                    }
					
					//print winner node subscript.

                    Console.WriteLine("Winner node: " + winner.subscript);

                    PrintEndLine();
                }
                PrintWeightMatrix();
            

        }


        public List<double[]> TrainingExamples
        {
            set;
            get;
        }

    }
}

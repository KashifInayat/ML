using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Initial.KohonenNetwork;

namespace ConsoleApplication3
{
    class Program
    {
        static void Main(string[] args)
        {
            InitialKohonenNetwork Ikn = new InitialKohonenNetwork(3, 3);
            Ikn.initilize();
            List<double[]> trainingExamples = new List<double[]>();
            trainingExamples.Add(new double[] { 2, 0, 0 });
            trainingExamples.Add(new double[] { 0, 2, 0 });
            trainingExamples.Add(new double[] { 0, 0, 2 });

            Ikn.TrainingExamples = trainingExamples;
            for(int i= 0; i< 10; i++)
                Ikn.Learn();
            Console.WriteLine("OKAY");
        }
    }
}

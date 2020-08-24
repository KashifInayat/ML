using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Initial.KohonenNetwork
{
    public class InputNode
    {
        double nodeValue;

		//this class is used to carry input vale, input value and output vlaue is same in input node  
        public double Input
        {
            set { nodeValue = value; }
        }

        public double Output
        {
            get { return nodeValue; }
        }

    }
}

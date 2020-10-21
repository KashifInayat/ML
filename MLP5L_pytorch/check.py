import torch
import torch.nn as nn
from torch.autograd import Variable
from monitor import *

def check_model(model, input_shape, input_shaping=False, inputs=None):
	batch_size = 256
	if inputs is None:
	    inputs = torch.normal(0,1, input_shape)

	print("Make sure you are using a single GPU. You can run 'export CUDA_VISIBLE_DEVICES=0', for example.")
	m = SingleBatchStatisticsPrinter(None) 

	if torch.cuda.is_available():
	    inputs = inputs.cuda()

	if input_shaping:
	    inputs = model(inputs)
	inputs = Variable(inputs, requires_grad=True)
	m.tensors["actin#model"] = inputs.data.cpu()
	m.set_network(model)

	m.start_epoch("train", 1)
	m.start_update("train", 1, 0)
	model.train()
	outputs = model(inputs)

	doutputs = torch.normal(0,1, outputs.shape)
	if torch.cuda.is_available():
	    doutputs = doutputs.cuda()
	outputs.backward(doutputs)
	m.end_update("train", 1, 0)
	m.end_epoch("train", 1)
	m.tensors["gradout#model"] = inputs.grad.data.cpu()
	print("var din: %f " % inputs.grad.var()) ## will be removed
	return m.tensors

###MLP Layers
class SimpleModel(nn.Module):
	def __init__(self):
		super(SimpleModel, self).__init__()
		self.layer1 = nn.Linear(50, 100)#50-100-150-200-300-500
		self.layer2 = nn.Linear(100,150)
		self.layer3 = nn.Linear(150,200)
		self.layer4 = nn.Linear(200,250)
		self.layer5 = nn.Linear(250,500)
	def forward(self, x):
		return self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
simple_model = SimpleModel().cuda()
check_model(simple_model, (200,50))

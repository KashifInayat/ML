import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import time
import math
import argparse

from utils import progress_bar
from torch.utils.tensorboard import SummaryWriter



parser = argparse.ArgumentParser(description='PyTorch ImageNet/Imagette Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('--bit', default=4, type=int, help='bit-width for lsq quantizer')

parser.add_argument('--dataset', default='imagenette', type=str,
                    help='dataset name for training')
parser.add_argument('--data_root', default = '/soc_local/data/pytorch/imagenet/', type=str,
                    help='path to dataset')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--arch', default='resnet18', type=str,
                    help='network architecture')
# ------
parser.add_argument('--init_from', type=str,
                    help='init weights from from checkpoint')
# ------

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--epochs', default=120, type=int, help='number of training epochs')

parser.add_argument('--train_id', type=str, default= 'train-01',
                    help='training id, is used for collect experiment results')

parser.add_argument('--train_scheme', type=str, default= 'fp32',
                    help='Training scheme')

parser.add_argument('--output_dir', type=str, default= 'outputs',
                    help='output directory')

parser.add_argument('--print_freq', default=10, type=int, help='log print frequency.')


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
working_dir = os.path.join(args.output_dir, args.train_id)
os.makedirs(working_dir, exist_ok=True)
writer = SummaryWriter(working_dir)

# Data
print('==> Preparing data..')

if args.dataset in ("imagenet", "imagenette"):
    traindir = os.path.join(args.data_root, 'train')
    valdir = os.path.join(args.data_root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))


    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, sampler=None)

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
else: 
    raise NotImplementedError('Not support this type of dataset: ' + args.dataset)

# Model
print('==> Building model..')
if args.dataset == "imagenette":
    if args.arch == "resnet18":
        from models.imagenette_resnet import *
        net = resnet18(num_classes=10)
else:
    raise NotImplementedError('Not support this type of dataset: ' + args.dataset)


if args.train_scheme.startswith("nips2019-quantization"):
    from nips2019_quantizer import NIPS2019_QConv2d, NIPS2019_QLinear, NIPS2019_QInputConv2d

    def replace_module(model, num_bits):
        def __replace_module(model, num_bits):
            for module_name in model._modules:
                m = model._modules[module_name]             

                if isinstance(m, nn.Conv2d):
                    model._modules[module_name] = NIPS2019_QConv2d(in_channels=m.in_channels, out_channels=m.out_channels, kernel_size=m.kernel_size, stride=m.stride, padding=m.padding, dilation=m.dilation, groups=m.groups, bias=m.bias, bit=num_bits)

                elif isinstance(m, nn.Linear):
                    # model._modules[module_name].quan_a.bits =num_bits
                    pass

                elif len(model._modules[module_name]._modules) > 0:
                    __replace_module(model._modules[module_name], num_bits)

        __replace_module(model, num_bits)



    # for n, m in net.named_modules():
    #     #print (n)
    #     if isinstance(m, nn.Conv2d):
    #         # : int, : int, : Union[T, Tuple[T, T]], : Union[T, Tuple[T, T]] = 1, : Union[T, Tuple[T, T]] = 0, : Union[T, Tuple[T, T]] = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'
    #         #net._modules[n] = 
    #         setattr(net, n, nn.ReLU())
    replace_module(net, args.bit)
    m =  net._modules["conv1"]
    net._modules["conv1"] = NIPS2019_QInputConv2d(in_channels=m.in_channels, out_channels=m.out_channels, kernel_size=m.kernel_size, stride=m.stride, padding=m.padding, dilation=m.dilation, groups=m.groups, bias=m.bias, bit=8)

    m =  net._modules["fc"]
    net._modules["fc"] = NIPS2019_QLinear(in_features=m.in_features, out_features=m.out_features, bias=(m.bias != None), bit=8)



if args.train_scheme.startswith("lsq"):
    from lsq import InputConv2dLSQ, LinearLSQ, Conv2dLSQ

    def replace_module(model, num_bits):
        def __replace_module(model, num_bits):
            for module_name in model._modules:
                m = model._modules[module_name]             

                if isinstance(m, nn.Conv2d):
                    model._modules[module_name] = Conv2dLSQ(in_channels=m.in_channels, out_channels=m.out_channels, kernel_size=m.kernel_size, stride=m.stride, padding=m.padding, dilation=m.dilation, groups=m.groups, bias=m.bias, bit=num_bits)

                elif isinstance(m, nn.Linear):
                    # model._modules[module_name].quan_a.bits =num_bits
                    pass

                elif len(model._modules[module_name]._modules) > 0:
                    __replace_module(model._modules[module_name], num_bits)

        __replace_module(model, num_bits)



    # for n, m in net.named_modules():
    #     #print (n)
    #     if isinstance(m, nn.Conv2d):
    #         # : int, : int, : Union[T, Tuple[T, T]], : Union[T, Tuple[T, T]] = 1, : Union[T, Tuple[T, T]] = 0, : Union[T, Tuple[T, T]] = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'
    #         #net._modules[n] = 
    #         setattr(net, n, nn.ReLU())
    replace_module(net, args.bit)
    m =  net._modules["conv1"]
    net._modules["conv1"] = InputConv2dLSQ(in_channels=m.in_channels, out_channels=m.out_channels, kernel_size=m.kernel_size, stride=m.stride, padding=m.padding, dilation=m.dilation, groups=m.groups, bias=m.bias, bit=8)

    m =  net._modules["fc"]
    net._modules["fc"] = LinearLSQ(in_features=m.in_features, out_features=m.out_features, bias=(m.bias != None), bit=8)

# print (net)
# exit()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

print (net)
time.sleep(5)

if args.init_from and os.path.isfile(args.init_from):
    # Load checkpoint.
    print('==> Initializing from checkpoint..')
    checkpoint = torch.load(args.init_from)

    net_state_dict = net.state_dict()
    net_state_dict.update(checkpoint['net'])
    net.load_state_dict(net_state_dict)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=args.wd)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if batch_idx % args.print_freq == 0:
            print ("[Train] Epoch=", epoch,  " BatchID=", batch_idx, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'  \
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return (train_loss/batch_idx, correct/total)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            if batch_idx % args.print_freq == 0:
                print ("[Test] Epoch=", epoch, " BatchID=", batch_idx, 'Loss: %.3f | Acc: %.3f%% (%d/%d)' \
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        
        torch.save(state,  os.path.join(working_dir, 'ckpt_best.pth'))
        best_acc = acc
        print ('Best accuracy: ', best_acc)
        

    return (test_loss/batch_idx, correct/total)

if args.evaluate:
    print ("==> Start evaluating ...")
    test(-1)
    exit()

for epoch in range(start_epoch, args.epochs):
    train_loss, train_acc1 = train(epoch)
    test_loss, test_acc1 = test(epoch)

    lr_scheduler.step()

    # tensorboard log
    writer.add_scalar('Train/Loss', train_loss, epoch)
    writer.add_scalar('Train/Acc1', train_acc1, epoch)

    writer.add_scalar('Test/Loss', test_loss, epoch)
    writer.add_scalar('Test/Acc1', test_acc1, epoch)

import os
import sys
import glob
import numpy as np
from numpy.core.fromnumeric import mean, repeat
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from model import NetworkCIFAR as Network
import time


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='CIFAR10_29.pt', help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--arch', type=str, default='OURS1_V3', help='which architecture to use')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

CIFAR_CLASSES = 10


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  torch.cuda.set_device(args.gpu)
  cudnn.enabled=True
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  model = model.cuda()
  try:
    utils.load(model, args.model_path)
  except:
    model = model.module
    utils.load(model, args.model_path)

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  _, test_transform = utils._data_transforms_cifar10(args)
  test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)

  test_queue = torch.utils.data.DataLoader(
      test_data, batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=2)

  model.drop_path_prob = 0.0
  #test_acc, test_obj = infer(test_queue, model, criterion)
  #logging.info('Test_acc %f', test_acc)

  # Test inference time
  a = infer2(model)
  print(a)
  


def infer(test_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(test_queue):
    input = input.cuda()
    target = target.cuda()
    with torch.no_grad():
        logits, _ = model(input)
        loss = criterion(logits, target)
    

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg



def infer2(model):
  model.eval()

  start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
  repetitions = 1000
  timings = np.zeros((repetitions,1))
  dummy = torch.randn(1, 3, 32, 32 , dtype= torch.float).cuda()

  # Warm up
  for i in range(10):
    logits, _ = model(dummy)

  # Measure performance
  with torch.no_grad():
    for rep in range(repetitions):
      start.record()
      logits, _ = model(dummy)
      end.record()
      torch.cuda.synchronize()
      curr_time = start.elapsed_time(end)
      timings[rep] = curr_time
      #loss = criterion(logits, target)
    
  mean_time = np.sum(timings) / repetitions

  return mean_time
    

if __name__ == '__main__':
  main()



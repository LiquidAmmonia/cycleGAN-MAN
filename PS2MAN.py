import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools

import torch.nn as nn
from torchvision import models, transforms
import sys
import argparse
from visualizer import Visualizer
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from data.
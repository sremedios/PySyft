"""Some syft imports..."""

'''
from . import core
from . import spdz
from .core.frameworks.torch import _SyftTensor

from torch.autograd import Variable
from torch.nn import Parameter
from torch.autograd import Variable as Var
from .core.frameworks.torch import TorchHook
from .core.frameworks.torch import _LocalTensor, _PointerTensor, _FixedPrecisionTensor, \
    _PlusIsMinusTensor, _GeneralizedPointerTensor, _SPDZTensor
from syft.core.workers import VirtualWorker, SocketWorker
from .core.frameworks.numpy import array

__all__ = ['core', 'spdz']

import syft
import torch

for f in dir(torch):
    if ("_" not in f):
        setattr(syft, f, getattr(torch, f))

setattr(syft, 'deser', _SyftTensor.deser)
'''

"""Some Tensorflow imports..."""
from . import core
from . import spdz
from .core.frameworks.tensorflow import _SyftTensor

from tensorflow import Variable
#from tensorflow.nn import Parameter
from tensorflow import Variable as Var
from .core.frameworks.tensorflow import TensorflowHook
from .core.frameworks.tensorflow import _LocalTensor, _PointerTensor, _FixedPrecisionTensor, \
    _PlusIsMinusTensor, _GeneralizedPointerTensor, _SPDZTensor
from syft.core.workers import VirtualWorker, SocketWorker
from .core.frameworks.numpy import array

__all__ = ['core', 'spdz']

import syft
import tensorflow

for f in dir(tensorflow):
    if ("_" not in f):
        setattr(syft, f, getattr(tensorflow, f))

setattr(syft, 'deser', _SyftTensor.deser)

from .hook import TensorflowHook
from .tensor import _SyftTensor, _LocalTensor, _PointerTensor
from .tensor import _FixedPrecisionTensor, _TensorflowTensor, _PlusIsMinusTensor, _GeneralizedPointerTensor
from .tensor import _SPDZTensor

__all__ = ['TensorflowHook', '_SyftTensor', '_LocalTensor',
           '_PointerTensor', '_FixedPrecisionTensor', '_TensorflowTensor',
           '_PlusIsMinusTensor', '_GeneralizedPointerTensor', '_SPDZTensor']

import tensorflow as tf 

# this is a list of all module functions in the tf module
tf.tf_funcs = dir(tf)

# this is a list of all module functions in tf.nn
tf.tf_nn_funcs = dir(tf.nn)

# Gathers all the functions from above
tf.tf_modules = {
    'tf': tf.tf_funcs,
    'tf.nn': tf.tf_nn_funcs
}

# this is the list of tf tensor types that we will override for remote execution
tf.tensor_types = [tf.Tensor]

tf.var_types = [tf.Variable]#, tf.nn.Parameter]

# a list of all classes in which we will override their methods for remote execution
tf.tensorvar_types = tf.tensor_types + [tf.Variable]

tf.tensorvar_types_strs = [x.__name__ for x in tf.tensorvar_types]

tf.tensorvar_methods = list(
    set(
        [method
         for tensorvar in tf.tensorvar_types
         for method in dir(tensorvar)]
    )
)
tf.tensorvar_methods.append('get_shape')
tf.tensorvar_methods.append("share")
tf.tensorvar_methods.append("fix_precision")
tf.tensorvar_methods.append("decode")

# Tensorflow functions we don't want to override
tf.tf_exclude = ['save', 'load', 'typename', 'is_tensor', 'manual_seed']

tf.guard = {
    'syft.core.frameworks.tf.tensor.Variable': tf.Variable,
    'syft.core.frameworks.tf.tensor._PointerTensor': _PointerTensor,
    'syft.core.frameworks.tf.tensor._SyftTensor': _SyftTensor,
    'syft.core.frameworks.tf.tensor._LocalTensor': _LocalTensor,
    'syft.core.frameworks.tf.tensor._FixedPrecisionTensor': _FixedPrecisionTensor,
    'syft.core.frameworks.tf.tensor._GeneralizedPointerTensor': _GeneralizedPointerTensor,
    'syft._PlusIsMinusTensor': _PlusIsMinusTensor,
    'syft._SPDZTensor': _SPDZTensor,
    'syft._FixedPrecisionTensor': _FixedPrecisionTensor,
    'syft.core.frameworks.tf.tensor.FloatTensor': tf.Tensor,
    'syft.core.frameworks.tf.tensor.DoubleTensor': tf.Tensor,
    'syft.core.frameworks.tf.tensor.HalfTensor': tf.Tensor,
    'syft.core.frameworks.tf.tensor.ByteTensor': tf.Tensor,
    'syft.core.frameworks.tf.tensor.CharTensor': tf.Tensor,
    'syft.core.frameworks.tf.tensor.ShortTensor': tf.Tensor,
    'syft.core.frameworks.tf.tensor.IntTensor': tf.Tensor,
    'syft.core.frameworks.tf.tensor.LongTensor': tf.Tensor,
    'syft.Variable': tf.Variable,
    'syft.FloatTensor': tf.Tensor,
    'syft.DoubleTensor': tf.Tensor,
    'syft.HalfTensor': tf.Tensor,
    'syft.ByteTensor': tf.Tensor,
    'syft.CharTensor': tf.Tensor,
    'syft.ShortTensor': tf.Tensor,
    'syft.IntTensor': tf.Tensor,
    'syft.LongTensor': tf.Tensor,
    #'syft.Parameter': tf.nn.Parameter
}


def _command_guard(command, allowed):
    if isinstance(allowed, dict):
        allowed_names = []
        for module_name, func_names in allowed.items():
            for func_name in func_names:
                allowed_names.append(module_name + '.' + func_name)
        allowed = allowed_names
    if command not in allowed:
        raise RuntimeError(
            'Command "{}" is not a supported Tensorflow operation.'.format(command))
    return command

tf._command_guard = _command_guard


def _is_command_valid_guard(command, allowed):
    try:
        tf._command_guard(command, allowed)
    except RuntimeError:
        return False
    return True

tf._is_command_valid_guard = _is_command_valid_guard

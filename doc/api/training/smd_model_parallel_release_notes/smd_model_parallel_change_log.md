# Sagemaker Distributed Model Parallel 1.2.0 Release Notes

- New Features
- Bug Fixes
- Known Issues

## New Features

### PyTorch

#### Add support for PyTorch 1.7

- Adds support for `gradient_as_bucket_view` (PyTorch 1.7 only), `find_unused_parameters` (PyTorch 1.7 only) and `broadcast_buffers` options to `smp.DistributedModel`. These options behave the same as the corresponding options (with the same names) in
`torch.DistributedDataParallel` API. Please refer to the [SageMaker distributed model parallel API documentation](https://sagemaker.readthedocs.io/en/stable/api/training/smd_model_parallel_pytorch.html#smp.DistributedModel) for more information.

- Adds support for `join` (PyTorch 1.7 only) context manager, which is to be used in conjunction with an instance of `smp.DistributedModel` to be able to train with uneven inputs across participating processes.

- Adds support for `_register_comm_hook` (PyTorch 1.7 only) which will register the callable as a communication hook for DDP. NOTE: Like in DDP, this is an experimental API and subject to change.

## Bug Fixes

### PyTorch

- `Serialization`: Fix a bug with serialization/flattening where instances of subclasses of dict/OrderedDicts were serialized/deserialized or internally flattened/unflattened as
regular dicts.

### Tensorflow

- Fix a bug that may cause a hang during evaluation when there is no model input for one partition.

## Known Issues

### PyTorch

- A performance regression was observed when training on SMP with PyTorch 1.7.1 compared to 1.6. The rootcause was found to be the slowdown in performance of `.grad` method calls in PyTorch 1.7.1 compared to 1.6. Please see the related discussion: https://github.com/pytorch/pytorch/issues/50636.


# Sagemaker Distributed Model Parallel 1.1.0 Release Notes

- New Features
- Bug Fixes
- Improvements
- Performance
- Known Issues

## New Features

The following sections describe new feature releases that are common across frameworks and that are framework specific.

### Common across frameworks

#### Custom slicing support (`smp_slice` method) for objects passed to `smp.step` decorated functions

To pass an object to `smp.step` that contains tensors that needs to be split across
microbatches and is not an instance of list, dict, tuple or set, you should implement `smp_slice` method for the object.

Below is an example of how to use this with PyTorch:

```
 class CustomType:
     def __init__(self, tensor):
         self.data = tensor

     # SMP will call this to invoke slicing on the object passing in total microbatches (num_mb)
     # and the current microbatch index (mb).
     def smp_slice(self, num_mb, mb, axis):
         dim_size = list(self.data.size())[axis]

         split_size = dim_size // num_mb
         sliced_tensor = self.data.narrow(axis, mb * split_size, split_size)
         return CustomType(sliced_tensor, self.other)

custom_obj = CustomType(torch.ones(4,))

@smp.step()
def step(custom_obj):
    loss = model(custom_obj)
    model.backward(loss)
    return loss
```

### PyTorch

#### Add support for smp.DistributedModel.cpu()

`smp.DistributedModel.cpu()`
[allgather](https://sagemaker.readthedocs.io/en/stable/api/training/smd_model_parallel_common_api.html#smp.allgather)s
parameters and buffers across all `mp_ranks` and moves them to the CPU.

#### Add `trace_memory_usage` option to `smp.DistributedModel` to measure memory usage per module

Adds `trace_memory_usage` option to `smp.DistributedModel`. This attempts to measure memory usage per module during
tracing. If this is disabled, memory usage is estimated through the sizes of tensors returned from the module.
This option is disabled by default.

## Bug Fixes

### PyTorch

- `torch.nn.Sequential`: Fix a bug with `torch.nn.Sequential` which causes a failure with the error message : `shouldnt go less than 0, there is a bug` when the inputs to the first module don't require grads.

- `smp.DistributedModel`: Fix a bug with `DistributedModel` execution when a module has multiple parents. The bug surfaces with the error message: `actual_parent should be different than module_execution_stack parent only for torch.nn.ModuleList`

- `apex.optimizers.FusedNovoGrad`: Fix a bug with `apex.optimizers.FusedNovoGrad` which surfaces with the error message: `KeyError: 'exp_avg_sq'`

## Improvements

### Usability

#### PyTorch

- `smp.DistributedModel`: Improve the error message when the forward pass on `smp.DistributedModel` is called outside the `smp.step` decorated function.

- `smp.load`: Add user friendly error messages when loading checkpoints with `smp.load`.

### Partitioning Algorithm

#### PyTorch

- Better memory balancing by taking into account the existing modules already assigned to the parent, while partitioning the children of a given module.

## Performance

### Tensorflow

- Addresses long pre-processing times introduced by SMP XLA optimizer when dealing with large graphs and large number of microbatches. BERT (large) preprocessing time goes down from 40 minutes to 6 minutes on p3.16xlarge.

## Known Issues

### PyTorch

- Serialization for Torch in SMP overwrites instances of dict subclass to be dict itself, instead of the instances of subclass. One of the use cases which fails because of this issue is when a user implements a subclass of OrderedDict with the `__getitem__` method. After serialization/deserialization in SMP, indexing on the object will lead to errors. A workaround is to use the dict keys to access the underlying item.

--[[ Torch api for auto completion.
     Author: Pushpendre Rastogi
     Date: 4 July 2017
     License: Anyone is free to copy and modify the data below,
     but they should reproduce this header in their modified copy.
]]
return {
  nn = {
    type = "lib",
    description = "This package provides an easy and modular way to build and train simple or complex neural networks using",
    childs = {
      Container = {
	type = "class",
	description = [[ This is an abstract Module class which declares methods defined in all containers. It reimplements many of the Module methods such that calls are propagated to the contained modules. For example, a call to zeroGradParameters will be propagated to all contained modules.]],
	childs = {
	  add = {
	      type = "function",
	      args = ":(module)",
	      returns = "(container)",
	      description = "Adds a module to container"
	  },
	  get = {
	      type = "function",
	      args = "(index: Integer)",
	      returns = "(module: Module)",
	      description = "Returns the contained module at index"
	  },
	  size = {
	      type = "function",
	      args = ":()",
	      returns = "(size: Int)",
	      description = ""
	     },
	},
      },
      Sequential = {
	  type = "Container",
	  args = "",
	  returns = "",
	  description = "Plugs layers in a feed-forward fully connected manner."
      },
      Parallel = {
	  type = "Container",
	  args = "(inputDimension, outputDimension)",
	  returns = "",
	  description = [[Applies its ith child module to the ith slice of the input Tensor

Creates a container module that applies its `ith` child module to the  `ith` slice of the input Tensor by using [select](https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor-selectdim-index)
on dimension `inputDimension`. It concatenates the results of its contained modules together along dimension `outputDimension`.

Example:
```lua
mlp = nn.Parallel(2,1);   -- Parallel container will associate a module to each slice of dimension 2
                           -- (column space), and concatenate the outputs over the 1st dimension.

mlp:add(nn.Linear(10,3)); -- Linear module (input 10, output 3), applied on 1st slice of dimension 2
mlp:add(nn.Linear(10,2))  -- Linear module (input 10, output 2), applied on 2nd slice of dimension 2

                                  -- After going through the Linear module the outputs are
                                  -- concatenated along the unique dimension, to form 1D Tensor
> mlp:forward(torch.randn(10,2)) -- of size 5.
-0.5300
-1.1015
 0.7764
 0.2819
-0.6026
[torch.Tensor of dimension 5]
```

A more complicated example:
```lua

mlp = nn.Sequential();
c = nn.Parallel(1,2)     -- Parallel container will associate a module to each slice of dimension 1
                         -- (row space), and concatenate the outputs over the 2nd dimension.

for i=1,10 do            -- Add 10 Linear+Reshape modules in parallel (input = 3, output = 2x1)
 local t=nn.Sequential()
 t:add(nn.Linear(3,2))   -- Linear module (input = 3, output = 2)
 t:add(nn.Reshape(2,1))  -- Reshape 1D Tensor of size 2 to 2D Tensor of size 2x1
 c:add(t)
end

mlp:add(c)               -- Add the Parallel container in the Sequential container

pred = mlp:forward(torch.randn(10,3)) -- 2D Tensor of size 10x3 goes through the Sequential container
                                      -- which contains a Parallel container of 10 Linear+Reshape.
                                      -- Each Linear+Reshape module receives a slice of dimension 1
                                      -- which corresponds to a 1D Tensor of size 3.
                                      -- Eventually all the Linear+Reshape modules' outputs of size 2x1
                                      -- are concatenated alond the 2nd dimension (column space)
                                      -- to form pred, a 2D Tensor of size 2x10.

> pred
-0.7987 -0.4677 -0.1602 -0.8060  1.1337 -0.4781  0.1990  0.2665 -0.1364  0.8109
-0.2135 -0.3815  0.3964 -0.4078  0.0516 -0.5029 -0.9783 -0.5826  0.4474  0.6092
[torch.DoubleTensor of size 2x10]


for i = 1, 10000 do     -- Train for a few iterations
 x = torch.randn(10,3);
 y = torch.ones(2,10);
 pred = mlp:forward(x)

 criterion = nn.MSECriterion()
 local err = criterion:forward(pred,y)
 local gradCriterion = criterion:backward(pred,y);
 mlp:zeroGradParameters();
 mlp:backward(x, gradCriterion);
 mlp:updateParameters(0.01);
 print(err)
end
```

]]
      },
      Concat = {
	  type = "Container",
	  args = "(dim)",
	  returns = "",
	  description = [[ Concatenates in one layer several modules along dimension dim

Concat concatenates the output of one layer of "parallel" modules along the
provided dimension `dim`: they take the same inputs, and their output is
concatenated.

]]
      },
      NaN = {
	  type = "Container",
	  args = "",
	  returns = "",
	  description = "Decorates module to detect the source of NaN error"
      },
      Profile = {
	  type = "Container",
	  args = "",
	  returns = "",
	  description = "Decorates module to time its forward and backwards passes."
      },
      Jacobian = {
	type = "lib",
	description = [[ nn.Jacobian` class for testing the derivatives of their class, together with the [torch.Tester](https://github.com/torch/torch7/blob/master/doc/tester.md) class.  ]],
	childs = {
	  testJacobian = {args = "(module, input, minval, maxval, perturbation)", type = "function", returns = "", description = [[

Test the jacobian of a module w.r.t. to its input.

`module` takes as its input a random tensor shaped the same as `input`.
`minval` and `maxval` specify the range of the random tensor ([-2, 2] by default).
`perturbation` is used as finite difference (1e-6 by default).

Returns the L-inf distance between the jacobian computed by backpropagation and by finite difference.
]]
	  },
	  testJacobianParameters = {args = "(module, input, param, dparam, minval, maxval, perturbation)", type = "function", returns = "", description = [[

Test the jacobian of a module w.r.t. its parameters (instead of its input).

The input and parameters of `module` are random tensors shaped the same as `input` and `param`.
`minval` and `maxval` specify the range of the random tensors ([-2, 2] by default).
`dparam` points to the gradient w.r.t. parameters.
`perturbation` is used as finite difference (1e-6 by default).

Returns the L-inf distance between the jacobian computed by backpropagation and by finite difference.]]
	  },
	  testJacobianUpdateParameters = {args = "(module, input, param, minval, maxval, perturbation)", type = "function", returns = "", description = [[

Test the amount of update of a module to its parameters.

The input and parameters of `module` are random tensors shaped the same as `input` and `param`.
`minval` and `maxval` specify the range of the random tensors ([-2, 2] by default).
`perturbation` is used as finite difference (1e-6 by default).

Returns the L-inf distance between the update computed by backpropagation and by finite difference.

]]
	  },
	  forward = {args = "(module, input, param, perturbation)", type = "function", returns = "", description = [[

Compute the jacobian by finite difference.

`module` has parameters `param` and input `input`.
If provided, `param` is regarded as independent variables, otherwise `input` is the independent variables.
`perturbation` is used as finite difference (1e-6 by default).

Returns the jacobian computed by finite difference.
]]
	  },
	  backward = {args = "(module, input, param, dparam)", type = "function", returns = "", description = [[

Compute the jacobian by backpropagation.

`module` has parameters `param` and input `input`.
If provided, `param` is regarded as independent variables, otherwise `input` is the independent variables.
`dparam` is the gradient w.r.t. parameters, it must present as long as `param` is present.

Returns the jacobian computed by backpropagation.
]]
	  },
	},
      },
      HardTanh = {args = "([min_value, max_value[, inplace]])", type = "function", returns = "Module", description = [[ ]]},
      HardShrink = {args = "([lambda])",  type = "function", returns = "Module", description = [[ ]]},
      SoftShrink = {args = "([lambda])",  type = "function", returns = "Module", description = [[ ]]},
      SoftMax = {args = "()",  type = "function", returns = "Module", description = [[ ]]},
      SoftMin = {args = "()",  type = "function", returns = "Module", description = [[ ]]},
      SoftPlus = {args = "()",  type = "function", returns = "Module", description = [[ ]]},
      SoftSign = {args = "()",  type = "function", returns = "Module", description = [[ ]]},
      LogSigmoid = {args = "()",  type = "function", returns = "Module", description = [[ ]]},
      LogSoftMax = {args = "()",  type = "function", returns = "Module", description = [[ ]]},
      Sigmoid = {args = "()",  type = "function", returns = "Module", description = [[ ]]},
      Tanh = {args = "()",  type = "function", returns = "Module", description = [[ ]]},
      ReLU = {args = "([inplace])", type = "function", returns = "Module", description = [[ ]]},
      ReLU6 = {args = "([inplace])",  type = "function", returns = "Module", description = [[ ]]},
      PReLU = {args = "()",  type = "function", returns = "Module", description = [[ ]]},
      RReLU = {args = "([l, u[, inplace]])",  type = "function", returns = "Module", description = [[ ]]},
      CReLU = {args = "(nInputDims, [inplace])",  type = "function", returns = "Module", description = [[ ]]},
      ELU = {args = "([alpha[, inplace]])",  type = "function", returns = "Module", description = [[ ]]},
      LeakyReLU = {args = "([negval[, inplace]])", type = "function", returns = "Module", description = [[ ]]},
      SpatialSoftMax = {args = "()",  type = "function", returns = "Module", description = [[ ]]},
      AddConstant = {args = "(k[, inplace])", type = "function", returns = "Module", description = [[ ]]},
      MulConstant = {args = "(k[, inplace])",  type = "function", returns = "Module", description = [[ ]]},
      Module = {
	  type = "class",
	  description = [[ `Module` is an abstract class which defines fundamental methods necessary for a training a neural network. Modules are [serializable]. Modules contain two states variables: [output] and [gradInput]. ]],
	  childs = {
	    forward = {
		type = "function",
		args = "(input: Tensor)",
		returns = "(output: Tensor)",
		description = [[Takes an `input` object, and computes the corresponding `output` of the module. In general `input` and `output` are [Tensors].

It is not advised to override this function. Instead, one should
implement [updateOutput(input)](#nn.Module.updateOutput)
function. The forward module in the abstract parent class
[Module](#nn.Module) will call `updateOutput(input)`.]]
	    },
	    backward = {
		type = "function",
		args = "(input, gradOutput)",
		returns = "(gradInput)",
		description = [[A _backpropagation step_ consist in computing two kind of gradients
at `input` given `gradOutput` (gradients with respect to the
output of the module).  This function simply performs this task using
two function calls:

  - A function call to [updateGradInput(input, gradOutput)](#nn.Module.updateGradInput).
  - A function call to accGradParameters(input,gradOutput,scale) ]]
	    },
	    updateOutput = {
		type = "function",
		args = "(input: Tensor)",
		returns = "(t: Tensor)",
		description = [[Computes the output using the current parameter set of the class and input. This function returns the result which is stored in the [output] field.]]
	    },
	    updateGradInput = {
		type = "function",
		args = "(input, gradOutput)",
		returns = "",
		description = [[Computing the gradient of the module with respect to its own
input. This is returned in `gradInput`. Also, the [gradInput] state variable is updated accordingly. ]]
	       },
	    accGradParameters = {
		type = "function",
		args = "",
		returns = "(input, gradOutput, scale)",
		description = [[Computing the gradient of the module with respect to its
own parameters. Many modules do not perform this step as they do not
have any parameters. The state variable name for the parameters is
module dependent. The module is expected to _accumulate_ the
gradients with respect to the parameters in some variable. This function works in tandem with `zeroGradParameters` and `updateParameters`]]
	    },
	    zeroGradParameters = {
		type = "function",
		args = "()",
		returns = "()",
		description = "Zero the accumulated gradients"
	    },
	    updateParameters = {
		type = "function",
		args = "(learningRate: Numeric)",
		returns = "()",
		description = [[If the module has parameters, this will update these parameters, according
to the accumulation of the gradients with respect to these parameters,
accumulated through [backward()](#nn.Module.backward) calls.]]
	    },
	    parameters = {
		type = "function",
		args = "()",
		returns = "[{weights}, {gradWeights}]",
		description = [[This function should returns two tables. One for the learnable
parameters `{weights}` and another for the gradients of the energy
wrt to the learnable parameters `{gradWeights}`.

Custom modules should override this function if they use learnable
parameters that are stored in tensors.
]]
	    },
	    findModules = {
		type = "function",
		args = "(typename)",
		returns = "{List of Modules}",
		description = [[Find all instances of modules in the network of a certain `typename`.  It returns a flattened list of the matching nodes, as well as a flattened list of the container modules for each matching node.

Modules that do not have a parent container (ie, a top level nn.Sequential for instance) will return their `self` as the container.

This function is very helpful for navigating complicated nested networks.  For example, a didactic example might be; if you wanted to print the output size of all `nn.SpatialConvolution` instances:

```lua
-- Construct a multi-resolution convolution network (with 2 resolutions):
model = nn.ParallelTable()
conv_bank1 = nn.Sequential()
conv_bank1:add(nn.SpatialConvolution(3,16,5,5))
conv_bank1:add(nn.Threshold())
model:add(conv_bank1)
conv_bank2 = nn.Sequential()
conv_bank2:add(nn.SpatialConvolution(3,16,5,5))
conv_bank2:add(nn.Threshold())
model:add(conv_bank2)
-- FPROP a multi-resolution sample
input = {torch.rand(3,128,128), torch.rand(3,64,64)}
model:forward(input)
-- Print the size of the Threshold outputs
conv_nodes = model:findModules('nn.SpatialConvolution')
for i = 1, #conv_nodes do
  print(conv_nodes[i].output:size())
end
```

Another use might be to replace all nodes of a certain `typename` with another.  For instance, if we wanted to replace all `nn.Threshold` with `nn.Tanh` in the model above:

```lua
threshold_nodes, container_nodes = model:findModules('nn.Threshold')
for i = 1, #threshold_nodes do
  -- Search the container for the current threshold node
  for j = 1, #(container_nodes[i].modules) do
    if container_nodes[i].modules[j] == threshold_nodes[i] then
      -- Replace with a new instance
      container_nodes[i].modules[j] = nn.Tanh()
    end
  end
end
```
]]
	    },
	    apply = {
		type = "function",
		args = ":(f::Module -> Void)",
		returns = "()",
		description = [[Calls provided function on itself and all child modules. This function takes
module to operate on as a first argument:

```lua
model:apply(function(module)
   module.train = true
end)
```

In the example above `train` will be set to to `true` in all modules of `model`.
This is how `training()` and `evaluate()` functions implemented.]]
	    },
	    replace = {
		type = "function",
		args = ":(f::Module -> Void)",
		returns = "",
		description = [[Similar to apply takes a function which applied to all modules of a model,
but uses return value to replace the module. Can be used to replace all
modules of one type to another or remove certain modules.

For example, can be used to remove `nn.Dropout` layers by replacing them with
`nn.Identity`:

```lua
model:replace(function(module)
   if torch.typename(module) == 'nn.Dropout' then
      return nn.Identity()
   else
      return module
   end
end)
```
]]
	    },
	  },
      },
      Linear = { type = "Module", returns = "Module", args = "(inputDimension, outputDimension, [bias = true])", description = [[ ]]},
      LinearWeightNorm = { type = "Module", returns = "Module", args = "(inputDimension, outputDimension, [bias = true])", description = [[LinearWeightNorm implements the reparametrization presented in [Weight Normalization](https://arxiv.org/pdf/1602.07868v3.pdf), which decouples the length of neural network weight vectors from their direction. The weight vector `w` is determined instead by parameters `g` and `v` such that `w = g * v / ||v||`, where `||v||` is the euclidean norm of vector `v`. In all other respects this layer behaves like `nn.Linear`.

To convert between `nn.Linear` and `nn.LinearWeightNorm` you can use the `nn.LinearWeightNorm.fromLinear(linearModule)` and `weightNormModule:toLinear()` functions.
 ]]
      },
      SparseLinear = { type = "Module", returns = "Module", args = "(inputDimension, outputDimension)", description = [[Applies a linear transformation to the incoming sparse data, i.e. `y = Ax + b`. The `input` tensor given in `forward(input)` must be a sparse vector represented as 2D tensor of the form torch.Tensor(N, 2) where the pairs represent indices and values. The first column contains indices, the second column contains values in a a vector where all other elements are zeros.

The SparseLinear layer is useful when the number of input dimensions is very large and the input data is sparse.
 ]]},
IndexLinear = { type = "Module", returns = "Module", args = "(inputSize, outputSize, doGradInput, keysOffset, weight, bias, normalize)", description = [[ Applies the following transformation to the incoming (optionally) normalized sparse input data:
`z = Weight * y + bias`, where
- `y_i = normalize and (x_i *  (1 / x_i_max) + b_i) or x_i`
- `x_i` is the `i'th` feature of the input,
- `b_i` is a per-feature bias,
- `x_i_max` is the maximum absolute value seen so far during training for feature `i`.

The normalization of input features is very useful to avoid explosions during training if sparse input values are really high. It also helps ditinguish between the presence and the absence of a given feature.
- `inputSize` is the maximum number of features.
- `outputSize` is the number of output neurons.
- `doGradInput`, if  `false` (the default), the gradInput will not be computed.
- `keysOffset` lets you specify input keys are in the `[1+keysOffset, N+keysOffset]` range. (defaults to `0`)
- `weight` and `bias` allow you to create the module with existing weights without using additional memory.
  When passing `weight` and `bias`, `inputSize` and `outputSize` are inferred from the weights.
- `normalize` will activate the normalization of the input feature values. (`false` by default)

#### Differences from SparseLinear ####
- The layout of `weight` is transposed compared to `SparseLinear`. This was done for performance considerations.
- The `gradWeight` that is computed for in-place updates is a sparse representation of the whole gradWeight matrix. Its size changes from one
backward pass to another. This was done for performance considerations.
- The input format differs from the [SparseLinear](#nn.SparseLinear) input format by accepting keys and values as a table of tensors. This enables `IndexLinear` to have a larger range for keys than `SparseLinear`.

The `input` tensors must be in one of the following formats.

- An array of size 2 containing a batch of `keys` followed by a batch of `values`.
```lua
x = {
      { torch.LongTensor({ 1, 200 }), torch.LongTensor({ 100, 200, 1000 }) },
      { torch.Tensor({ 1, 0.1 }), torch.Tensor({ 10, 0.5, -0.5 }) }
}
```

- an array of size 3 containing a flattened (pre-concatenated) batch of `keys`, followed by `values`, and `sizes`.
```lua
-- Equivalent to the input shown above
x = {
      torch.LongTensor({ 1, 200, 100, 200, 1000 }),
      torch.Tensor({ 1, 0.1, 10, .5, -0.5 }),
      torch.LongTensor({ 2, 3 })
}
```
]]},
Bilinear = { type = "Module", returns = "Module", args = "(inputDimension1, inputDimension2, outputDimension, [bias = true])", description = [[Applies a bilinear transformation to the incoming data, i.e. `\forall k: y_k = x_1 A_k x_2 + b`. The `input` tensor given in `forward(input)` is a table containing both inputs `x_1` and `x_2`, which are tensors of size `N x inputDimension1`
and `N x inputDimension2`, respectively. The layer can be trained without biases by setting `bias = false`.
 ]]},
PartialLinear = { type = "Module", returns = "Module", args = "(inputSize, outputSize, [bias = true])", description = [[ PartialLinear is a Linear layer that allows the user to a set a collection of
column indices. When the column indices are set, the layer will behave like a
Linear layer that only has those columns. Meanwhile, all parameters are
preserved, so resetting the PartialLinear layer will result in a module that
behaves just like a regular Linear layer.

This module is useful, for instance, when you want to do forward-backward on
only a subset of a Linear layer during training but use the full Linear layer
at test time.

You can create a layer in the following way:

```lua
 module = nn.PartialLinear(5, 3)  -- 5 inputs, 3 outputs
```

Input data for this layer would look as follows:
```lua
 input = torch.randn(128, 5)  -- 128 input examples
 module:forward(input)
```

One can set the partition of indices to compute using the function `setPartition(indices)` where `indices` is a tensor containing the indices to compute.
```lua
module = nn.PartialLinear(5, 3)  -- 5 inputs, 3 outputs
module:setPartition(torch.Tensor({2,4})) -- only compute the 2nd and 4th indices out of a total of 5 indices
```

One can reset the partition via the `resetPartition()` function that resets the partition to compute all indices, making it's behaviour equivalent to `nn.Linear`

]]},
Dropout = { type = "Module", returns = "Module", args = "(p=0.5)", description = [[ During training, `Dropout` masks parts of the `input` using binary samples from a [bernoulli](http://en.wikipedia.org/wiki/Bernoulli_distribution) distribution.
Each `input` element has a probability of `p` of being dropped, i.e having its commensurate output element be zero. This has proven an effective technique for regularization and preventing the co-adaptation of neurons (see [Hinton et al. 2012](http://arxiv.org/abs/1207.0580)).

Furthermore, the outputs are scaled by a factor of `1/(1-p)` during training. This allows the `input` to be simply forwarded as-is during evaluation.
]]},
Add = { type = "Module", returns = "Module", args = "(inputDimension, scalar)", description = [[Applies a bias term to the incoming data, i.e. `yi = x_i + b_i`,  or if `scalar = true` then uses a single bias term, `yi = x_i + b`. So if `scalar = true` then `inputDimension` value will be disregarded.

Full Example of training the bias
```lua
y = torch.Tensor(5)
mlp = nn.Sequential()
mlp:add(nn.Add(5))

function gradUpdate(mlp, x, y, criterion, learningRate)
   local pred = mlp:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   mlp:zeroGradParameters()
   mlp:backward(x, gradCriterion)
   mlp:updateParameters(learningRate)
   return err
end

for i = 1, 10000 do
   x = torch.rand(5)
   y:copy(x);
   for i = 1, 5 do y[i] = y[i] + i; end
   err = gradUpdate(mlp, x, y, nn.MSECriterion(), 0.01)
end

print(mlp:get(1).bias)
```

gives the output:

```lua
 1.0000
 2.0000
 3.0000
 4.0000
 5.0000
[torch.Tensor of dimension 5]
```


]]},
CAdd = { type = "Module", returns = "Module", args = "(size)", description = [[Applies a component-wise addition to the incoming data, i.e. `y_i = x_i + b_i`. Argument `size` can be one or many numbers (sizes) or a `torch.LongStorage`. For example, `nn.CAdd(3,4,5)` is equivalent to `nn.CAdd(torch.LongStorage{3,4,5})`. If the size for a particular dimension is 1, the addition will be expanded along the entire axis.
 ]]},
Mul = { type = "Module", returns = "Module", args = "()", description = [[Applies a _single_ scaling factor to the incoming data, i.e. `y = w x`, where `w` is a scalar. ]]},
CMul = { type = "Module", returns = "Module", args = "(size)", description = [[Applies a component-wise multiplication to the incoming data, i.e. `y_i = w_i * x_i`. Argument `size` can be one or many numbers (sizes) or a `torch.LongStorage`. For example, `nn.CMul(3,4,5)` is equivalent to `nn.CMul(torch.LongStorage{3,4,5})`.
If the size for a particular dimension is 1, the multiplication will be expanded along the entire axis. ]]},
Max = { type = "Module", returns = "Module", args = "(dimension, nInputDim)", description = [[Applies a max operation over dimension `dimension`.
Hence, if an `nxpxq` Tensor was given as input, and `dimension` = `2` then an `nxq` matrix would be output.
When `nInputDim` is provided, inputs larger than that value will be considered batches where the actual `dimension` to apply the max operation will be dimension `dimension + 1`.
 ]]},
Min = { type = "Module", returns = "Module", args = "(dimension, nInputDim)", description = [[ Applies a min operation over dimension `dimension`.
Hence, if an `nxpxq` Tensor was given as input, and `dimension` = `2` then an `nxq` matrix would be output.
When `nInputDim` is provided, inputs larger than that value will be considered batches where the actual `dimension` to apply the min operation will be dimension `dimension + 1`.
]]},
Mean = { type = "Module", returns = "Module", args = "(dimension, nInputDim)", description = [[Applies a mean operation over dimension `dimension`.
Hence, if an `nxpxq` Tensor was given as input, and `dimension` = `2` then an `nxq` matrix would be output.
When `nInputDim` is provided , inputs larger than that value will be considered batches where the actual `dimension` to apply the sum operation will be dimension `dimension + 1`.
This module is based on [nn.Sum](#nn.Sum).
 ]]},
Sum = { type = "Module", returns = "Module", args = "(dimension, nInputDim, sizeAverage, squeeze)", description = [[Applies a sum operation over dimension `dimension`.
Hence, if an `nxpxq` Tensor was given as input, and `dimension` = `2` then an `nxq` matrix would be output. If argument `squeeze` is set to `false` then the output would be of size `nx1xq`.
When `nInputDim` is provided , inputs larger than that value will be considered batches where the actual `dimension` to apply the sum operation will be dimension `dimension + 1`.
Negative indexing is allowed by providing a negative value to `nInputDim`.
When `sizeAverage` is provided, the sum is divided by the size of the input in this `dimension`. This is equivalent to the mean operation performed by the [nn.Mean](#nn.Mean) module. ]]},
Euclidean = { type = "Module", returns = "Module", args = "(inputSize,outputSize)", description = [[Outputs the Euclidean distance of the input to `outputSize` centers, i.e. this layer has the weights `w_j`,  for `j` = `1`,..,`outputSize`, where `w_j` are vectors of dimension `inputSize`.

The distance `y_j` between center `j` and input `x` is formulated as `y_j = || w_j - x ||`.
 ]]},
WeightedEuclidean = { type = "Module", returns = "Module", args = "(inputSize,outputSize)", description = [[ This module is similar to [Euclidean](#nn.Euclidean), but additionally learns a separate diagonal covariance matrix across the features of the input space _for each center_.

In other words, for each of the `outputSize` centers `w_j`, there is a diagonal covariance matrices `c_j`, for `j` = `1`,..,`outputSize`, where `c_j` are stored as vectors of size `inputSize`.

The distance `y_j` between center `j` and input `x` is formulated as `y_j = || c_j * (w_j - x) ||`.]]},
Cosine = { type = "Module", returns = "Module", args = "(inputSize,outputSize)", description = [[This module is similar to [Euclidean](#nn.Euclidean), but additionally learns a separate diagonal covariance matrix across the features of the input space _for each center_.

In other words, for each of the `outputSize` centers `w_j`, there is a diagonal covariance matrices `c_j`, for `j` = `1`,..,`outputSize`, where `c_j` are stored as vectors of size `inputSize`.

The distance `y_j` between center `j` and input `x` is formulated as `y_j = || c_j * (w_j - x) ||`. ]]},
Kmeans = {type = "Module", returns = "Module", args = "(k, dim)", description = [[`k` is the number of centroids and `dim` is the dimensionality of samples.
The `forward` pass computes distances with respect to centroids and returns index of closest centroid.
Centroids can be updated using gradient descent.
Centroids can be initialized randomly or by using [kmeans++](https://en.wikipedia.org/wiki/K-means%2B%2B) algoirthm:

```lua
km:initRandom(samples) -- Randomly initialize centroids from input samples.
km:initKmeansPlus(samples) -- Use Kmeans++ to initialize centroids.
```
Example showing how to use Kmeans module to do standard Kmeans clustering.

```lua
attempts = 10
iter = 100 -- Number of iterations
bestKm = nil
bestLoss = math.huge
learningRate = 1
for j=1, attempts do
   local km = nn.Kmeans(k, dim)
   km:initKmeansPlus(samples)
   for i=1, iter do
      km:zeroGradParameters()
      km:forward(samples) -- sets km.loss
      km:backward(samples, gradOutput) -- gradOutput is ignored

      -- Gradient Descent weight/centroids update
      km:updateParameters(learningRate)
   end

   if km.loss < bestLoss then
      bestLoss = km.loss
      bestKm = km:clone()
   end
end
```
`nn.Kmeans()` module maintains loss only for the latest forward. If you want to maintain loss over the whole dataset then you who would need do it my adding the module loss for every forward.

You can also use `nn.Kmeans()` as an auxillary layer in your network.
A call to `forward` will generate an `output` containing the index of the nearest cluster for each sample in the batch.
The `gradInput` generated by `updateGradInput` will be zero.
]]},
Identity = { type = "Module", returns = "Module", args = "()", description = [[
```lua
pred_mlp = nn.Sequential()  -- A network that makes predictions given x.
pred_mlp:add(nn.Linear(5, 4))
pred_mlp:add(nn.Linear(4, 3))
xy_mlp = nn.ParallelTable() -- A network for predictions and for keeping the
xy_mlp:add(pred_mlp)        -- true label for comparison with a criterion
xy_mlp:add(nn.Identity())   -- by forwarding both x and y through the network.
mlp = nn.Sequential()       -- The main network that takes both x and y.
mlp:add(xy_mlp)             -- It feeds x and y to parallel networks;
mlp:add(                    -- and then applies the criterion.
  nn.CriterionTable(
    nn.MSECriterion()))
for i = 1, 100 do           -- Do a few training iterations
   x = torch.ones(5)        -- Make input features.
   y = torch.Tensor(3)
   y:copy(x:narrow(1,1,3))  -- Make output label.
   err = mlp:forward{x,y}   -- Forward both input and output.
   print(err)               -- Print error from criterion.

   mlp:zeroGradParameters() -- Do backprop...
   mlp:backward({x, y})
   mlp:updateParameters(0.05)
end
```
]]},
Copy = { type = "Module", returns = "Module", args = "(inputType, outputType, [forceCopy, dontCast])", description = [[This layer copies the input to output with type casting from `inputType` to `outputType`. Unless `forceCopy` is true, when the first two arguments are the same, the input isn't copied, only transferred as the output.
The default `forceCopy` is false.
When `dontCast` is true, a call to `nn.Copy:type(type)` will not cast the module's `output` and `gradInput` `Tensor`s to the new type.
The default is false. ]]},
Narrow = { type = "Module", returns = "Module", args = "(dimension, offset, length)", description = [[ ]]},
Replicate = { type = "Module", returns = "Module", args = "(nFeature [, dim, ndim])", description = [[This class creates an output where the input is replicated `nFeature` times along dimension `dim` (default 1).
There is no memory allocation or memory copy in this module.
It sets the [stride](https://github.com/torch/torch7/blob/master/doc/tensor.md#torch.Tensor.stride) along the `dim`th dimension to zero.
When provided, `ndim` should specify the number of non-batch dimensions.
This allows the module to replicate the same non-batch dimension `dim` for both batch and non-batch `inputs`. ]]},
Reshape = { type = "Module", returns = "Module", args = "(dimension1, dimension2, ... [, batchMode])", description = [[Reshapes an `nxpxqx..` `Tensor` into a `dimension1xdimension2x...` `Tensor`, taking the elements row-wise.

The optional last argument `batchMode`, when `true` forces the first dimension of the input to be considered the batch dimension, and thus keep its size fixed.
This is necessary when dealing with batch sizes of one.
When `false`, it forces the entire input (including the first dimension) to be reshaped to the input size.
Default `batchMode=nil`, which means that the module considers inputs with more elements than the produce of provided sizes, i.e. `dimension1xdimension2x...`, to be batches. ]]},
View = { type = "Module", returns = "Module", args = "(sizes)", description = [[ ]]},
Contiguous = { type = "Module", returns = "Module", args = "()", description = [[ ]]},
Select = { type = "Module", returns = "Module", args = "(dim, index)", description = [[ ]]},
MaskedSelect = { type = "Module", returns = "Module", args = "()", description = [[ ]]},
Index = { type = "Module", returns = "Module", args = "(dim)", description = [[ ]]},
Squeeze = { type = "Module", returns = "Module", args = "([dim, numInputDims])", description = [[ ]]},
Unsqueeze = { type = "Module", returns = "Module", args = "(pos [, numInputDims])", description = [[Insert singleton dim (i.e., dimension 1) at position `pos`.
For an `input` with `dim = input:dim()`, there are `dim + 1` possible positions to insert the singleton dimension.
For example, if `input` is `3` dimensional `Tensor` in size `p x q x r`, then the singleton dim can be inserted at the following `4` positions
```
pos = 1: 1 x p x q x r
pos = 2: p x 1 x q x r
pos = 3: p x q x 1 x r
pos = 4: p x q x r x 1
``` ]]},
Transpose = { type = "Module", returns = "Module", args = "({dim1, dim2} [, {dim3, dim4}, ...])", description = [[Swaps dimension `dim1` with `dim2`, then `dim3` with `dim4`, and so on. So

```lua
nn.Transpose({dim1, dim2}, {dim3, dim4}):forward(t)
```

gives the same output as

```lua
t:transpose(dim1, dim2)
t:transpose(dim3, dim4)
```

The method `setNumInputDims()` allows to specify the expected number of dimensions of the inputs of the modules. This makes it possible to use minibatch inputs.  ]]},
Exp = { type = "Module", returns = "Module", args = "()", description = [[ ]]},
Log = { type = "Module", returns = "Module", args = "()", description = [[ ]]},
Square = { type = "Module", returns = "Module", args = "()", description = [[ ]]},
Sqrt = { type = "Module", returns = "Module", args = "()", description = [[ ]]},
Power = { type = "Module", returns = "Module", args = "(p)", description = [[ ]]},
Clamp = { type = "Module", returns = "Module", args = "(min_value, max_value)", description = [[ ]]},
Normalize = { type = "Module", returns = "Module", args = "(p, [eps])", description = [[ ]]},
MM = { type = "Module", returns = "Module", args = "(transA, transB)", description = [[ ]]},
BatchNormalization = { type = "Module", returns = "Module", args = "(N [, eps] [, momentum] [,affine])", description = [[where `N` is the dimensionality of input
`eps` is a small value added to the standard-deviation to avoid divide-by-zero. Defaults to `1e-5`.
`affine` is a boolean. When set to false, the learnable affine transform is disabled. Defaults to true

During training, this layer keeps a running estimate of its computed mean and std.
The running sum is kept with a default momentum of 0.1 (unless over-ridden)
During evaluation, this running mean/std is used for normalization.

Implements Batch Normalization as described in [the paper](http://arxiv.org/pdf/1502.03167v3.pdf): "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" by Sergey Ioffe, Christian Szegedy.

The operation implemented is:

```lua
              x - mean(x)
y =  ----------------------------- * gamma + beta
      standard-deviation(x) + eps
```

where the mean and standard-deviation are calculated per-dimension over the mini-batches and where gamma and beta are learnable parameter vectors of size `N` (where `N` is the input size).
The learning of gamma and beta is optional.
The module only accepts 2D inputs. ]]},
PixelShuffle = { type = "Module", returns = "Module", args = "(r)", description = [[ ]]},
Padding = { type = "Module", returns = "Module", args = "(dim, pad [, nInputDim, value, index])", description = [[ ]]},
L1Penalty = { type = "Module", returns = "Module", args = "(L1weight, sizeAverage)", description = [[L1Penalty is an inline module that in its forward propagation copies the input Tensor directly to the output, and computes an L1 loss of the latent state (input) and stores it in the module's `loss` field.
During backward propagation: `gradInput = gradOutput + gradLoss`.

This module can be used in autoencoder architectures to apply L1 losses to internal latent state without having to use Identity and parallel containers to carry the internal code to an output criterion.

Example (sparse autoencoder, note: decoder should be normalized):

```lua
encoder = nn.Sequential()
encoder:add(nn.Linear(3, 128))
encoder:add(nn.Threshold())
decoder = nn.Linear(128, 3)

autoencoder = nn.Sequential()
autoencoder:add(encoder)
autoencoder:add(nn.L1Penalty(l1weight))
autoencoder:add(decoder)

criterion = nn.MSECriterion()  -- To measure reconstruction error
-- ...
``` ]]},
GradientReversal = { type = "Module", returns = "Module", args = "([lambda = 1])", description = [[This module preserves the input, but takes the gradient from the subsequent layer, multiplies it by `-lambda` and passes it to the preceding layer. This can be used to maximise an objective function whilst using gradient descent, as described in "Domain-Adversarial Training of Neural Networks" (http://arxiv.org/abs/1505.07818).
 ]]},
TemporalDynamicKMaxPooling = { type = "Module", returns = "Module", args = "(minK, [factor])", description = [[ ]]},
Constant = { type = "Module", returns = "Module", args = "(value, nInputDim)", description = [[ ]]},
WhiteNoise = { type = "Module", returns = "Module", args = "([mean, stdev])", description = [[ ]]},
OneHot = { type = "Module", returns = "Module", args = "(outputSize)", description = [[ ]]},
PrintSize = { type = "Module", returns = "Module", args = "(name)", description = [[This module is useful for debugging complicated module composites.
It prints the size of the `input` and `gradOutput` during `forward`
and `backward` propagation respectively.
The `name` is a string used to identify the module along side the printed size.
 ]]},
ZeroGrad = { type = "Module", returns = "Module", args = "()", description = [[The module zeros the `gradInput` but forwards the `input` as-is. ]]},
Collapse = { type = "Module", returns = "Module", args = "(nInputDim)", description = [[It collapses all non-batch dimensions. This is useful for converting
a spatial feature map to the single dimension required by a dense
hidden layer like Linear.

This module is the equivalent of:
```
view = nn.View(-1)
view:setNumInputDim(nInputDim)
```]]},
Convert = { type = "Module", returns = "Module", args = "([inputShape, outputShape])", description = [[ ]]},
    },
  },
}

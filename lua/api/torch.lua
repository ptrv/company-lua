--[[ Torch api for auto completion.
     Author: Pushpendre Rastogi
     Date: 4 July 2017
     License: Anyone is free to copy and modify the data below,
     but they should reproduce this header in their modified copy.
]]
return {
  torch = {
    type = "lib",
    description = [[__Torch__ is the main package in [Torch7](http://torch.ch) where data
structures for multi-dimensional tensors and mathematical operations
over these are defined. Additionally, it provides many utilities for
accessing files, serializing objects of arbitrary types and other
useful utilities.]],
    childs = {
      setdefaulttensortype = {
	  type = "function",
	  args = [[(s: String 'torch.[Float|Double|Int]Tensor')]],
	  returns = "()",
	  description = "Set the default tensor type throughout torch."
      },
      isTensor = {
	  type = "function",
	  args = "(t:Any)",
	  returns = "(v: Boolean)",
	  description = "Is Tensor?"
      },
      randn = {
	  type = "function",
	  args = "(x: Int, y: Int)",
	  returns = "(t: Tensor)",
	  description = "Returns a random tensor with standard guassian val"
      },
      range = {
	  type = "function",
	  args = "(start: Int, end: Int)",
	  returns = "(t: Tensor)",
	  description = "1d Range"
      },
      save = {
	  type = "function",
	  args = "(filename: String, obj: Object, [format: '(ascii|binary)', referenced: Boolean]",
	  returns = "()",
	  description = [[Writes `object` into a file named `filename`. The `format` can be set to
`ascii` or `binary` (default is binary). Binary format is platform
dependent, but typically more compact and faster to read/write. The ASCII
format is platform-independent, and should be used to share data structures
across platforms. The option `referenced` specifies if
[object references] should be tracked or not (`true` by default).


Sets the referenced property of the File to `ref`. `ref` has to be `true`
or `false`.

By default `ref` is true, which means that a File object keeps track of
objects written (using [writeObject](#torch.File.writeObject) method) or
read (using [readObject](#torch.File.readObject) method). Objects with the
same address will be written or read only once, meaning that this approach
preserves shared memory structured.

Keeping track of references has a cost: every object which is serialized in
the file is kept alive (even if one discards the object after
writing/reading) as File needs to track their pointer. This is not always a
desirable behavior, especially when dealing with large data structures.

Another typical example when does not want reference tracking is when
one needs to push the same tensor repeatedly into a file but every time
changing its contents: calling `referenced(false)` ensures desired
behaviour.]]
      },
      load = {
	  type = "function",
	  args = "(filename: String, [format, referenced])",
	  returns = "(obj: Any Type)",
	  description = [[Reads `object` from a file named `filename`.
The `format` can be set to `ascii`, `binary`, `b32` or `b64` (default is binary).
Binary format is platform dependent, but typically more compact and faster to read/write.
Use `b32`/`b64`, instead of `binary`, for loading files saved on a 32/64 bit OS.
The ASCII format is platform-independent, and may be used to share data structures across platforms.

The option `referenced` specifies if [object references](file.md#torch.File.referenced) should be tracked or not (`true` by default).
Note that files written with `referenced` at `true` cannot be loaded with `referenced` at `false`.]]
      },
      seed = {
	  type = "function",
	  args = "([rng: RNG])",
	  returns = "(seed: Long)",
	  description = [[Set the seed of the random number generator using `/dev/urandom`
(on Windows the time of the computer with granularity of seconds is used).
Returns the seed obtained. ]]
	 },
      manualSeed = {
	  type = "function",
	  args = "([rng: RNG, ] num: Long)",
	  returns = "()",
	  description = "Seed rng provided, or the global one, using `num`."
      },
      rand = {
	  type = "function",
	  args = "",
	  returns = "",
	  description = ""
      },
      randn = {
	  type = "function",
	  args = "",
	  returns = "",
	  description = ""
      },
      randperm = {
	  type = "function",
	  args = "(n: Long)",
	  returns = "(t: Tensor)",
	  description = "returns a random permutation of integers from 1 to `n`."
      },
      Generator = {
	  type = "function",
	  args = "()",
	  returns = "(rng: RNG)",
	  description = [[Creates a non-global random generator that carries its own state and can be
passed as the first argument to any function that generates a random number.]]
	 },
      serialize = {
	  type = "function",
	  args = "(object: Any Object, [format: String])",
	  returns = "(s: String)",
	  description = "Serializes `object` into a string."
      },
      deserialize = {
	  type = "function",
	  args = "(object: Object, [format: String])",
	  returns = "(o: AnyObject)",
	  description = ""
      },
      random = {
	  type = "function",
	  args = "([rng, start=1, end=2^32])",
	  returns = "(r: UInt32)",
	  description = "Return random int from [start, end)."
      },
      uniform = {
	  type = "function",
	  args = "([gen, a=0, b=1])",
	  returns = "(r: Double)",
	  description = [[Returns a random real number according to uniform distribution on `[a,b)`.]]
      },
      normal = {
	  type = "function",
	  args = "([gen, mean=0, stdev=1])",
	  returns = "(r: Double)",
	  description = ""
      },
      exponential = {
	  type = "function",
	  args = "([gen,]  lambda)",
	  returns = "(r: Double)",
	  description = ""
      },
      cauchy = {
	  type = "function",
	  args = "([gen,] median, sigma)",
	  returns = "(r: Double)",
	  description = [[Returns a random real number according to the Cauchy distribution
`p(x) = sigma/(pi*(sigma^2 + (x-median)^2))`]]
      },
      geometric = {
	  type = "function",
	  args = "([gen,] p)",
	  returns = "",
	  description = ""
      },
      bernoulli = {
	  type = "function",
	  args = "([gen, p=0.5])",
	  returns = "",
	  description = ""
      },
      Timer = {
	type = "class",
	description = [[This class is able to measure time (in seconds) elapsed in a particular period. Example:
```lua
  timer = torch.Timer() -- the Timer starts to count now
  x = 0
  for i=1,1000000 do
    x = x + math.sin(x)
  end
  print('Time elapsed for 1,000,000 sin: ' .. timer:time().real .. ' seconds')
```
]],
	childs = {
	  reset = {
		type = "function",
		args = ":()",
		returns = ":()",
		description = ""
	  },
	  resume = {
	      type = "function",
	      args = ":()",
	      returns = "()",
	      description = ""
	  },
	  stop = {
	      type = "function",
	      args = ":()",
	      returns = "()",
	      description = ""
	  },
	  time = {
	      type = "function",
	      args = ":()",
	      returns = ":(b: Table)",
	      description = [[Returns a table reporting the accumulated time elapsed until now. Following the UNIX shell `time` command,
there are three fields in the table:
  * `real`: the wall-clock elapsed time.
  * `user`: the elapsed CPU time. Note that the CPU time of a threaded program sums time spent in all threads.
  * `sys`: the time spent in system usage.]]
	     },
	},
      },
      Tensor = {
        type = "class",
        description = [[A `Tensor` is a multi-dimensional matrix. The number of
dimensions is unlimited, up to what can be created using
[LongStorage](storage.md). The elements in the same row , i.e. along the last dimension, are contiguous in memory for a tensor.]],
        args = "(n: Integer, m: Integer, ...)",
        childs = {
	  nDimension = {
	      type = "function",
	      args = ":()",
	      returns = "(nDim: Integer)",
	      description = "Return the number of dimensions in tensor"
	  },
	  dim = {
	      type = "function",
	      args = ":()",
	      returns = "(nDim: Integer)",
	      description = "Return the number of dimensions in tensor"
	  },
	  size = {
	      type = "function",
	      args = "(i: Integer)",
	      returns = "(size: Integer)",
	      description = "Return the size along a dimension."
	  },
	  apply = {
	      type = "function",
	      args = ":(f::(x:Numeric -> y:Numeric): function)",
	      returns = "()",
	      description = "Elementwise apply function inplace"
	  },
	  zero = {
	      type = "function",
	      args = ":()",
	      returns = "()",
	      description = "Zero all elements of tensor inplace."
	  },
	  fill = {
	      type = "function",
	      args = ":(x: Numeric)",
	      returns = "()",
	      description = "Fill tensor with x"
	  },
	  narrow = {
	      type = "function",
	      args = ":(dim:Int, index:Int, size:Int)",
	      returns = "(t: Tensor)",
	      description = "Returns view of a slice of original tensor along dimension `dim` from `index` to `index+size` EXCLUSIVE."
	  },
	  sub = {
	      type = "function",
	      args = ":(d1_start:Int, d1_end:Integer, ...)",
	      returns = "(t:Tensor)",
	      description = [[Returns a sub-tensor view of original tensor with di_start and di_end indices (INCLUSIVE)
> x = torch.Tensor(5, 6):zero()
> y = x:sub(2,4):fill(1)
> print(x)               -- x has been modified!

 0  0  0  0  0  0
 1  1  1  1  1  1
 1  1  1  1  1  1
 1  1  1  1  1  1
 0  0  0  0  0  0
]]
	  },
	  select = {
	      type = "function",
	      args = ":(dim:Int, index:int)",
	      returns = "(t: Tensor)",
	      description = "Returns the slice at `index` along dimension `dim`."
	  },
	  copy = {
	      type = "function",
	      args = ":()",
	      returns = "(t:Tensor)",
	      description = "A copy of original tensor"
	  },
	  contiguous = {
	      type = "function",
	      args = ":()",
	      returns = "(t: Tensor)",
	      description = "If the tensor is contiguous in storage then same reference otherwise copy tensor to contiguous memory and return that."
	  },
	  type = {
	      type = "function",
	      args = ":([s: TensorType])",
	      returns = "(t: Tensor)",
	      description = "If no arg then type of tensor otherwise a new tensor with desired type"
	  },
	  copy = {
	      type = "function",
	      args = ":(t: Tensor)",
	      returns = "(x: Tensor)",
	      description = "Copies values from t to x"
	  },
	  indexCopy = {
	      type = "function",
	      args = ":(dim: Int, index:Int Or Tensor, T:Tensor)",
	      returns = "void",
	      description = [[
> x
 0.8020  0.7246  0.1204  0.3419  0.4385
 0.0369  0.4158  0.0985  0.3024  0.8186
 0.2746  0.9362  0.2546  0.8586  0.6674
 0.7473  0.9028  0.1046  0.9085  0.6622
 0.1412  0.6784  0.1624  0.8113  0.3949
[torch.DoubleTensor of dimension 5x5]

z=torch.Tensor(5,2)
z:select(2,1):fill(-1)
z:select(2,2):fill(-2)
> z
-1 -2
-1 -2
-1 -2
-1 -2
-1 -2
[torch.DoubleTensor of dimension 5x2]

x:indexCopy(2,torch.LongTensor{5,1},z)
> x
-2.0000  0.7246  0.1204  0.3419 -1.0000
-2.0000  0.4158  0.0985  0.3024 -1.0000
-2.0000  0.9362  0.2546  0.8586 -1.0000
-2.0000  0.9028  0.1046  0.9085 -1.0000
-2.0000  0.6784  0.1624  0.8113 -1.0000
[torch.DoubleTensor of dimension 5x5]
]]
	  },
	  indexAdd = {
	      type = "function",
	      args = ":(dim:Int, index:Int or Tensor, t:Tensor)",
	      returns = "(t:Tensor)",
	      description = [[
Accumulate the elements of `tensor` into the original tensor by adding to the indices in the order
given in `index`. The shape of `tensor` must exactly match the elements indexed or an error will be thrown.

> x
-2.1742  0.5688 -1.0201  0.1383  1.0504
 0.0970  0.2169  0.1324  0.9553 -1.9518
-0.7607  0.8947  0.1658 -0.2181 -2.1237
-1.4099  0.2342  0.4549  0.6316 -0.2608
 0.0349  0.4713  0.0050  0.1677  0.2103
[torch.DoubleTensor of size 5x5]

z=torch.Tensor(5, 2)
z:select(2,1):fill(-1)
z:select(2,2):fill(-2)
> z
-1 -2
-1 -2
-1 -2
-1 -2
-1 -2
[torch.DoubleTensor of dimension 5x2]

> x:indexAdd(2,torch.LongTensor{5,1},z)
> x
-4.1742  0.5688 -1.0201  0.1383  0.0504
-1.9030  0.2169  0.1324  0.9553 -2.9518
-2.7607  0.8947  0.1658 -0.2181 -3.1237
-3.4099  0.2342  0.4549  0.6316 -1.2608
-1.9651  0.4713  0.0050  0.1677 -0.7897
[torch.DoubleTensor of size 5x5]
]]
	  },
	  maskedSelect = {
	      type = "function",
	      args = ":(mask:ByteTensor)",
	      returns = "(t: Tensor)",
	      description = "Select according to mask"
	  },
	  maskedCopy = {
	      type = "function",
	      args = ":(mask: ByteTensor, src:Tensor)",
	      returns = "",
	      description = "Copy from `src` to tensor based on the mask."
	  },
	  nonzero = {
	      type = "function",
	      args = ":()",
	      returns = "(t: LongTensor)",
	      description = "Returns N x d LongTensor where d is the number of dimensions that are nonzero."
	  },
	  le = {
	      type = "function",
	      args = ":(x: Numeric or Tensor)",
	      returns = "(t: ByteTensor)",
	      description = ""
	  },
	  expand = {
	      type = "function",
	      args = "(sizes: torch.LongStorage or Numbers)",
	      returns = "(t: Tensor)",
	      description = [[An expanded view of the original tensor.
Expanding a tensor does not allocate new memory, but only creates a
new view on the existing tensor where singleton dimensions can be
expanded to multiple ones by setting the `stride` to 0.

x = torch.rand(2,1)
> x
 0.3837
 0.5966
y = torch.expand(x,2,2)
> y
 0.3837  0.3837
 0.5966  0.5966
]]
	  },
	  repeatTensor = {
	      type = "function",
	      args = "([result: Tensor], sizes=Ints)",
	      returns = "(t: Tensor)",
	      description = [[
`sizes` can either be a `torch.LongStorage` or numbers. Repeating a tensor allocates
 new memory, unless `result` is provided, in which case its memory is
 resized. `sizes` specify the number of times the tensor is repeated in each dimension.

 ```lua
x = torch.rand(5)
> x
 0.7160
 0.6514
 0.0704
 0.7856
 0.7452
[torch.DoubleTensor of dimension 5]

> torch.repeatTensor(x,3,2)
 0.7160  0.6514  0.0704  0.7856  0.7452  0.7160  0.6514  0.0704  0.7856  0.7452
 0.7160  0.6514  0.0704  0.7856  0.7452  0.7160  0.6514  0.0704  0.7856  0.7452
 0.7160  0.6514  0.0704  0.7856  0.7452  0.7160  0.6514  0.0704  0.7856  0.7452
[torch.DoubleTensor of dimension 3x10]
]]
	  },
	  squeeze = {
	      type = "function",
	      args = "([dim: Int])",
	      returns = "(t: Tensor)",
	      description = [[Removes all singleton dimensions of the tensor.
If `dim` is given, squeezes only that particular dimension of the tensor.]]
	     },
	  permute = {
	      type = "function",
	      args = ":(dim1: Int, dim2: Int, ..., dimN: Int)",
	      returns = "(t: Tensor)",
	      description = [[
Generalizes the function [transpose()](#torch.Tensor.transpose) and can be used
as a convenience method replacing a sequence of transpose() calls.
Returns a tensor where the dimensions were permuted according to the permutation
given by (dim1, dim2, ... , dimn). The permutation must be specified fully, i.e.
there must be as many parameters as the tensor has dimensions.
]]
	  },
	  unfold = {
	      type = "function",
	      args = ":(dim: Int, size: Int, step: Int)",
	      returns = "(t: Tensor)",
	      description = [[
Returns a tensor which contains all slices of size `size` in the dimension `dim`. Step between
two slices is given by `step`. An additional dimension of size `size` is appended in the returned tensor.
]]
	  },
	  map = {
	      type = "function",
	      args = ":(x: Tensor, f::(a: Numeric, b: Numeric)->(c:Numeric): Function)",
	      returns = ":(t: Tensor)",
	      description = [[Apply the given function to all elements of self and `tensor`. The number of elements of both tensors must match, but sizes do not matter. Also see `map2`]]
	  },
	  split = {
	      type = "function",
	      args = ":([result: table], size: Tensor Size, dim: Integer)",
	      returns = "(b: Table)",
	      description = [[Splits Tensor `tensor` along dimension `dim`
into a `result` table of Tensors of size `size` (a number)
or less (in the case of the last Tensor). The sizes of the non-`dim`
dimensions remain unchanged.
x = torch.randn(3,4,5)

> x:split(2,1)
{
  1 : DoubleTensor - size: 2x4x5
  2 : DoubleTensor - size: 1x4x5
}
]]
	  },
	  chunk = {
	      type = "function",
	      args = ":([result: table], size: Tensor Size, dim: Integer)",
	      returns = "(b: Table)",
	      description = [[Splits tensor into n chunks of approx equal size, just like split function]]
	  },
	  free = {
	      type = "function",
	      args = ":()",
	      returns = "()",
	      description = "Free the memory held by tensor."
	  },
	  abs = {type = "function", args="",  returns = "()", description = [[ ]]},
acos = {type = "function", args="",  returns = "()", description = [[ ]]},
add = {type = "function", args="",  returns = "()", description = [[ ]]},
addbmm = {type = "function", args="",  returns = "()", description = [[Batch matrix matrix product of matrices stored in `batch1` and `batch2`, with a reduced add step ]]},
addcdiv = {type = "function", args = "([res,] x [,value], tensor1, tensor2)", returns = "()", description = [[Performs the element-wise division of `tensor1` by `tensor2`, multiply the result by the scalar `value` and add it to `x`. ]]},
addcmul = {type = "function", args="",  returns = "()", description = [[Performs the element-wise multiplication of `tensor1` by `tensor2`, multiply the result by the scalar `value` (1 if not present) and add it to `x`. ]]},
addmm = {type = "function", args="([res,] [v1,] M, [v2,] mat1, mat2)",  returns = "()", description = [[res = (v1 * M) + (v2 * mat1 * mat2) ]]},
addmv = {type = "function", args="([res,] [v1,] vec1, [v2,] mat, vec2)",  returns = "()", description = [[ res = (v1 * vec1) + (v2 * (mat * vec2))]]},
addr = {type = "function", args="([res,] [v1,] mat, [v2,] vec1, vec2)",  returns = "()", description = [[res_ij = (v1 * mat_ij) + (v2 * vec1_i * vec2_j) ]]},
all = {type = "function", args="",  returns = "(b: Boolean)", description = [[Logical and over allocate elements of ByteTensor ]]},
any = {type = "function", args="",  returns = "(b: Boolean)", description = [[ Logical OR over allocate elements of ByteTensor]]},
asin = {type = "function", args="",  returns = "()", description = [[ ]]},
atan = {type = "function", args="",  returns = "()", description = [[ ]]},
atan2 = {type = "function", args="",  returns = "()", description = [[ ]]},
baddbmm = {type = "function", args="([res,] [v1,] M, [v2,] batch1, batch2)",  returns = "()", description = [[res_i = (v1 * M_i) + (v2 * batch1_i * batch2_i) ]]},
bhistc = {type = "function", args="([res,] x [,nbins=100, min_value=-inf, max_value=inf])",  returns = "()", description = [[returns the histogram of the elements in 2d tensor `x` along the last dimension. with nbins ]]},
bitand = {type = "function", args="(b: Boolean)",  returns = "()", description = [[Performs bitwise `and` operation on all elements in the `Tensor` by the given `value`. ]]},
bitor = {type = "function", args="",  returns = "()", description = [[ ]]},
bitxor = {type = "function", args="",  returns = "()", description = [[ ]]},
bmm = {type = "function", args="",  returns = "()", description = [[ ]]},
cat = {type = "function", args="cat( [res,] x_1, x_2, [dimension] )",  returns = "()", description = [[ ]]},
cbitand = {type = "function", args="cbitand([res,] tensor1, tensor2)",  returns = "()", description = [[ ]]},
cbitor = {type = "function", args="([res,] tensor1, tensor2)",  returns = "()", description = [[ ]]},
cbitxor = {type = "function", args="([res,] tensor1, tensor2)",  returns = "()", description = [[ ]]},
cdiv = {type = "function", args="([res,] tensor1, tensor2)",  returns = "()", description = [[ ]]},
ceil = {type = "function", args="([res,] x)",  returns = "()", description = [[ ]]},
cfmod = {type = "function", args="([res,] tensor1, tensor2)",  returns = "()", description = [[ ]]},
cinv = {type = "function", args="",  returns = "()", description = [[ ]]},
clamp = {type = "function", args="([res,] tensor, min_value, max_value)",  returns = "()", description = [[ ]]},
clshift = {type = "function", args="([res,] tensor1, tensor2)",  returns = "()", description = [[ ]]},
cmax = {type = "function", args="([res,] tensor, value|tensor)",  returns = "()", description = [[ ]]},
cmin = {type = "function", args="([res,] tensor, value)",  returns = "()", description = [[ ]]},
cmod = {type = "function", args="([res,] tensor1, tensor2)",  returns = "()", description = [[ ]]},
cmul = {type = "function", args="([res,] tensor1, tensor2)",  returns = "()", description = [[ ]]},
conv2 = {type = "function", args="([res,] x, k, [, 'F' or 'V'])",  returns = "()", description = [[ ]]},
conv3 = {type = "function", args="([res,] x, k, [, 'F' or 'V'])",  returns = "()", description = [[ ]]},
cos = {type = "function", args="([res,] x)",  returns = "()", description = [[ ]]},
cosh = {type = "function", args="([res,] x)",  returns = "()", description = [[ ]]},
cpow = {type = "function", args="([res,] tensor1, tensor2)",  returns = "()", description = [[ ]]},
cremainder = {type = "function", args="([res,] tensor1, tensor2)",  returns = "()", description = [[ ]]},
cross = {type = "function", args="([res,] a, b [,n])",  returns = "()", description = [[ ]]},
crshift = {type = "function", args="([res,] tensor1, tensor2)",  returns = "()", description = [[ ]]},
csub = {type = "function", args="",  returns = "()", description = [[Subtracts the given value from all elements in the `Tensor`, in place. ]]},
cumprod = {type = "function", args="([res,] x [,dim])",  returns = "()", description = [[ ]]},
cumsum = {type = "function", args="([res,] x [,dim])",  returns = "()", description = [[ ]]},
diag = {type = "function", args="([res,] x [,k])",  returns = "()", description = [[ ]]},
dist = {type = "function", args="(x, y, [p=2])",  returns = "(n: Numeric)", description = [[ Return p-norm of x-y]]},
div = {type = "function", args="([res,] tensor, value)",  returns = "()", description = [[ ]]},
dot = {type = "function", args="(t: Tensor)",  returns = "(n: Numeric)", description = [[Elem wise multiply and reduced by sums]]},
eig = {type = "function", args="([rese, resv,] a [, 'N' or 'V'])",  returns = "(e, V)", description = [[`e, V = torch.eig(A)` returns eigenvalues and eigenvectors of a general real square matrix `A`.

`A` and `V` are `m × m` matrices and `e` is a `m` dimensional vector.

This function calculates all right eigenvalues (and vectors) of `A` such that `A = V diag(e) V'`.

Third argument defines computation of eigenvectors or eigenvalues only.
If it is `'N'`, only eigenvalues are computed.
If it is `'V'`, both eigenvalues and eigenvectors are computed.

The eigen values returned follow [LAPACK convention](https://software.intel.com/sites/products/documentation/hpc/mkl/mklman/GUID-16EB5901-5644-4DA6-A332-A052309010C4.htm) and are returned as complex (real/imaginary) pairs of numbers (`2 * m` dimensional `Tensor`).
 ]]},
eq = {type = "function", args="(t: Tensor)",  returns = "(t: New ByteTensor)", description = [[ ]]},
equal = {type = "function", args="(t: Tensor)",  returns = "(b: Boolean)", description = [[ ]]},
exp = {type = "function", args="([res,] x)",  returns = "()", description = [[ ]]},
eye = {type = "function", args="([res,] n [,m])",  returns = "()", description = [[ ]]},
floor = {type = "function", args="([res,] x)",  returns = "()", description = [[ ]]},
fmod = {type = "function", args="([res,] tensor, value)",  returns = "()", description = [[ ]]},
frac = {type = "function", args="([res,] x)",  returns = "()", description = [[ ]]},
ge = {type = "function", args="",  returns = "(t: ByteTensor)", description = [[ ]]},
gels = {type = "function", args="(b: Tensor, a: Tensor)",  returns = "(x: Tensor)", description = [[ Solution of least squares and least norm problems for a full rank `m × n` matrix `A`.

  * If `n ≤ m`, then solve `||AX-B||_F`.
  * If `n > m` , then solve `min ||X||_F` s.t. `AX = B`.

On return, first `n` rows of `x` matrix contains the solution and the rest contains residual information.
Square root of sum squares of elements of each column of `x` starting at row `n + 1` is the residual for corresponding column.

Note: Irrespective of the original strides, the returned matrices `resb` and `resa` will be transposed, i.e. with strides `1, m` instead of `m, 1`.
]]},
ger = {type = "function", args="([res,] vec1, vec2)",  returns = "()", description = [[ ]]},
gesv = {type = "function", args="([resb, resa,] B, A)",  returns = "[x, lu]", description = [[ `X, LU = torch.gesv(B, A)` returns the solution of `AX = B` and `LU` contains `L` and `U` factors for `LU` factorization of `A`.

If `resb` and `resa` are given, then they will be used for temporary storage and returning the result.

  * `resa` will contain `L` and `U` factors for `LU` factorization of `A`.
  * `resb` will contain the solution `X`.
]]},
gt = {type = "function", args="",  returns = "()", description = [[ ]]},
histc = {type = "function", args="([res,] x [,nbins, min_value, max_value])",  returns = "()", description = [[ ]]},
inverse = {type = "function", args="",  returns = "()", description = [[ ]]},
kthvalue = {type = "function", args="([resval, resind,] x, k [,dim])",  returns = "(y: Tensor, i: LongTensor)", description = [[`y = torch.kthvalue(x, k)` returns the `k`-th smallest element of `x` over its last dimension.

`y, i = torch.kthvalue(x, k, 1)` returns the `k`-th smallest element in each column (across rows) of `x`, and a `Tensor` `i` of their corresponding indices in `x`.

`y, i = torch.kthvalue(x, k, 2)` performs the `k`-th value operation for each row.

`y, i = torch.kthvalue(x, k, n)` performs the `k`-th value operation over the dimension `n`.
 ]]},
le = {type = "function", args="",  returns = "()", description = [[ ]]},
lerp = {type = "function", args="([res,] a, b, weight)",  returns = "()", description = [[ ]]},
linspace = {type = "function", args="([res,] x1, x2, [,n])",  returns = "()", description = [[ ]]},
log = {type = "function", args="([res,] x)",  returns = "()", description = [[ ]]},
log1p = {type = "function", args="([res,] x)",  returns = "()", description = [[ ]]},
logspace = {type = "function", args="([res,] x1, x2, [,n])",  returns = "()", description = [[ ]]},
lshift = {type = "function", args="([res,] tensor, value)",  returns = "()", description = [[ ]]},
lt = {type = "function", args="",  returns = "()", description = [[ ]]},
max = {type = "function", args="([resval, resind,] x [,dim])",  returns = "(m: Tensor [, i: Tensor])", description = [[
`y = torch.max(x)` returns the single largest element of `x`.

`y, i = torch.max(x, 1)` returns the largest element in each column (across rows) of `x`, and a `Tensor` `i` of their corresponding indices in `x`.

`y, i = torch.max(x, 2)` performs the max operation for each row.

`y, i = torch.max(x, n)` performs the max operation over the dimension `n`.

th> m = torch.range(1, 12):reshape(3,4)
th> a = torch.Tensor(4)
th> b = torch.LongTensor(4)
th> torch.max(a, b, m, 1)
  9  10  11  12
[torch.DoubleTensor of size 1x4]
 3  3  3  3
[torch.LongTensor of size 1x4]

Now the results are stored in a, b
]]},
mean = {type = "function", args="([res,] x [,dim])",  returns = "()", description = [[ ]]},
median = {type = "function", args="",  returns = "()", description = [[ ]]},
min = {type = "function", args="",  returns = "()", description = [[ ]]},
mm = {type = "function", args="([res,] mat1, mat2)",  returns = "()", description = [[ ]]},
mod = {type = "function", args="([res,] tensor, value)",  returns = "()", description = [[ ]]},
mode = {type = "function", args="",  returns = "()", description = [[ ]]},
mul = {type = "function", args="([res,] tensor1, value)",  returns = "()", description = [[ ]]},
multinomial = {type = "function", args="([res,], p, n, [,replacement])",  returns = "()", description = [[ ]]},
mv = {type = "function", args="([res,] mat, vec)",  returns = "()", description = [[ ]]},
ne = {type = "function", args="",  returns = "()", description = [[ ]]},
neg = {type = "function", args="",  returns = "()", description = [[ ]]},
norm = {type = "function", args="",  returns = "()", description = [[ ]]},
numel = {type = "function", args="",  returns = "()", description = [[ ]]},
ones = {type = "function", args="([res,] m [,n...])",  returns = "()", description = [[ ]]},
orgqr = {type = "function", args="",  returns = "()", description = [[ ]]},
ormqr = {type = "function", args="",  returns = "()", description = [[ ]]},
potrf = {type = "function", args="",  returns = "()", description = [[ ]]},
potri = {type = "function", args="",  returns = "()", description = [[ ]]},
potrs = {type = "function", args="",  returns = "()", description = [[ ]]},
pow = {type = "function", args="([res,] x, n)",  returns = "()", description = [[ ]]},
prod = {type = "function", args="([res,] x [,n])",  returns = "()", description = [[ ]]},
pstrf = {type = "function", args="",  returns = "()", description = [[ ]]},
qr = {type = "function", args="",  returns = "()", description = [[ ]]},
rand = {type = "function", args="([res,] [gen,] m [,n...])",  returns = "()", description = [[ ]]},
randn = {type = "function", args="([res,] [gen,] m [,n...])",  returns = "()", description = [[ ]]},
randperm = {type = "function", args="([res,] [gen,] n)",  returns = "()", description = [[ ]]},
range = {type = "function", args="([res,] x, y [,step])",  returns = "()", description = [[ ]]},
remainder = {type = "function", args="([res,] tensor, value)",  returns = "()", description = [[ ]]},
renorm = {type = "function", args="",  returns = "()", description = [[ ]]},
reshape = {type = "function", args="([res,] x, m [,n...])",  returns = "()", description = [[ ]]},
round = {type = "function", args="([res,] x)",  returns = "()", description = [[ ]]},
rshift = {type = "function", args="([res,] tensor, value)",  returns = "()", description = [[ ]]},
rsqrt = {type = "function", args="([res,] x)",  returns = "()", description = [[ ]]},
sigmoid = {type = "function", args="([res,] x)",  returns = "()", description = [[ ]]},
sign = {type = "function", args="([res,] x)",  returns = "()", description = [[ ]]},
sin = {type = "function", args="([res,] x)",  returns = "()", description = [[ ]]},
sinh = {type = "function", args="([res,] x)",  returns = "()", description = [[ ]]},
sort = {type = "function", args="",  returns = "()", description = [[ ]]},
sqrt = {type = "function", args="([res,] x)",  returns = "()", description = [[ ]]},
std = {type = "function", args="([res,] x, [,dim] [,flag])",  returns = "()", description = [[ ]]},
sum = {type = "function", args="([res,] x)",  returns = "()", description = [[ ]]},
svd = {type = "function", args="",  returns = "()", description = [[ ]]},
symeig = {type = "function", args="",  returns = "()", description = [[ ]]},
tan = {type = "function", args="([res,] x)",  returns = "()", description = [[ ]]},
tanh = {type = "function", args="([res,] x)",  returns = "()", description = [[ ]]},
topk = {type = "function", args="([resval, resind,] x, k, [,dim] [,dir] [,sort])",  returns = "(y, i)", description = [[`y, i = torch.topk(x, k)` returns all `k` smallest elements in `x` over its last dimension including their indices, in unsorted order.

`y, i = torch.topk(x, k, dim, dir, true)` specifies that the results in `y` should be sorted with respect to `dir`; by default, the results are potentially unsorted since the computation may be faster, but if sorting is desired, the sort flag may be passed, in which case the results are returned from smallest to `k`-th smallest (`dir == false`) or highest to `k`-th highest (`dir == true`).
]]},
trace = {type = "function", args="",  returns = "()", description = [[ ]]},
tril = {type = "function", args="([res,] x [,k])",  returns = "()", description = [[ ]]},
triu = {type = "function", args="([res,] x, [,k])",  returns = "()", description = [[ ]]},
trtrs = {type = "function", args="[x]",  returns = "([resb, resa,] b, a [, 'U' or 'L'] [, 'N' or 'T'] [, 'N' or 'U'])", description = [[`X = torch.trtrs(B, A)` returns the solution of `AX = B` where `A` is upper-triangular.

`A` has to be a square, triangular, non-singular matrix (2D `Tensor`).
`A` and `resa` are `m × m`, `X` and `B` are `m × k`.
(To be very precise: `A` does not have to be triangular and non-singular, rather only its upper or lower triangle will be taken into account and that part has to be non-singular.)

The function has several options:

* `uplo` (`'U'` or `'L'`) specifies whether `A` is upper or lower triangular; the default value is `'U'`.
* `trans` (`'N'` or `'T`') specifies the system of equations: `'N'` for `A * X = B` (no transpose), or `'T'` for `A^T * X = B` (transpose); the default value is `'N'`.
* `diag` (`'N'` or `'U'`) `'U'` specifies that `A` is unit triangular, i.e., it has ones on its diagonal; `'N'` specifies that `A` is not (necessarily) unit triangular; the default value is `'N'`.

If `resb` and `resa` are given, then they will be used for temporary storage and returning the result.
`resb` will contain the solution `X`.

Note: Irrespective of the original strides, the returned matrices `resb` and `resa` will be transposed, i.e. with strides `1, m` instead of `m, 1`.
 ]]},
trunc = {type = "function", args="([res,] x)",  returns = "()", description = [[ ]]},
var = {type = "function", args="([res,] x [,dim] [,flag])",  returns = "()", description = [[ ]]},
xcorr2 = {type = "function", args="([res,] x, k, [, 'F' or 'V'])",  returns = "()", description = [[ ]]},
xcorr3 = {type = "function", args="([res,] x, k, [, 'F' or 'V'])",  returns = "()", description = [[ ]]},
zeros = {type = "function", args="",  returns = "()", description = [[ ]]},
	},
      },
    },
  },
}

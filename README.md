# Regent-FFT

[![Regent-FFT
Tests](https://github.com/StanfordLegion/regent-fft/actions/workflows/main.yml/badge.svg)](https://github.com/StanfordLegion/regent-fft/actions)

This is a fast fourier transform (FFT) library built in [Regent](https://regent-lang.org/).

## Description

Regent-FFT takes the input matrix for the discrete fourier
transform (DFT) in a region, and saves the output in an output region.

The library currently supports transforms up to 3 dimensions, and can be
configured to run on either CPUs or GPUs.

The CPU mode is powered by [FFTW](https://www.fftw.org/), and the GPU mode by
[cuFFT](https://developer.nvidia.com/cufft).

Both Complex-to-Complex and Real-To-Complex transformations are supported.

Both single-precision and double-precision modes are supported (i.e., both `float`
/ `complex32` and `double` / `complex64` types).

Batched transforms are also supported.

## Getting Started

### Install

First, make sure you have Regent installed. (If you are using Sapling, please
refer to the [Sapling guide](https://github.com/StanfordLegion/sapling-guide).)
Be sure to install with `MAX_DIM=4` (for Make) or `-DLegion_MAX_DIM=4` (for CMake).

Note that on Sapling you will want to load the following modules:

```shell
module load slurm mpi cmake cuda llvm
```

Then, clone the repo:

```shell
git clone https://github.com/StanfordLegion/regent-fft.git
```

Next, run the install script and add environment variables:

```shell
./install.py
source env.sh
```

Then, run your `.rg` script, which can be set up using the instructions in the
'Executing Program' section:

```shell
../legion/language/regent.py test/fft_test.rg
```

## Usage

### Execute Program

Regent-FFT supports four distinct axes of usage:

1. GPU vs. CPU
2. Complex-to-Complex vs. Real-to-Complex
3. Float vs. Double
4. Single vs. Batched

API usage generally follows the following steps.

First, an FFT interface has to be generated depending on the type of transform
you want to do. Then, we:

- Create a plan,
- Execute the said plan, and then
- Destroy the plan once we are done.

There are several sample code snippets in the `fft_test.rg` file for reference
as well.

#### 1. Import `fft.rg` and Create the FFT Interface

To generate a specific instance of the library, use `fft.generate_fft_interface(...)`.

- The first argument is the index type of the input and output region: `int1d`, `int2d`, or `int3d`. This also tells the FFT interface what dimensionality to expect.
- The second argument is the data type of the input: `complex64`, `complex32`,
  `float`, or `double`.
- The third argument is the data type of the output: `complex64` or `complex32`.
- The fourth argument is a flag for batched transforms: `false` in regular mode, and `true` in batched mode.

```lua
local fft = require("fft")
local fft1d = fft.generate_fft_interface(int1d, complex64, complex64, false)
```

#### 2. Make Plan

Like many FFT libraries, Regent FFT requires the use of plans. Plans are specific to the sizes of the input and output regions being used, as well as the machine node the plan is initialized on. Currently the enforcement of these assumptions is the responsibility of the user.

The first step is to call `make_plan`, which takes three arguments:

1. `r`: input region
2. `s`: output region
3. `p`: plan region

```lua
fft1d.make_plan(r, s, p)
```

The input region should be initialized with index space of the form
`ispace(<type>, N)`, where N is the size of the array, and `<type>` is either
`int1d`, `int2d`, or `int3d` depending on the dimension of the transform. The fieldspace of
the region is the type supported by the transform - e.g, in a Real-to-Complex
transform with double precision, the input array will have fieldspace `double`
and output array will have fieldspace `complex64`. Larger fieldspaces that
contain the appropriate types can also be passed in via field polymorphism -
[here](https://gitlab.com/StanfordLegion/legion/-/blob/master/language/tests/regent/run_pass/call_task_polymorphic1.rg) is
an example.

For example, in a 1D double-to-complex64 transform of size 3, the input and
output regions may be initialized as follows:

```lua
var r = region(ispace(int1d, 3), double)
var s = region(ispace(int1d, 3), complex64)
```

The way that a plan is initialized depends on the usage mode. In
general, plans are stored in a region which is managed by the
user. The plan region may be a subregion and need not start at zero,
but it must contain at least a number of elements depending on the
mode: 1 in non-distributed mode, and `N` in distributed mode - where `N` is
the number of nodes.

In non-distributed mode, the plan region can be initialized as follows:

```lua
var p = region(ispace(int1d, 1), fft1d.plan)
```

Every task in the Regent-FFT interface provides two versions: one which is
`__demand(__inline)` and one which is just a regular task. Here, these are
called `make_plan` (for inline) and `make_plan_task` (for a task). Users
should pick the appropriate version depending on whether they want to launch a
new task or not.

> [!IMPORTANT]
>
> `make_plan` overwrites the input and output regions `r` and `s`. This is
> mandated by FFTW, which Regent FFT uses on CPUs. In order to avoid overwriting
> data, the user must either initialize the plan prior to loading the regions
> with data, or else must create a temporary region (of the same size and layout
> as the real one) for use in initialization.

#### 3. Execute Plan

Next, we execute the plan. This takes the same 3 regions as mentioned above.

```lua
fft1d.execute_plan(r, s, p)
```

> [!IMPORTANT]
>
> Because `execute_plan` is a `__demand(__inline)` task, it will never execute
> on the GPU (unless the parent task is running on the GPU). Therefore, in most
> cases, it is necessary to use `execute_plan_task` if one wants to use the GPU.

> [!IMPORTANT]
>
> While `execute_plan_task` may be executed on the GPU, the contents of the `p`
> region must still be available on the CPU, because the plans must be used by
> the host-side code to launch the FFT kernels. Therefore, when
> `execute_plan_task` is mapped onto the GPU, it is very important to map the
> `p` region into zero-copy memory.

#### 4. Destroy Plan

When a plan is no longer needed, it can be destroyed:

```lua
fft1d.destroy_plan(p)
```

#### 5. Batched Transforms

Batched transforms allow users to perform multiple transforms of the same size simultaneously, in a 'batch'.

To illustrate how to perform a batched transform, let us use the example where
you want to perform 7 batches of a 256 x 256 transform.

Since the transform is a 2D one, the user creates a interface with `itype` of the same
dimension: in this case, an `int2d`. Be sure to pass in `true` as the fourth argument to indicate we are performing a batched transform.

```lua
local fft2d_batch_complex64_complex64 = fft.generate_fft_interface(int2d, complex64, complex64, true)
```

The input and output regions should be of dimension 'N+1', in this case 256 x 256 x 7. The size of the last
dimension is the number of batches. The plan region remains the same as before:

```lua
var r = region(ispace(int3d, {256, 256, 7}), complex64)
var s = region(ispace(int3d, {256, 256, 7}), complex64)
var p = region(ispace(int1d, 1), fft2d_batch_double_complex64.plan)
```

The key difference is that we call `make_plan_batch` instead of `make_plan`.

```lua
fft2d_batch_complex64_complex64.make_plan_batch(r, s, p)
```

Then, we execute and destroy as in the regular case.

```lua
fft2d_batch_complex64_complex64.execute_plan(r, s, p)
fft2d_batch_complex64_complex64.destroy_plan(p)
```

Please also refer to the `test_2d_complex64_to_complex64_batch_transform` and
`test_2d_double_to_complex64_batch_transform` examples in `fft_test.rg` for further
examples.

#### 6. Distributed Mode

The API also supports a distributed mode, where every machine in a distributed
job executes an independent FFT.

To initialize in distributed mode, we might do the following (where we have `n` 1-D
`complex64` to `complex64` transforms of size `m`, and partitions `r_part` and `s_part` that are distributed around the
machine):

```lua
var n = fft1d.get_num_nodes()
var p = region(ispace(int1d, n), fft1d.plan)
var p_part = partition(equal, p, ispace(int1d, n))
var r = region(ispace(int1d, m*n), complex64)
var r_part = partition(equal, r, ispace(int1d, n))
var s = region(ispace(int1d, m*n), complex64)
var s_part = partition(equal, s, ispace(int1d, n))
fft1d_complex64_complex64.make_plan_distrib(r, r_part, s, s_part, p, p_part)
```

Note the use of `get_num_nodes` to determine the size of the `p` region and
partition. The task `make_plan_distrib` is a `__demand(__inline)` task that
internally performs an index launch over the machine to initialize `p`
(i.e., it will launch one `make_plan_task` per subregion of the inputs).

> [!IMPORTANT]
>
> Like `make_plan`, `make_plan_distrib` overwrites the input and output regions
> `r` and `s`.

> [!IMPORTANT]
>
> In order for the distributed API to work correctly, it is essential that each
> task in the index launch inside of `make_plan_distrib` is mapped onto a
> separate node. This ensures that when the region `p` is used later, there is a
> plan for every node in the machine.

Then, we execute the plan:

```lua
fft1d_complex64_complex64.execute_plan_distrib(r, r_part, s, s_part, p, p_part)
```

Lastly, to destroy the plan:

```lua
fft1d_complex64_complex64.destroy_plan_distrib(p, p_part)
```

> [!NOTE]
>
> Be sure to be consistent in using either `make_plan`, `execute_plan` and `destroy_plan`; or `make_plan_distrib`, `execute_plan_distrib` and `destroy_plan_distrib`
> depending on which version of the API you are using, as the processor that runs `make_plan` should be the same processor used for `execute_plan` and `destroy_plan`.

> [!NOTE]
>
> As with `make_plan_distrib`, `execute_plan_distrib` and `destroy_plan_distrib` will
> internally perform an index launch to destroy the plans on each node.

> [!IMPORTANT]
>
> Like `make_plan_distrib`, the index launches issued by `execute_plan_distrib` and `destroy_plan_distrib`
> must be mapped so that each point task runs on the node where the plan was
> originally created.

## Caveats

- FFTW's planner is *not thread-safe*, but this isn't currently protected by a lock in the Regent wrapper.

- Optimization: Currently we take the approach of "always measure" with FFTW. This isn't even an option with cuFFT. Should this be exposed?

- Missing features:
  - Strides
  - Backwards
  - Split

## Authors

- Elliott Slaughter (<slaughter@cs.stanford.edu>)
- Arjun Kunna (<arjunkunna@gmail.com>)

## Additional Resources

- For information on the FFT transform, please refer to these set of
  [notes](https://web.stanford.edu/class/cs168/l/l15.pdf) or this
  [course](https://see.stanford.edu/Course/EE261).

## Acknowledgments

- [FFTW](https://www.fftw.org/)
- [cuFFT](https://developer.nvidia.com/cufft)
- [Regent](https://regent-lang.org/)
- [Legion](https://legion.stanford.edu/starting/)

# Regent FFT library

## Installation

```
./install.py
source env.sh
../regent.py test/fft_test.rg
```

## Usage

Regent FFT currently supports two distinct classes of use cases:

 1. Every machine in a distributed job executes an independent FFT of
    a different size. (Use the "non-distributed" API below.)
 2. Every machine in a distributed job executes an independent FFT of
    the same size. (Use the "distributed" API below.)

In the future we may add support for collective FFTs where multiple
machines cooperate to jointly compute a single FFT which is too large
to fit in any single node's memory.

## API Overview

In order to use the FFT library, you must first initialize it for the
specific number of dimensions and data type you are interested
in. Currently only 1 to 3 dimensions and `complex64` are
supported. For example, to initialize with 1 dimension:

```
local fft = require("fft")

local fft1d = fft.generate_fft_interface(int1d, complex64)
```

### Initialization

Like many FFT libraries, Regent FFT requires the use of *plans*. Plans
are specific to the sizes of the input and output regions being used,
as well as the machine node the plan is initialized on. Currently *the
enforcement of these assumptions is the responsibility of the user*.

The way that a plan is initialized depends on the usage mode. In
general, plans are stored in a region which is managed by the
user. The plan region may be a subregion and need not start at zero,
but it must contain at least a number of elements depending on the
mode: 1 in non-distributed mode, `N` in distributed mode where `N` is
the number of nodes.

To initialize in non-distributed mode:

```
var p = region(ispace(int1d, 1), fft1d.plan)
fft1d.make_plan(r, s, p)
```

**Important:** `make_plan` overwites the input and output regions `r`
and `s`. This is mandated by FFTW, which Regent FFT uses on CPUs. In
order to avoid overwriting data, the user must either initialize the
plan prior to loading the regions with data, or else must create a
temporary region (of the same size and shape as the real one) for use
in initialization.

Note that, like all non-distributed mode tasks, `make_plan` is a
`__demand(__inline)` task. This means that if the user wants it to
execute it in a separate task, they must wrap the task themselves.

To initialize in distributed mode:

```
var n = fft1d.get_num_nodes()
var p = region(ispace(int1d, n), fft1d.plan)
var p_part = partition(equal, p, ispace(int1d, n))
fft1d.make_plan_distrib(r, r_part, s, s_part, p, p_part)
```

Note the use of `get_num_nodes` to determine the size of the `p`
region and partition. The task `make_plan_distrib` is a
`__demand(__inline)` task that internally performs an index launch
over the machine to initialize `p`.

**Important:** like `make_plan`, `make_plan_distrib` overwrites the
input and output regions `r` and `s`.

**Important:** in order for the distributed API to work correctly, it
is essential that each task in the index launch inside of
`make_plan_distrib` is mapped onto a separate node. This ensures that
when the region `p` is used later, there is a plan for every node in
the machine.

### Execution

To perform an FFT, execute the plan. Note that, unlike with
initialization, this looks similar whether or not distributed mode is
being used.

For example, if there is a single input `r` and output `s`:

```
fft1d.execute_plan(r, s, p)
```

Or if there are partitions `r_part` and `s_part` that are distributed
around the machine, one might do:

```
__demand(__index_launch)
for i in r_part.colors do
  fft1d.execute_plan_task(r_part[i], s_part[i], p)
end
```

Note that `execute_plan` is a `__demand(__inline)` task (as described
above). The task `execute_plan_task` is simply a wrapper around
`execute_plan` for convenience, to avoid needing to define this
explicitly.

**Important:** because `execute_plan` is a `__demand(__inline)` task,
it will never execute on the GPU (unless the parent task is running on
the GPU). Therefore, in most cases it is necessary to use
`execute_plan_task` if one wants to use the GPU.

**Important:** while `execute_plan_task` may be executed on the GPU,
the contents of the `p` region must still be available on the CPU,
because the plans must be used by the host-side code to launch the FFT
kernels. Therefore, when `execute_plan_task` is mapped onto the GPU it
is very important to map the `p` region into zero-copy memory.

### Finalization

When a plan is no longer needed it can be distroyed.

In the non-distributed API:

```
fft1d.destroy_plan(p)
```

In the distributed API:

```
fft1d.destroy_plan_distrib(p, p_part)
```

Note: these are both `__demand(__inline)` tasks, and
`destroy_plan_distrib` will internally perform an index launch to
destroy the plans on each node.

**Important:** like `make_plan_distrib`, the index launch issued by
`destroy_plan_distrib` must be mapped so that each point task runs on
the node where the plan was originally created.

## Caveats

  * FFTW's planner is *not thread-safe*, but this isn't currently
    protected by a lock in the Regent wrapper.

  * FFT operations are backed by FFTW, so at the moment the run
    CPU-only and only on a single node. (OpenMP is hypothetically
    possible, but I haven't tested it yet.)

  * Optimization: Currently we take the approach of "always measure"
    with FFTW. This isn't even an option with cuFFT. Should this be
    exposed?

  * Missing features:
      * strides
      * batches
      * complex32
      * backwards
      * split
      * real

  * Multi-GPU is probably not supported right now because cuFFT plans
    are (probably) tied to a context for a specific GPU.

## Implementation Notes

  * In a distributed execution, plans need to be collected into a big
    region in order to allow load-balancing (and flexible mapping in
    general). This means all nodes need to be executing FFTs of the
    same size. If this isn't the case, then plans can be created
    locally (with the non-distributed API) but then the FFT cannot be
    moved for execution somewhere else.

  * Plans for the GPU need to go in zero-copy memory so that they can
    be accessed by the CPU thread which is assigned to execute the
    task

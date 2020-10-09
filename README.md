# Regent FFT library

## Installation

```
./install.py
source env.sh
../regent.py test/fft_test.rg
```

## Usage

In order to use the FFT library, you must first initialize it for the
specific number of dimensions and data type you are interested
in. Currently only 1 to 3 dimensions and `complex64` are supported.

```
local fft = require("fft")

local fft1d = fft.generate_fft_interface(int1d, complex64)
```

The FFT library requires the use of *plans*. Plans are specific to the
sizes of the input and output regions being used, as well as the
machine node the plan is initialized on. Currently *these assumptions
are the responsibility of the user*.

To initialize a plan for some regions `r` (input) and `s` (output):

```
var p = region(ispace(int1d, 1), fft1d.plan)
fft1d.make_plan(r, s, p)
```

Note: it's ok to initialize a region `p` larger than size 1, as long
as you partition it before use. The FFT library will only ever use the
first element of the `p` region which it is given.

Then to use the plan:

```
fft1d.execute_plan(r, s, p)
fft1d.destroy_plan(p)
```

Note: it's ok to use different regions `r` and `s` here as long as
they are the same size and layout as those used to originally
construct the plan.

Note: all of the tasks in the FFT library are inline tasks. That means
they will execute in the same context, synchronously blocking the
parent task. If you want asynchronous execute, use a task to call
these.

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

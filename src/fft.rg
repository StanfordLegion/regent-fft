-- Copyright 2020 Stanford University
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

--[[--
Regent FFT library.
]]

import "regent"

local data = require("common/data")
local cudahelper = require("regent/cudahelper")

local c = regentlib.c
local fftw_c = terralib.includec("fftw3.h")
terralib.linklibrary("libfftw3.so")

local use_cuda = cudahelper.check_cuda_available()
local cufft_c
if use_cuda then
  cufft_c = terralib.includec("cufft.h")
  terralib.linklibrary("libcufft.so")
end

-- Hack: get defines from fftw3.h
fftw_c.FFTW_FORWARD = -1
fftw_c.FFTW_BACKWARD = 1

fftw_c.FFTW_MEASURE = 0
fftw_c.FFTW_DESTROY_INPUT = (2 ^ 0)
fftw_c.FFTW_UNALIGNED = (2 ^ 1)
fftw_c.FFTW_CONSERVE_MEMORY = (2 ^ 2)
fftw_c.FFTW_EXHAUSTIVE = (2 ^ 3) -- NO_EXHAUSTIVE is default
fftw_c.FFTW_PRESERVE_INPUT = (2 ^ 4) -- cancels FFTW_DESTROY_INPUT
fftw_c.FFTW_PATIENT = (2 ^ 5) -- IMPATIENT is default
fftw_c.FFTW_ESTIMATE = (2 ^ 6)
fftw_c.FFTW_WISDOM_ONLY = (2 ^ 21)

local fft = {}

function fft.generate_fft_interface(itype, dtype)
  assert(regentlib.is_index_type(itype), "requires an index type as the first argument")
  local dim = itype.dim
  assert(dim >= 1 and dim <= 3, "currently only 1 <= dim <= 3 is supported")
  assert(dtype == complex64, "currently only complex64 is supported")

  local iface = {}

  local iface_plan
  if use_cuda then
    fspace iface_plan {
      p : fftw_c.fftw_plan,
      cufft_p : cufft_c.cufftHandle,
      address_space : c.legion_address_space_t,
    }
  else
    fspace iface_plan {
      p : fftw_c.fftw_plan,
      address_space : c.legion_address_space_t,
    }
  end
  iface.plan = iface_plan
  iface.plan.__no_field_slicing = true -- don't field slice this struct

  local function make_get_base(d, t)
    local rect_t = c["legion_rect_" .. d .. "d_t"]
    local get_accessor = c["legion_physical_region_get_field_accessor_array_" .. d .. "d"]
    local raw_rect_ptr = c["legion_accessor_array_" .. d .. "d_raw_rect_ptr"]
    local destroy_accessor = c["legion_accessor_array_" .. d .. "d_destroy"]

    local terra get_base(rect : rect_t,
                 physical : c.legion_physical_region_t,
                 field : c.legion_field_id_t)
      var subrect : rect_t
      var offsets : c.legion_byte_offset_t[d]
      var accessor = get_accessor(physical, field)
      var base_pointer = [&t](raw_rect_ptr(accessor, rect, &subrect, &(offsets[0])))
      regentlib.assert(base_pointer ~= nil, "base pointer is nil")
      escape
        for i = 0, d-1 do
          emit quote
            regentlib.assert(subrect.lo.x[i] == rect.lo.x[i], "subrect not equal to rect")
          end
        end
      end
      regentlib.assert(offsets[0].offset == terralib.sizeof(t), "stride does not match expected value")

      destroy_accessor(accessor)

      return base_pointer
    end

    return rect_t, get_base
  end
  local rect_plan_t, get_base_plan = make_get_base(1, iface.plan)
  local rect_t, get_base = make_get_base(dim, dtype)

  -- Hack: need to retrieve context without __context() to circumvent leaf checker here.
  local terra get_executing_processor(runtime : c.legion_runtime_t)
    var ctx = c.legion_runtime_get_context()
    var result = c.legion_runtime_get_executing_processor(runtime, ctx)
    c.legion_context_destroy(ctx)
    return result
  end

  -- FIXME: Keep this in sync with default_mapper.h
  local DEFAULT_TUNABLE_NODE_COUNT = 0
  local DEFAULT_TUNABLE_LOCAL_CPUS = 1
  local DEFAULT_TUNABLE_LOCAL_GPUS = 2
  local DEFAULT_TUNABLE_LOCAL_IOS = 3
  local DEFAULT_TUNABLE_LOCAL_OMPS = 4
  local DEFAULT_TUNABLE_LOCAL_PYS = 5
  local DEFAULT_TUNABLE_GLOBAL_CPUS = 6
  local DEFAULT_TUNABLE_GLOBAL_GPUS = 7
  local DEFAULT_TUNABLE_GLOBAL_IOS = 8
  local DEFAULT_TUNABLE_GLOBAL_OMPS = 9
  local DEFAULT_TUNABLE_GLOBAL_PYS = 10

  __demand(__inline)
  task iface.get_tunable(tunable_id : int)
    var f = c.legion_runtime_select_tunable_value(__runtime(), __context(), tunable_id, 0, 0)
    var n = __future(int64, f)

    -- FIXME (Elliott): I thought Regent was supposed to copy on
    -- assignment, but that seems not to happen here, so this would
    -- result in a double destroy if we free here.

    -- c.legion_future_destroy(f)
    return n
  end

  __demand(__inline)
  task iface.get_num_nodes()
    return iface.get_tunable(DEFAULT_TUNABLE_NODE_COUNT)
  end

  __demand(__inline)
  task iface.get_num_local_gpus()
    return iface.get_tunable(DEFAULT_TUNABLE_LOCAL_GPUS)
  end

  __demand(__inline)
  task iface.get_plan(plan : region(ispace(int1d), iface.plan), check : bool) : &iface.plan
  where reads(plan) do
    -- Hack: 3Bneed to use raw access to circument CUDA checker here.
    var pr = __physical(plan)[0]
    regentlib.assert(c.legion_physical_region_get_memory_count(pr) == 1, "plan instance has more than one memory?")
    var mem_kind = c.legion_memory_kind(c.legion_physical_region_get_memory(pr, 0))
    regentlib.assert(
      mem_kind == c.SYSTEM_MEM or mem_kind == c.REGDMA_MEM or mem_kind == c.Z_COPY_MEM,
      "plan instance must be stored in sysmem, regmem, or zero copy mem")
    var plan_base = get_base_plan(rect_plan_t(plan.ispace.bounds), __physical(plan)[0], __fields(plan)[0])
    var i = c.legion_processor_address_space(get_executing_processor(__runtime()))
    var p : &iface.plan
    var bounds = plan.ispace.bounds
    if bounds.hi - bounds.lo + 1 > int1d(1) then
      p = plan_base + i
    else
      p = plan_base
    end
    regentlib.assert(not check or p.address_space == i, "plans can only be used on the node where they are originally created")
    return p
  end

  local plan_dft = fftw_c["fftw_plan_dft_" .. dim .. "d"]

  local make_plan_gpu
  if use_cuda then
    __demand(__cuda, __leaf)
    task make_plan_gpu(input : region(ispace(itype), dtype),
                       output : region(ispace(itype), dtype),
                       address_space : c.legion_address_space_t) : cufft_c.cufftHandle
    where reads writes(input, output) do
      var proc = get_executing_processor(__runtime())
      if c.legion_processor_kind(proc) == c.TOC_PROC then
        var i = c.legion_processor_address_space(proc)
        regentlib.assert(address_space == i, "make_plan_gpu must be executed on a processor in the same address space")

        var input_base = get_base(rect_t(input.ispace.bounds), __physical(input)[0], __fields(input)[0])
        var output_base = get_base(rect_t(output.ispace.bounds), __physical(output)[0], __fields(output)[0])
        var lo = input.ispace.bounds.lo:to_point()
        var hi = input.ispace.bounds.hi:to_point()
        var n : int[dim]
        ;[data.range(dim):map(function(i) return rquote n[i] = hi.x[i] - lo.x[i] + 1 end end)]
        var cufft_p : cufft_c.cufftHandle
        var ok = cufft_c.cufftPlanMany(
          &cufft_p,
          dim,
          &n[0],
          [&int](0), 0, 0, -- inembed, istride, idist
          [&int](0), 0, 0, -- onembed, ostride, odist
          cufft_c.CUFFT_C2C,
          1 -- batch
        )
        regentlib.assert(ok == cufft_c.CUFFT_SUCCESS, "cufftPlanMany failed")
        return cufft_p
      else
        regentlib.assert(false, "make_plan_gpu must be executed on a GPU processor")
      end
    end
  end

  -- Important: overwrites input/output!
  __demand(__inline)
  task iface.make_plan(input : region(ispace(itype), dtype),
                       output : region(ispace(itype), dtype),
                       plan : region(ispace(int1d), iface.plan))
  where reads writes(input, output, plan) do
    var p = iface.get_plan(plan, false)

    var address_space = c.legion_processor_address_space(
      get_executing_processor(__runtime()))

    regentlib.assert(input.ispace.bounds == output.ispace.bounds, "input and output regions must be identical in size")
    var input_base = get_base(rect_t(input.ispace.bounds), __physical(input)[0], __fields(input)[0])
    var output_base = get_base(rect_t(output.ispace.bounds), __physical(output)[0], __fields(output)[0])
    var lo = input.ispace.bounds.lo:to_point()
    var hi = input.ispace.bounds.hi:to_point()
    var flags = fftw_c.FFTW_MEASURE
    p.p = plan_dft(
      [data.range(dim):map(function(i) return rexpr hi.x[i] - lo.x[i] + 1 end end)],
      [&fftw_c.fftw_complex](input_base),
      [&fftw_c.fftw_complex](output_base),
      fftw_c.FFTW_FORWARD,
      flags)
    p.address_space = address_space
    ;[(function()
         if use_cuda then
           return rquote
             if iface.get_num_local_gpus() > 0 then
               p.cufft_p = make_plan_gpu(input, output, address_space)
             end
           end
         else
           return rquote end
         end
       end)()]
  end

  task iface.make_plan_task(input : region(ispace(itype), dtype),
                            output : region(ispace(itype), dtype),
                            plan : region(ispace(int1d), iface.plan))
  where reads writes(input, output, plan) do
    iface.make_plan(input, output, plan)
  end

  __demand(__inline)
  task iface.make_plan_distrib(input : region(ispace(itype), dtype),
                               input_part : partition(disjoint, input, ispace(int1d)),
                               output : region(ispace(itype), dtype),
                               output_part : partition(disjoint, output, ispace(int1d)),
                               plan : region(ispace(int1d), iface.plan),
                               plan_part : partition(disjoint, plan, ispace(int1d)))
  where reads writes(input, output, plan) do
    var n = iface.get_num_nodes()
    regentlib.assert(input_part.colors.bounds.hi - input_part.colors.bounds.lo + 1 == int1d(n), "input_part colors size must be equal to the number of nodes")
    regentlib.assert(input_part.colors.bounds == output_part.colors.bounds, "input_part and output_part colors must be equal")
    regentlib.assert(input_part.colors.bounds == plan_part.colors.bounds, "input_part and plan_part colors must be equal")

    var p : iface.plan
    p.p = [fftw_c.fftw_plan](0)
    p.address_space
    ;[(function() if use_cuda then return rquote p.cufft_p = 0 end else return rquote end end end)()]
    fill(plan, p)

    __demand(__index_launch)
    for i in plan_part.colors do
      iface.make_plan_task(input_part[i], output_part[i], plan_part[i])
    end
  end

  __demand(__inline)
  task iface.execute_plan(input : region(ispace(itype), dtype),
                          output : region(ispace(itype), dtype),
                          plan : region(ispace(int1d), iface.plan))
  where reads(input, plan), writes(output) do
    var p = iface.get_plan(plan, true)
    var input_base = get_base(rect_t(input.ispace.bounds), __physical(input)[0], __fields(input)[0])
    var output_base = get_base(rect_t(output.ispace.bounds), __physical(output)[0], __fields(output)[0])

    var proc = get_executing_processor(__runtime())
    if c.legion_processor_kind(proc) == c.TOC_PROC then
      c.printf("execute plan via cuFFT\n")
    else
      c.printf("execute plan via FFTW\n")
      fftw_c.fftw_execute_dft(p.p, [&fftw_c.fftw_complex](input_base), [&fftw_c.fftw_complex](output_base))
    end
  end

  __demand(__cuda, __leaf)
  task iface.execute_plan_task(input : region(ispace(itype), dtype),
                               output : region(ispace(itype), dtype),
                               plan : region(ispace(int1d), iface.plan))
  where reads(input, plan), writes(output) do
    iface.execute_plan(input, output, plan)
  end

  __demand(__inline)
  task iface.destroy_plan(plan : region(ispace(int1d), iface.plan))
  where reads writes(plan) do
    var p = iface.get_plan(plan, true)
    fftw_c.fftw_destroy_plan(p.p)
  end

  task iface.destroy_plan_task(plan : region(ispace(int1d), iface.plan))
  where reads writes(plan) do
    iface.destroy_plan(plan)
  end

  __demand(__inline)
  task iface.destroy_plan_distrib(plan : region(ispace(int1d), iface.plan),
                                  plan_part : partition(disjoint, plan, ispace(int1d)))
  where reads writes(plan) do
    for i in plan_part.colors do
      iface.destroy_plan_task(plan_part[i])
    end
  end

  return iface
end

return fft

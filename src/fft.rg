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

local c = regentlib.c
local fftw_c = terralib.includec("fftw3.h")
terralib.linklibrary("libfftw3.so")

-- Hack: get defines from fftw3.h
fftw_c.FFTW_FORWARD = -1
fftw_c.FFTW_BACKWARD = 1
fftw_c.FFTW_ESTIMATE = 2 ^ 6

local fft = {}

function fft.generate_fft_interface(itype, dtype)
  assert(regentlib.is_index_type(itype), "requires an index type as the first argument")
  local dim = itype.dim
  assert(dim >= 1 and dim <= 3, "currently only 1 <= dim <= 3 is supported")
  assert(dtype == complex64, "currently only complex64 is supported")

  local rect_t = c["legion_rect_" .. dim .. "d_t"]
  local get_accessor = c["legion_physical_region_get_field_accessor_array_" .. dim .. "d"]
  local raw_rect_ptr = c["legion_accessor_array_" .. dim .. "d_raw_rect_ptr"]
  local destroy_accessor = c["legion_accessor_array_" .. dim .. "d_destroy"]

  local iface = {}

  local fspace iface_plan {
    p : fftw_c.fftw_plan,
    address_space : c.legion_address_space_t,
  }
  iface.plan = iface_plan

  local terra get_base(rect : rect_t,
                 physical : c.legion_physical_region_t,
                 field : c.legion_field_id_t)
    var subrect : rect_t
    var offsets : c.legion_byte_offset_t[dim]
    var accessor = get_accessor(physical, field)
    var base_pointer = [&dtype](raw_rect_ptr(accessor, rect, &subrect, &(offsets[0])))
    regentlib.assert(base_pointer ~= nil, "base pointer is nil")
    escape
      for i = 0, dim-1 do
        emit quote
          regentlib.assert(subrect.lo.x[i] == rect.lo.x[i], "subrect not equal to rect")
        end
      end
    end
    regentlib.assert(offsets[0].offset == terralib.sizeof(dtype), "stride does not match expected value")

    destroy_accessor(accessor)

    return base_pointer
  end

  local plan_dft = fftw_c["fftw_plan_dft_" .. dim .. "d"]

  -- Important: overwrites input/output!
  __demand(__inline)
  task iface.make_plan(input : region(ispace(itype), dtype),
                       output : region(ispace(itype), dtype),
                       plan : region(ispace(int1d), iface.plan))
  where reads writes(input, output, plan) do
    var p : int1d(iface.plan, plan)
    for x in plan do
      p = x
      break
    end

    var address_space = c.legion_processor_address_space(
      c.legion_runtime_get_executing_processor(__runtime(), __context()))

    regentlib.assert(input.ispace.bounds == output.ispace.bounds, "input and output regions must be identical in size")
    var input_base = get_base(rect_t(input.ispace.bounds), __physical(input)[0], __fields(input)[0])
    var output_base = get_base(rect_t(output.ispace.bounds), __physical(output)[0], __fields(output)[0])
    var lo = input.ispace.bounds.lo:to_point()
    var hi = input.ispace.bounds.hi:to_point()
    @p = iface.plan {
      p = plan_dft(
        [data.range(dim):map(function(i) return rexpr hi.x[i] - lo.x[i] + 1 end end)],
        [&fftw_c.fftw_complex](input_base),
        [&fftw_c.fftw_complex](output_base),
        fftw_c.FFTW_FORWARD,
        fftw_c.FFTW_ESTIMATE),
      address_space = address_space,
    }
  end

  __demand(__inline)
  task iface.execute_plan(input : region(ispace(itype), dtype),
                          output : region(ispace(itype), dtype),
                          plan : region(ispace(int1d), iface.plan))
  where reads(input, plan), writes(output) do
    var p : int1d(iface.plan, plan)
    for x in plan do
      p = x
      break
    end

    var address_space = c.legion_processor_address_space(
      c.legion_runtime_get_executing_processor(__runtime(), __context()))
    regentlib.assert(p.address_space == address_space, "plans can only be used on the node where they are originally created")

    var input_base = get_base(rect_t(input.ispace.bounds), __physical(input)[0], __fields(input)[0])
    var output_base = get_base(rect_t(output.ispace.bounds), __physical(output)[0], __fields(output)[0])
    fftw_c.fftw_execute_dft(p.p, [&fftw_c.fftw_complex](input_base), [&fftw_c.fftw_complex](output_base))
  end

  __demand(__inline)
  task iface.destroy_plan(plan : region(ispace(int1d), iface.plan))
  where reads writes(plan) do
    var p : int1d(iface.plan, plan)
    for x in plan do
      p = x
      break
    end

    var address_space = c.legion_processor_address_space(
      c.legion_runtime_get_executing_processor(__runtime(), __context()))
    regentlib.assert(p.address_space == address_space, "plans can only be used on the node where they are originally created")

    fftw_c.fftw_destroy_plan(p.p)
  end

  return iface
end

return fft

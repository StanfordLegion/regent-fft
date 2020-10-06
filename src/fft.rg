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

  struct iface.plan {
    p : fftw_c.fftw_plan,
  }

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

  -- Important: overwrites input/output!
  __demand(__inline)
  task iface.make_plan(input : region(ispace(itype), dtype),
                       output : region(ispace(itype), dtype))
  where reads writes(input, output) do
    regentlib.assert(input.ispace.bounds == output.ispace.bounds, "input and output regions must be identical in size")
    var input_base = get_base(rect_t(input.ispace.bounds), __physical(input)[0], __fields(input)[0])
    var output_base = get_base(rect_t(output.ispace.bounds), __physical(output)[0], __fields(output)[0])
    return iface.plan {
      fftw_c.fftw_plan_dft_1d(
        input.ispace.volume,
        [&fftw_c.fftw_complex](input_base),
        [&fftw_c.fftw_complex](output_base),
        fftw_c.FFTW_FORWARD,
        fftw_c.FFTW_ESTIMATE)
    }
  end

  __demand(__inline)
  task iface.execute_plan(input : region(ispace(itype), dtype),
                          output : region(ispace(itype), dtype),
                          p : iface.plan)
  where reads(input), writes(output) do
    fftw_c.fftw_execute(p.p)
  end

  __demand(__inline)
  task iface.destroy_plan(p : iface.plan)
    fftw_c.fftw_destroy_plan(p.p)
  end

  return iface
end

local fft1d = fft.generate_fft_interface(int1d, complex64)

task main()
  var r = region(ispace(int1d, 128), complex64)
  var s = region(ispace(int1d, 128), complex64)
  fill(r, 0)
  fill(s, 0)
  var p = fft1d.make_plan(r, s)
  fill(r, 0)
  fill(s, 0)
  fft1d.execute_plan(r, s, p)
  fft1d.destroy_plan(p)
end
regentlib.start(main)

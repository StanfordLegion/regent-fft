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

terra get_base(rect : c.legion_rect_1d_t,
               physical : c.legion_physical_region_t,
               field : c.legion_field_id_t)
  var subrect : c.legion_rect_1d_t
  var offsets : c.legion_byte_offset_t[1]
  var accessor = c.legion_physical_region_get_field_accessor_array_1d(physical, field)
  var base_pointer = [&complex64](c.legion_accessor_array_1d_raw_rect_ptr(
                                      accessor, rect, &subrect, &(offsets[0])))
  regentlib.assert(base_pointer ~= nil, "base pointer is nil")
  regentlib.assert(subrect.lo.x[0] == rect.lo.x[0] and subrect.lo.x[1] == rect.lo.x[1], "subrect not equal to rect")
  regentlib.assert(offsets[0].offset == terralib.sizeof(complex64), "stride does not match expected value")

  c.legion_accessor_array_1d_destroy(accessor)

  return base_pointer
end

struct plan {
    p : fftw_c.fftw_plan,
}

-- Important: overwrites input/output!
__demand(__inline)
task make_plan(input : region(ispace(int1d), complex64),
               output : region(ispace(int1d), complex64))
where reads writes(input, output) do
  regentlib.assert(input.ispace.bounds == output.ispace.bounds, "input and output regions must be identical in size")
  var input_base = get_base(
    c.legion_rect_1d_t(input.ispace.bounds),
    __physical(input)[0],
    __fields(input)[0])
  var output_base = get_base(
    c.legion_rect_1d_t(output.ispace.bounds),
    __physical(output)[0],
    __fields(output)[0])
  return plan {
    fftw_c.fftw_plan_dft_1d(
      input.ispace.volume,
      [&fftw_c.fftw_complex](input_base),
      [&fftw_c.fftw_complex](output_base),
      fftw_c.FFTW_FORWARD,
      fftw_c.FFTW_ESTIMATE)
  }
end

__demand(__inline)
task execute_plan(input : region(ispace(int1d), complex64),
                  output : region(ispace(int1d), complex64),
                  p : plan)
where reads(input), writes(output) do
  fftw_c.fftw_execute(p.p)
end

__demand(__inline)
task destroy_plan(p : plan)
  fftw_c.fftw_destroy_plan(p.p)
end

task main()
  var r = region(ispace(int1d, 128), complex64)
  var s = region(ispace(int1d, 128), complex64)
  fill(r, 0)
  fill(s, 0)
  var p = make_plan(r, s)
  fill(r, 0)
  fill(s, 0)
  execute_plan(r, s, p)
  destroy_plan(p)
end
regentlib.start(main)

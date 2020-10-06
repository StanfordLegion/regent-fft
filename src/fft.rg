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

task main()
  var r = region(ispace(int1d, 128), complex64)
  var s = region(ispace(int1d, 128), complex64)
  fill(r, 0)
  fill(s, 0)
  -- Important: overwrites input/output!
  var p = fftw_c.fftw_plan_dft_1d(
    r.ispace.volume,
    [&fftw_c.fftw_complex](get_base(c.legion_rect_1d_t(r.ispace.bounds), __physical(r)[0], __fields(r)[0])),
    [&fftw_c.fftw_complex](get_base(c.legion_rect_1d_t(s.ispace.bounds), __physical(s)[0], __fields(s)[0])),
    fftw_c.FFTW_FORWARD,
    fftw_c.FFTW_ESTIMATE)
  fill(r, 0)
  fill(s, 0)
  fftw_c.fftw_execute(p)
  fftw_c.fftw_destroy_plan(p)
end
regentlib.start(main)

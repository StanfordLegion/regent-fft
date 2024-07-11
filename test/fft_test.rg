-- Copyright 2024 Stanford University
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

import "regent"

local fft = require("fft")

local cmapper = require("test_mapper")
local format = require("std/format")

-- PRINT FUNCTIONS

local function make_print_region_task(title, input)
  local t = regentlib.newsymbol(title, "t")
  local i = regentlib.newsymbol(input, "i")
  local task print_region_task([t], [i])
  where reads(i) do
    format.println("{} = [", t)
    for x in i do
      format.println("{},", i[x])
    end
    format.println("]")
  end
  return print_region_task
end

local print_region_1d_float = make_print_region_task(regentlib.string, region(ispace(int1d), float))
local print_region_1d_double = make_print_region_task(regentlib.string, region(ispace(int1d), double))
local print_region_1d_complex32 = make_print_region_task(regentlib.string, region(ispace(int1d), complex32))
local print_region_1d_complex64 = make_print_region_task(regentlib.string, region(ispace(int1d), complex64))
local print_region_2d_complex64 = make_print_region_task(regentlib.string, region(ispace(int2d), complex64))
local print_region_3d_double = make_print_region_task(regentlib.string, region(ispace(int3d), double))
local print_region_3d_complex64 = make_print_region_task(regentlib.string, region(ispace(int3d), complex64))
local print_region_4d_complex64 = make_print_region_task(regentlib.string, region(ispace(int4d), complex64))

-- COMPARISON FUNCTIONS

local function make_compare_regions_task(output, expected)
  local o = regentlib.newsymbol(output, "o")
  local e = regentlib.newsymbol(expected, "e")
  local task compare_region_task([o], [e])
  where reads(o, e) do
    var status = true
    for x in o do
      if (o[x].real ~= e[x].real) or (o[x].imag ~= e[x].imag) then
        status = false
        break
      end
    end
    return status
  end
  return compare_region_task
end

local compare_regions_1d_complex32 = make_compare_regions_task(region(ispace(int1d), complex32), region(ispace(int1d), complex32))
local compare_regions_1d_complex64 = make_compare_regions_task(region(ispace(int1d), complex64), region(ispace(int1d), complex64))
local compare_regions_2d_complex64 = make_compare_regions_task(region(ispace(int2d), complex64), region(ispace(int2d), complex64))
local compare_regions_3d_complex64 = make_compare_regions_task(region(ispace(int3d), complex64), region(ispace(int3d), complex64))
local compare_regions_4d_complex64 = make_compare_regions_task(region(ispace(int4d), complex64), region(ispace(int4d), complex64))

-- INTERFACES

local fft1d_float_complex32 = fft.generate_fft_interface(int1d, float, complex32, false)
local fft1d_complex32_complex32 = fft.generate_fft_interface(int1d, complex32, complex32, false)

local fft1d_double_complex64 = fft.generate_fft_interface(int1d, double, complex64, false)
local fft1d_complex64_complex64 = fft.generate_fft_interface(int1d, complex64, complex64, false)
local fft2d_complex64_complex64 = fft.generate_fft_interface(int2d, complex64, complex64, false)
local fft3d_complex64_complex64 = fft.generate_fft_interface(int3d, complex64, complex64, false)

local fft2d_batch_double_complex64 = fft.generate_fft_interface(int3d, double, complex64, false)
local fft2d_batch_complex64_complex64 = fft.generate_fft_interface(int2d, complex64, complex64, true)
local fft3d_batch_complex64_complex64 = fft.generate_fft_interface(int3d, complex64, complex64, true)

-- TEST FUNCTIONS

__demand(__inline)
task test_1d_float_to_complex32_transform()
  format.println(">> test_1d_float_to_complex32_transform")

  var r = region(ispace(int1d, 3), float)
  var s = region(ispace(int1d, 3), complex32)
  var p = region(ispace(int1d, 1), fft1d_float_complex32.plan)

  fill(r, 3)
  fill(s, 0)

  fft1d_float_complex32.make_plan(r, s, p)
  fft1d_float_complex32.execute_plan(r, s, p)
  fft1d_float_complex32.destroy_plan(p)

  -- Verify
  var e = region(ispace(int1d, 3), complex32)
  fill(e, 0)
  e[0].real = 9

  print_region_1d_float("Input", r)
  print_region_1d_complex32("Output", s)
  print_region_1d_complex32("Expected", e)

  var status = compare_regions_1d_complex32(s, e)
  format.println("<< test_1d_float_to_complex32_transform [PASSED? {}]", status)
  regentlib.assert(status, "test_1d_float_to_complex32_transform should pass")
end

__demand(__inline)
task test_1d_complex32_to_complex32_transform()
  format.println(">> test_1d_complex32_to_complex32_transform")
  var r = region(ispace(int1d, 3), complex32)
  var s = region(ispace(int1d, 3), complex32)
  var p = region(ispace(int1d, 1), fft1d_complex32_complex32.plan)

  for x in r do
    r[x].real = 3
    r[x].imag = 3
  end
  fill(s, 0)

  fft1d_complex32_complex32.make_plan(r, s, p)
  fft1d_complex32_complex32.execute_plan(r, s, p)
  fft1d_complex32_complex32.destroy_plan(p)

  -- Verify
  var e = region(ispace(int1d, 3), complex32)
  fill(e, 0)
  e[0].real = 9
  e[0].imag = 9

  print_region_1d_complex32("Input", r)
  print_region_1d_complex32("Output", s)
  print_region_1d_complex32("Expected", e)

  var status = compare_regions_1d_complex32(s, e)
  format.println("<< test_1d_complex32_to_complex32_transform [PASSED? {}]", status)
  regentlib.assert(status, "test_1d_complex32_to_complex32_transform should pass")
end

__demand(__inline)
task test_1d_double_to_complex64_transform()
  format.println(">> test_1d_double_to_complex64_transform")

  var r = region(ispace(int1d, 3), double)
  var s = region(ispace(int1d, 3), complex64)
  var p = region(ispace(int1d, 1), fft1d_double_complex64.plan)

  fill(r, 3)
  fill(s, 0)

  fft1d_double_complex64.make_plan(r, s, p)
  fft1d_double_complex64.execute_plan(r, s, p)
  fft1d_double_complex64.destroy_plan(p)

  -- Verify
  var e = region(ispace(int1d, 3), complex64)
  fill(e, 0)
  e[0].real = 9

  print_region_1d_double("Input", r)
  print_region_1d_complex64("Output", s)
  print_region_1d_complex64("Expected", e)

  var status = compare_regions_1d_complex64(s, e)
  format.println("<< test_1d_double_to_complex64_transform [PASSED? {}]", status)
  regentlib.assert(status, "test_1d_double_to_complex64_transform should pass")
end

__demand(__inline)
task test_1d_complex64_to_complex64_transform()
  format.println(">> test_1d_complex64_to_complex64_transform")

  var r = region(ispace(int1d, 5), complex64)
  var s = region(ispace(int1d, 5), complex64)
  var p = region(ispace(int1d, 1), fft1d_complex64_complex64.plan)

  for x in r do
    r[x].real = 3
    r[x].imag = 3
  end
  fill(s, 0)

  fft1d_complex64_complex64.make_plan(r, s, p)
  fft1d_complex64_complex64.execute_plan(r, s, p)
  fft1d_complex64_complex64.destroy_plan(p)

  -- Verify
  var e = region(ispace(int1d, 3), complex64)
  fill(e, 0)
  e[0].real = 15
  e[0].imag = 15

  print_region_1d_complex64("Input", r)
  print_region_1d_complex64("Output", s)
  print_region_1d_complex64("Expected", e)

  var status = compare_regions_1d_complex64(s, e)
  format.println("<< test_1d_complex64_to_complex64_transform [PASSED? {}]", status)
  regentlib.assert(status, "test_1d_complex64_to_complex64_transform should pass")
end

__demand(__inline)
task test_1d_complex64_to_complex64_distrib_transform()
  format.println(">> test_1d_complex64_to_complex64_distrib_transform")

  var n = fft1d_complex64_complex64.get_num_nodes()
  format.println("Num nodes in distrib is {}", n)

  var r = region(ispace(int1d, 3*n), complex64)
  var r_part = partition(equal, r, ispace(int1d, n))
  var s = region(ispace(int1d, 3*n), complex64)
  var s_part = partition(equal, s, ispace(int1d, n))
  var p = region(ispace(int1d, n), fft1d_complex64_complex64.plan)
  var p_part = partition(equal, p, ispace(int1d, n))

  for x in r do
    r[x].real = 4
    r[x].imag = 4
  end
  fill(s, 0)

  -- Important: this overwrites r and s!
  fft1d_complex64_complex64.make_plan_distrib(r, r_part, s, s_part, p, p_part)
  fft1d_complex64_complex64.execute_plan_distrib(r, r_part, s, s_part, p, p_part)
  fft1d_complex64_complex64.destroy_plan_distrib(p, p_part)

  -- Verify
  var e = region(ispace(int1d, 3), complex64)
  fill(e, 0)
  e[0].real = 12
  e[0].imag = 12

  print_region_1d_complex64("Input", r)
  print_region_1d_complex64("Output", s)
  print_region_1d_complex64("Expected", e)

  var status = compare_regions_1d_complex64(s, e)
  format.println("<< test_1d_complex64_to_complex64_distrib_transform [PASSED? {}]", status)
  regentlib.assert(status, "test_1d_complex64_to_complex64_distrib_transform should pass")
end

__demand(__inline)
task test_2d_complex64_to_complex64_transform()
  format.println(">> test_2d_complex64_to_complex64_transform")

  var r = region(ispace(int2d, { 2, 2 }), complex64)
  var s = region(ispace(int2d, { 2, 2 }), complex64)
  var p = region(ispace(int1d, 1), fft2d_complex64_complex64.plan)

  for x in r do
    r[x].real = 5
    r[x].imag = 5
  end
  fill(s, 0)

  fft2d_complex64_complex64.make_plan(r, s, p)
  fft2d_complex64_complex64.execute_plan(r, s, p)
  fft2d_complex64_complex64.destroy_plan(p)

  -- Verify
  var e = region(ispace(int2d, { 2, 2 }), complex64)
  fill(e, 0)
  e[{x=0, y=0}].real = 20
  e[{x=0, y=0}].imag = 20

  print_region_2d_complex64("Input", r)
  print_region_2d_complex64("Output", s)
  print_region_2d_complex64("Expected", e)

  var status = compare_regions_2d_complex64(s, e)
  format.println("<< test_2d_complex64_to_complex64_transform [PASSED? {}]", status)
  regentlib.assert(status, "test_2d_complex64_to_complex64_transform should pass")
end

__demand(__inline)
task test_3d_complex64_to_complex64_transform()
  format.println(">> test_3d_complex64_to_complex64_transform")

  var r = region(ispace(int3d, { 3, 3, 3 }), complex64)
  var s = region(ispace(int3d, { 3, 3, 3 }), complex64)
  var p = region(ispace(int1d, 1), fft3d_complex64_complex64.plan)

  for x in r do
    r[x].real = 3
    r[x].imag = 3
  end
  fill(s, 0)

  -- Important: this overwrites r and s!
  fft3d_complex64_complex64.make_plan(r, s, p)
  fft3d_complex64_complex64.execute_plan(r, s, p)
  fft3d_complex64_complex64.destroy_plan(p)

  -- Verify
  var e = region(ispace(int3d, { 3, 3, 3 }), complex64)
  fill(e, 0)
  e[{x=0, y=0, z=0}].real = 81
  e[{x=0, y=0, z=0}].imag = 81

  print_region_3d_complex64("Input", r)
  print_region_3d_complex64("Output", s)
  print_region_3d_complex64("Expected", e)

  var status = compare_regions_3d_complex64(s, e)
  format.println("<< test_3d_complex64_to_complex64_transform [PASSED? {}]", status)
  regentlib.assert(status, "test_3d_complex64_to_complex64_transform should pass")
end

__demand(__inline)
task test_2d_double_to_complex64_batch_transform()
  format.println(">> test_2d_double_to_complex64_batch_transform")

  var r = region(ispace(int3d, { 3, 3, 2 }), double)
  var s = region(ispace(int3d, { 3, 3, 2 }), complex64)
  var p = region(ispace(int1d, 1), fft2d_batch_double_complex64.plan)

  -- fill(r, 3)
  for x in r do
    r[x] = 3
  end
  fill(s, 0)

  fft2d_batch_double_complex64.make_plan_batch(r, s, p)
  fft2d_batch_double_complex64.execute_plan(r, s, p)
  fft2d_batch_double_complex64.destroy_plan(p)

  -- Verify
  var e = region(ispace(int3d, { 3, 3, 2 }), complex64)
  fill(e, 0)
  e[{x=0, y=0, z=0}].real = 27
  e[{x=0, y=0, z=1}].real = 27

  print_region_3d_double("Input", r)
  print_region_3d_complex64("Output", s)
  print_region_3d_complex64("Expected", e)

  var status = compare_regions_3d_complex64(s, e)
  format.println("<< test_2d_double_to_complex64_batch_transform [PASSED? {}]", status)
  regentlib.assert(status, "test_2d_double_to_complex64_batch_transform should pass")
end

__demand(__inline)
task test_2d_complex64_to_complex64_batch_transform()
  format.println(">> test_2d_complex64_to_complex64_batch_transform")

  var r = region(ispace(int3d, { 3, 3, 2 }), complex64)
  var s = region(ispace(int3d, { 3, 3, 2 }), complex64)
  var p = region(ispace(int1d, 1), fft2d_batch_complex64_complex64.plan)

  for x in r do
    r[x].real = 3
    r[x].imag = 3
  end
  fill(s, 0)

  fft2d_batch_complex64_complex64.make_plan_batch(r, s, p)
  fft2d_batch_complex64_complex64.execute_plan(r, s, p)
  fft2d_batch_complex64_complex64.destroy_plan(p)

  -- Verify
  var e = region(ispace(int3d, { 3, 3, 2 }), complex64)
  fill(e, 0)
  e[{x=0, y=0, z=0}].real = 27
  e[{x=0, y=0, z=0}].imag = 27
  e[{x=0, y=0, z=1}].real = 27
  e[{x=0, y=0, z=1}].imag = 27

  print_region_3d_complex64("Input", r)
  print_region_3d_complex64("Output", s)
  print_region_3d_complex64("Expected", e)

  var status = compare_regions_3d_complex64(s, e)
  format.println("<< test_2d_complex64_to_complex64_batch_transform [PASSED? {}]", status)
  regentlib.assert(status, "test_2d_complex64_to_complex64_batch_transform should pass")
end

__demand(__inline)
task test_3d_complex64_to_complex64_batch_transform()
  format.println(">> test_3d_complex64_to_complex64_batch_transform")

  var r = region(ispace(int4d, { 3, 3, 3, 2 }), complex64)
  var s = region(ispace(int4d, { 3, 3, 3, 2 }), complex64)
  var p = region(ispace(int1d, 1), fft3d_batch_complex64_complex64.plan)

  for x in r do
    r[x].real = 3
    r[x].imag = 3
  end
  fill(s, 0)

  fft3d_batch_complex64_complex64.make_plan_batch(r, s, p)
  fft3d_batch_complex64_complex64.execute_plan(r, s, p)
  fft3d_batch_complex64_complex64.destroy_plan(p)

  -- Verify
  var e = region(ispace(int4d, { 3, 3, 3, 2 }), complex64)
  fill(e, 0)
  e[{x=0, y=0, z=0, w=0}].real = 81
  e[{x=0, y=0, z=0, w=0}].imag = 81
  e[{x=0, y=0, z=0, w=1}].real = 81
  e[{x=0, y=0, z=0, w=1}].imag = 81

  print_region_4d_complex64("Input", r)
  print_region_4d_complex64("Output", s)
  print_region_4d_complex64("Expected", e)

  var status = compare_regions_4d_complex64(s, e)
  format.println("<< test_3d_complex64_to_complex64_batch_transform [PASSED? {}]", status)
  regentlib.assert(status, "test_3d_complex64_to_complex64_batch_transform should pass")
end

task main()
  test_1d_float_to_complex32_transform()
  test_1d_complex32_to_complex32_transform()

  test_1d_double_to_complex64_transform()
  test_1d_complex64_to_complex64_transform()
  test_1d_complex64_to_complex64_distrib_transform()
  test_2d_complex64_to_complex64_transform()
  test_3d_complex64_to_complex64_transform()

  test_2d_double_to_complex64_batch_transform()
  test_2d_complex64_to_complex64_batch_transform()
  test_3d_complex64_to_complex64_batch_transform()
end

regentlib.start(main, cmapper.register_mappers)

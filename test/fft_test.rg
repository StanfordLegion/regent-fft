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

import "regent"

local fft = require("fft")

local cmapper = require("test_mapper")

local fft1d = fft.generate_fft_interface(int1d, complex64)
local fft2d = fft.generate_fft_interface(int2d, complex64)

__demand(__inline)
task test1d()
  var r = region(ispace(int1d, 128), complex64)
  var s = region(ispace(int1d, 128), complex64)
  fill(r, 0)
  fill(s, 0)
  var p = region(ispace(int1d, 1), fft1d.plan)
  -- Important: this overwrites r and s!
  fft1d.make_plan(r, s, p)
  fill(r, 0)
  fill(s, 0)
  fft1d.execute_plan(r, s, p)
  fft1d.destroy_plan(p)
end

__demand(__inline)
task test1d_distrib()
  var n = fft1d.get_num_nodes()
  var r = region(ispace(int1d, 128*n), complex64)
  var r_part = partition(equal, r, ispace(int1d, n))
  var s = region(ispace(int1d, 128*n), complex64)
  var s_part = partition(equal, s, ispace(int1d, n))
  fill(r, 0)
  fill(s, 0)
  var p = region(ispace(int1d, n), fft1d.plan)
  var p_part = partition(equal, p, ispace(int1d, n))
  -- Important: this overwrites r and s!
  fft1d.make_plan_distrib(r, r_part, s, s_part, p, p_part)
  fill(r, 0)
  fill(s, 0)
  __demand(__index_launch)
  for i in r_part.colors do
    fft1d.execute_plan_task(r_part[i], s_part[i], p)
  end
  fft1d.destroy_plan_distrib(p, p_part)
end

__demand(__inline)
task test2d()
  var r = region(ispace(int2d, { 128, 128 }), complex64)
  var s = region(ispace(int2d, { 128, 128 }), complex64)
  fill(r, 0)
  fill(s, 0)
  -- Important: this overwrites r and s!
  var p = region(ispace(int1d, 1), fft2d.plan)
  fft2d.make_plan(r, s, p)
  fill(r, 0)
  fill(s, 0)
  fft2d.execute_plan(r, s, p)
  fft2d.destroy_plan(p)
end

__demand(__inline)
task test2d_distrib()
  var n = fft2d.get_num_nodes()
  var t : transform(2, 1)
  t[{0, 0}] = 0
  t[{1, 0}] = 128
  var e = rect2d { lo = int2d { 0, 0 }, hi = int2d { 128-1, 128-1 } }
  var r = region(ispace(int2d, { 128, 128*n }), complex64)
  var r_part = restrict(disjoint, r, t, e, ispace(int1d, n))
  var s = region(ispace(int2d, { 128, 128*n }), complex64)
  var s_part = restrict(disjoint, s, t, e, ispace(int1d, n))
  fill(r, 0)
  fill(s, 0)
  var p = region(ispace(int1d, n), fft2d.plan)
  var p_part = partition(equal, p, ispace(int1d, n))
  -- Important: this overwrites r and s!
  fft2d.make_plan_distrib(r, r_part, s, s_part, p, p_part)
  fill(r, 0)
  fill(s, 0)
  __demand(__index_launch)
  for i in r_part.colors do
    fft2d.execute_plan_task(r_part[i], s_part[i], p)
  end
  fft2d.destroy_plan_distrib(p, p_part)
end

task main()
  test1d()
  test1d_distrib()
  test2d()
  test2d_distrib()
end
regentlib.start(main, cmapper.register_mappers)

/* Copyright 2020 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "test_mapper.h"

#include "mappers/default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

///
/// Mapper
///

static LegionRuntime::Logger::Category log_fft_test_mapper("fft_test_mapper");

class FFTTestMapper : public DefaultMapper
{
public:
  FFTTestMapper(MapperRuntime *rt, Machine machine, Processor local,
             const char *mapper_name);
  virtual Memory default_policy_select_target_memory(MapperContext ctx,
                                             Processor target_proc,
                                             const RegionRequirement &req);
};

FFTTestMapper::FFTTestMapper(MapperRuntime *rt, Machine machine, Processor local,
                             const char *mapper_name)
  : DefaultMapper(rt, machine, local, mapper_name)
{
}

Memory FFTTestMapper::default_policy_select_target_memory(MapperContext ctx,
                                                       Processor target_proc,
                                                       const RegionRequirement &req)
{
  // Hack (Elliott): just force everything into zero-copy memory for now
  bool prefer_rdma = true; //((req.tag & DefaultMapper::PREFER_RDMA_MEMORY) != 0);

  // TODO: deal with the updates in machine model which will
  //       invalidate this cache
  std::map<Processor,Memory>::iterator it;
  if (prefer_rdma)
  {
	it = cached_rdma_target_memory.find(target_proc);
	if (it != cached_rdma_target_memory.end()) return it->second;
  } else {
    it = cached_target_memory.find(target_proc);
	if (it != cached_target_memory.end()) return it->second;
  }

  // Find the visible memories from the processor for the given kind
  Machine::MemoryQuery visible_memories(machine);
  visible_memories.has_affinity_to(target_proc);
  if (visible_memories.count() == 0)
  {
    log_fft_test_mapper.error("No visible memories from processor " IDFMT "! "
                     "This machine is really messed up!", target_proc.id);
    assert(false);
  }
  // Figure out the memory with the highest-bandwidth
  Memory best_memory = Memory::NO_MEMORY;
  unsigned best_bandwidth = 0;
  Memory best_rdma_memory = Memory::NO_MEMORY;
  unsigned best_rdma_bandwidth = 0;
  std::vector<Machine::ProcessorMemoryAffinity> affinity(1);
  for (Machine::MemoryQuery::iterator it = visible_memories.begin();
        it != visible_memories.end(); it++)
  {
    affinity.clear();
    machine.get_proc_mem_affinity(affinity, target_proc, *it,
				      false /*not just local affinities*/);
    assert(affinity.size() == 1);
    if (!best_memory.exists() || (affinity[0].bandwidth > best_bandwidth)) {
      best_memory = *it;
      best_bandwidth = affinity[0].bandwidth;
    }
    if ((it->kind() == Memory::REGDMA_MEM || it->kind() == Memory::Z_COPY_MEM) &&
	    (!best_rdma_memory.exists() ||
	     (affinity[0].bandwidth > best_rdma_bandwidth))) {
      best_rdma_memory = *it;
      best_rdma_bandwidth = affinity[0].bandwidth;
    }
  }
  assert(best_memory.exists());
  if (prefer_rdma)
  {
	if (!best_rdma_memory.exists()) best_rdma_memory = best_memory;
	cached_rdma_target_memory[target_proc] = best_rdma_memory;
	return best_rdma_memory;
  } else {
	cached_target_memory[target_proc] = best_memory;
	return best_memory;
  }
}

static void create_mappers(Machine machine, Runtime *runtime, const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    FFTTestMapper* mapper = new FFTTestMapper(runtime->get_mapper_runtime(),
                                        machine, *it, "fft_test_mapper");
    runtime->replace_default_mapper(mapper, *it);
  }
}

void register_mappers()
{
  Runtime::add_registration_callback(create_mappers);
}

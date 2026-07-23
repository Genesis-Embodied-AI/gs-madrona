/*
 * Copyright 2021-2023 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include "mw_gpu/const.hpp"
#include "mw_gpu/cu_utils.hpp"

namespace madrona {

Context::Context(WorldBase *world_data, const WorkerInit &init)
    : data_(world_data),
      world_id_(init.worldID)
{}

template <typename ArchetypeT>
Entity Context::makeEntity()
{
    uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();
    return makeEntity(archetype_id);
}

Entity Context::makeEntity(uint32_t archetype_id)
{
    StateManager *state_mgr = mwGPU::getStateManager();
    return state_mgr->makeEntityNow(world_id_, archetype_id);
}

template <typename ArchetypeT>
Loc Context::makeTemporary()
{
    uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();
    return makeTemporary(archetype_id);
}

Loc Context::makeTemporary(uint32_t archetype_id)
{
    StateManager *state_mgr = mwGPU::getStateManager();
    return state_mgr->makeTemporary(world_id_, archetype_id);
}

void Context::destroyEntity(Entity e)
{
    return mwGPU::getStateManager()->destroyEntityNow(e);
}

Loc Context::loc(Entity e) const
{
    return mwGPU::getStateManager()->getLoc(e);
}

template <typename ComponentT>
ComponentT & Context::get(Entity e)
{
    return mwGPU::getStateManager()->getUnsafe<ComponentT>(e);
}

template <typename ComponentT>
ComponentT & Context::get(Loc l)
{
    return mwGPU::getStateManager()->getUnsafe<ComponentT>(l);
}

template <typename ComponentT>
ResultRef<ComponentT> Context::getSafe(Entity e)
{
    return mwGPU::getStateManager()->get<ComponentT>(e);
}

template <typename ComponentT>
ResultRef<ComponentT> Context::getCheck(Entity e)
{
    return mwGPU::getStateManager()->get<ComponentT>(e);
}

template <typename ComponentT>
ResultRef<ComponentT> Context::getCheck(Loc l)
{
    return mwGPU::getStateManager()->get<ComponentT>(l);
}

template <typename ComponentT>
ComponentT & Context::getDirect(int32_t column_idx, Loc l)
{
    return mwGPU::getStateManager()->getDirect<ComponentT>(column_idx, l);
}

template <typename SingletonT>
SingletonT & Context::singleton()
{
    return mwGPU::getStateManager()->getSingleton<SingletonT>(world_id_);
}

inline void * Context::tmpAlloc(uint64_t num_bytes)
{
    return mwGPU::TmpAllocator::get().alloc(num_bytes);
}

template <typename... ComponentTs>
inline Query<ComponentTs...> Context::query() 
{
    return mwGPU::getStateManager()->query<ComponentTs...>();
}
    
template <typename... ComponentTs, typename Fn>
inline void Context::iterateQuery(Query<ComponentTs...> &query, Fn&& fn) 
{
    mwGPU::getStateManager()->iterateQuery<sizeof...(ComponentTs)>(
        world_id_.idx,
        query.getSharedRef(),
    [&](int32_t offset, auto ...raw_ptrs){
        // offset is a global offset computed from world offset and 
        // a count of the current entity within the archetype table.
        cuda::std::tuple typed_ptrs {
            (ComponentTs *)raw_ptrs
            ...
        };
        
        std::apply([&](auto ...ptrs) {
            fn(ptrs[offset] ...);
        }, typed_ptrs);
    });
}


}

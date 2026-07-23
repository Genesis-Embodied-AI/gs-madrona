/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

namespace madrona {

template <typename ArchetypeT>
Entity Context::makeEntity()
{
    return state_mgr_->makeEntityNow<ArchetypeT>(
        MADRONA_MW_COND(cur_world_id_,) *state_cache_);
}

Entity Context::makeEntity(uint32_t archetype_id)
{
    return state_mgr_->makeEntityNow(
        MADRONA_MW_COND(cur_world_id_,) *state_cache_, archetype_id);
}

template <typename ArchetypeT>
Loc Context::makeTemporary()
{
    return state_mgr_->makeTemporary<ArchetypeT>(
        MADRONA_MW_COND(cur_world_id_));
}

Loc Context::makeTemporary(uint32_t archetype_id)
{
    return state_mgr_->makeTemporary(MADRONA_MW_COND(cur_world_id_,)
                                     archetype_id);
}

void Context::destroyEntity(Entity e)
{
    state_mgr_->destroyEntityNow(MADRONA_MW_COND(cur_world_id_,)
                                 *state_cache_, e);
}

Loc Context::loc(Entity e) const
{
    return state_mgr_->getLoc(e);
}

template <typename ComponentT>
ComponentT & Context::get(Entity e)
{
    return state_mgr_->getUnsafe<ComponentT>(
        MADRONA_MW_COND(cur_world_id_,) e.id);
}

template <typename ComponentT>
ComponentT & Context::get(Loc l)
{
    return state_mgr_->getUnsafe<ComponentT>(
        MADRONA_MW_COND(cur_world_id_,) l);
}

template <typename ComponentT>
ResultRef<ComponentT> Context::getSafe(Entity e)
{
    return state_mgr_->get<ComponentT>(
        MADRONA_MW_COND(cur_world_id_,) e);
}

template <typename ComponentT>
ResultRef<ComponentT> Context::getCheck(Entity e)
{
    return state_mgr_->get<ComponentT>(
        MADRONA_MW_COND(cur_world_id_,) e);
}

template <typename ComponentT>
ResultRef<ComponentT> Context::getCheck(Loc l)
{
    return state_mgr_->get<ComponentT>(
        MADRONA_MW_COND(cur_world_id_,) l);
}

template <typename ComponentT>
ComponentT & Context::getDirect(int32_t column_idx, Loc loc)
{
    return state_mgr_->getDirect<ComponentT>(
        MADRONA_MW_COND(cur_world_id_,) column_idx, loc);
}

template <typename SingletonT>
SingletonT & Context::singleton()
{
    return state_mgr_->getSingleton<SingletonT>(MADRONA_MW_COND(cur_world_id_));
}

void * Context::tmpAlloc(uint64_t num_bytes)
{
    return state_mgr_->tmpAlloc(MADRONA_MW_COND(cur_world_id_,) num_bytes);
}

template <typename... ComponentTs>
Query<ComponentTs...> Context::query()
{
    return state_mgr_->query<ComponentTs...>();
}

template <typename Fn, typename... ComponentTs>
inline void Context::iterateQuery(const Query<ComponentTs...> &query, Fn &&fn)
{
    state_mgr_->iterateQuery(MADRONA_MW_COND(cur_world_id_, ) query,
        std::forward<Fn>(fn));
}


#ifdef MADRONA_MW_MODE
WorldID Context::worldID() const
{
    return WorldID { (int32_t)cur_world_id_ };
}
#endif


}

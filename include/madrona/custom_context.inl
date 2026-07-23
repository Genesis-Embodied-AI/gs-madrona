/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

namespace madrona {

template <typename ContextT, typename DataT>
CustomContext<ContextT, DataT>::CustomContext(DataT *world_data,
                                              const WorkerInit &worker_init)
    : Context(world_data, worker_init)
{}


}

//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_IMPL_KOKKOS_GRAPHNODETHENPOLICY_HPP
#define KOKKOS_IMPL_KOKKOS_GRAPHNODETHENPOLICY_HPP

#include <type_traits>

namespace Kokkos::Experimental {

template <typename WorkTag>
struct ThenPolicy {
  static_assert(std::is_empty_v<WorkTag> || std::is_void_v<WorkTag>);

  using work_tag = WorkTag;
};

}  // namespace Kokkos::Experimental

#endif  // KOKKOS_IMPL_KOKKOS_GRAPHNODETHENPOLICY_HPP

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

module;

#include <Kokkos_DynRankView.hpp>

export module kokkos.dyn_rank_view;

export {
  namespace Kokkos {
  using ::Kokkos::DynRankView;

  using ::Kokkos::is_dyn_rank_view;
  using ::Kokkos::is_dyn_rank_view_v;

  using ::Kokkos::Subdynrankview;
  using ::Kokkos::subdynrankview;
  using ::Kokkos::subview;

  using ::Kokkos::rank;

  using ::Kokkos::deep_copy;
  using ::Kokkos::realloc;
  using ::Kokkos::resize;

  using ::Kokkos::create_mirror;
  using ::Kokkos::create_mirror_view;
  using ::Kokkos::create_mirror_view_and_copy;

  using ::Kokkos::operator!=;
  using ::Kokkos::operator==;

  namespace Impl {  // FIXME
  using ::Kokkos::Impl::ApplyToViewOfStaticRank;
  using ::Kokkos::Impl::as_view_of_rank_n;
  }  // namespace Impl
  }  // namespace Kokkos
}

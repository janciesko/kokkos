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

#include <Kokkos_OffsetView.hpp>

export module kokkos.offset_view;

export {
  namespace Kokkos {
  namespace Experimental {
  using ::Kokkos::Experimental::OffsetView;

  using ::Kokkos::Experimental::is_offset_view;
  using ::Kokkos::Experimental::is_offset_view_v;

  using ::Kokkos::Experimental::index_list_type;
  using ::Kokkos::Experimental::IndexRange;

  using ::Kokkos::Experimental::operator==;
  using ::Kokkos::Experimental::operator!=;
  }  // namespace Experimental

  using ::Kokkos::create_mirror;
  using ::Kokkos::create_mirror_view;
  using ::Kokkos::create_mirror_view_and_copy;

  using ::Kokkos::deep_copy;

  using ::Kokkos::subview;
  }  // namespace Kokkos
}

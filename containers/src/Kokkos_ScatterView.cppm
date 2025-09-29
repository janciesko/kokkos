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

#include <Kokkos_ScatterView.hpp>

export module kokkos.scatter_view;

export {
  namespace Kokkos {

  namespace Experimental {
  using ::Kokkos::Experimental::ScatterView;

  using ::Kokkos::Experimental::contribute;
  using ::Kokkos::Experimental::create_scatter_view;

  using ::Kokkos::Experimental::is_scatter_view;
  using ::Kokkos::Experimental::is_scatter_view_v;

  using ::Kokkos::Experimental::ScatterDuplicated;
  using ::Kokkos::Experimental::ScatterNonDuplicated;

  using ::Kokkos::Experimental::ScatterAccess;

  using ::Kokkos::Experimental::ScatterAtomic;
  using ::Kokkos::Experimental::ScatterNonAtomic;

  using ::Kokkos::Experimental::ScatterMax;
  using ::Kokkos::Experimental::ScatterMin;
  using ::Kokkos::Experimental::ScatterProd;
  using ::Kokkos::Experimental::ScatterSum;
  }  // namespace Experimental

  namespace Impl::Experimental {  // FIXME
  using ::Kokkos::Impl::Experimental::DefaultContribution;
  using ::Kokkos::Impl::Experimental::DefaultDuplication;
  }  // namespace Impl::Experimental

  using ::Kokkos::realloc;
  using ::Kokkos::resize;
  }  // namespace Kokkos
}

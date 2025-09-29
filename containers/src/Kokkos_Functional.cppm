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

#include <Kokkos_Functional.hpp>

export module kokkos.functional;

export {
  namespace Kokkos {
  using ::Kokkos::equal_to;
  using ::Kokkos::greater;
  using ::Kokkos::greater_equal;
  using ::Kokkos::less;
  using ::Kokkos::less_equal;
  using ::Kokkos::not_equal_to;
  using ::Kokkos::pod_equal_to;
  using ::Kokkos::pod_hash;
  using ::Kokkos::pod_not_equal_to;
  }  // namespace Kokkos
}

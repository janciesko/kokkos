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

#include <TestAtomicOperations.hpp>

namespace Test {
KOKKOS_IMPL_DISABLE_UNREACHABLE_WARNINGS_PUSH()
TEST(TEST_CATEGORY, atomic_operations_int16) {
  // FIXME_OPENMPTARGET - causes runtime failure with CrayClang compiler
#if defined(KOKKOS_COMPILER_CRAY_LLVM) && defined(KOKKOS_ENABLE_OPENMPTARGET)
  GTEST_SKIP() << "known to fail with OpenMPTarget+Cray LLVM";
#endif
  // FIXME_OPENACC - does not support atomic operations on int16_t data
#if defined(KOKKOS_ENABLE_OPENACC) && defined(KOKKOS_COMPILER_NVHPC)
  GTEST_SKIP() << "unsupported atomic data type for OpenACC+NVHPC";
#endif
  const int16_t start = -5;
  const int16_t end   = 11;
  for (int16_t i = start; i < end; ++i) {
    for (int16_t t = 0; t < 16; t++)
      ASSERT_TRUE((TestAtomicOperations::AtomicOperationsTestIntegralType<
                   int16_t, TEST_EXECSPACE>(i, end - i + start, t)));
  }
}
KOKKOS_IMPL_DISABLE_UNREACHABLE_WARNINGS_POP()
}  // namespace Test

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

#ifndef KOKKOS_DESUL_ATOMICS_CONFIG_HPP
#define KOKKOS_DESUL_ATOMICS_CONFIG_HPP

#include <impl/Kokkos_NvidiaGpuArchitectures.hpp>

#ifdef KOKKOS_IMPL_ARCH_NVIDIA_GPU

#if KOKKOS_IMPL_ARCH_NVIDIA_GPU < 60
#define DESUL_CUDA_ARCH_IS_PRE_PASCAL
#endif

#if KOKKOS_IMPL_ARCH_NVIDIA_GPU < 70
#define DESUL_CUDA_ARCH_IS_PRE_VOLTA
#endif

#if KOKKOS_IMPL_ARCH_NVIDIA_GPU < 90
#define DESUL_CUDA_ARCH_IS_PRE_HOPPER
#endif

#endif

#endif

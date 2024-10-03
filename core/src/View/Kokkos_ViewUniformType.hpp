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

#ifndef KOKKOS_EXPERIMENTAL_VIEWUNIFORMTYPE_HPP
#define KOKKOS_EXPERIMENTAL_VIEWUNIFORMTYPE_HPP

#include <Kokkos_Macros.hpp>

namespace Kokkos {
namespace Impl {
template <class ScalarType, int Rank>
struct ViewScalarToDataType {
  using type = typename ViewScalarToDataType<ScalarType, Rank - 1>::type *;
  using const_type =
      typename ViewScalarToDataType<ScalarType, Rank - 1>::const_type *;
};

template <class ScalarType>
struct ViewScalarToDataType<ScalarType, 0> {
  using type       = ScalarType;
  using const_type = const ScalarType;
};

template <class LayoutType, int Rank>
struct ViewUniformLayout {
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
  using array_layout KOKKOS_DEPRECATED_WITH_COMMENT(
      "Use layout_type instead.") = LayoutType;
#endif
  using layout_type = LayoutType;
};

template <class LayoutType>
struct ViewUniformLayout<LayoutType, 0> {
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
  using array_layout KOKKOS_DEPRECATED_WITH_COMMENT(
      "Use layout_type instead.") = Kokkos::LayoutLeft;
#endif
  using layout_type = Kokkos::LayoutLeft;
};

template <>
struct ViewUniformLayout<Kokkos::LayoutRight, 1> {
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
  using array_layout KOKKOS_DEPRECATED_WITH_COMMENT(
      "Use layout_type instead.") = Kokkos::LayoutRight;
#endif
  using layout_type = Kokkos::LayoutRight;
};

template <class ViewType, int Traits>
struct ViewUniformType {
  using data_type       = typename ViewType::data_type;
  using const_data_type = typename ViewType::const_data_type;
  using runtime_data_type =
      typename ViewScalarToDataType<typename ViewType::value_type,
                                    ViewType::rank>::type;
  using runtime_const_data_type =
      typename ViewScalarToDataType<typename ViewType::value_type,
                                    ViewType::rank>::const_type;

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
  using array_layout KOKKOS_DEPRECATED_WITH_COMMENT(
      "Use layout_type instead.") =
      typename ViewUniformLayout<typename ViewType::array_layout,
                                 ViewType::rank>::array_layout;
#endif
  using layout_type = typename ViewUniformLayout<typename ViewType::layout_type,
                                                 ViewType::rank>::layout_type;
  using device_type = typename ViewType::device_type;
  using anonymous_device_type =
      typename Kokkos::Device<typename device_type::execution_space,
                              Kokkos::AnonymousSpace>;

  using memory_traits = typename Kokkos::MemoryTraits<Traits>;
  using type = Kokkos::View<data_type, layout_type, device_type, memory_traits>;
  using const_type =
      Kokkos::View<const_data_type, layout_type, device_type, memory_traits>;
  using runtime_type =
      Kokkos::View<runtime_data_type, layout_type, device_type, memory_traits>;
  using runtime_const_type = Kokkos::View<runtime_const_data_type, layout_type,
                                          device_type, memory_traits>;

  using nomemspace_type = Kokkos::View<data_type, layout_type,
                                       anonymous_device_type, memory_traits>;
  using const_nomemspace_type =
      Kokkos::View<const_data_type, layout_type, anonymous_device_type,
                   memory_traits>;
  using runtime_nomemspace_type =
      Kokkos::View<runtime_data_type, layout_type, anonymous_device_type,
                   memory_traits>;
  using runtime_const_nomemspace_type =
      Kokkos::View<runtime_const_data_type, layout_type, anonymous_device_type,
                   memory_traits>;
};
}  // namespace Impl
}  // namespace Kokkos

#endif

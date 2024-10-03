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

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif

#ifndef KOKKOS_EXPERIMENTAL_MDSPAN_LAYOUT_HPP
#define KOKKOS_EXPERIMENTAL_MDSPAN_LAYOUT_HPP

#include "Kokkos_MDSpan_Extents.hpp"
#include <View/Kokkos_ViewDataAnalysis.hpp>

// The difference between a legacy Kokkos array layout and an
// mdspan layout is that the array layouts can have state, but don't have the
// nested mapping. This file provides interoperability helpers.

namespace Kokkos::Impl {

template <class LayoutType>
struct LayoutFromLayoutType;

template <>
struct LayoutFromLayoutType<Kokkos::LayoutLeft> {
  using type = Kokkos::Experimental::layout_left_padded<dynamic_extent>;
};

template <>
struct LayoutFromLayoutType<Kokkos::LayoutRight> {
  using type = Kokkos::Experimental::layout_right_padded<dynamic_extent>;
};

template <>
struct LayoutFromLayoutType<Kokkos::LayoutStride> {
  using type = layout_stride;
};

template <class LayoutType, class MDSpanType>
KOKKOS_INLINE_FUNCTION auto layout_type_from_mapping(
    const typename MDSpanType::mapping_type &mapping) {
  using mapping_type = typename MDSpanType::mapping_type;
  using extents_type = typename mapping_type::extents_type;

  constexpr auto rank = extents_type::rank();
  const auto &ext     = mapping.extents();

  static_assert(rank <= LAYOUT_TYPE_MAX_RANK,
                "Unsupported rank for mdspan (must be <= 8)");

  if constexpr (std::is_same_v<LayoutType, LayoutStride>) {
    return Kokkos::LayoutStride{
        rank > 0 ? ext.extent(0) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 0 ? mapping.stride(0) : 0,
        rank > 1 ? ext.extent(1) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 1 ? mapping.stride(1) : 0,
        rank > 2 ? ext.extent(2) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 2 ? mapping.stride(2) : 0,
        rank > 3 ? ext.extent(3) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 3 ? mapping.stride(3) : 0,
        rank > 4 ? ext.extent(4) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 4 ? mapping.stride(4) : 0,
        rank > 5 ? ext.extent(5) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 5 ? mapping.stride(5) : 0,
        rank > 6 ? ext.extent(6) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 6 ? mapping.stride(6) : 0,
        rank > 7 ? ext.extent(7) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 7 ? mapping.stride(7) : 0,
    };
  } else {
    LayoutType layout{rank > 0 ? ext.extent(0) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                      rank > 1 ? ext.extent(1) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                      rank > 2 ? ext.extent(2) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                      rank > 3 ? ext.extent(3) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                      rank > 4 ? ext.extent(4) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                      rank > 5 ? ext.extent(5) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                      rank > 6 ? ext.extent(6) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                      rank > 7 ? ext.extent(7) : KOKKOS_IMPL_CTOR_DEFAULT_ARG};

    if constexpr (rank > 1 &&
                  std::is_same_v<typename mapping_type::layout_type,
                                 Kokkos::Experimental::layout_left_padded<
                                     dynamic_extent>>) {
      layout.stride = mapping.stride(1);
    }
    if constexpr (std::is_same_v<typename mapping_type::layout_type,
                                 Kokkos::Experimental::layout_right_padded<
                                     dynamic_extent>>) {
      if constexpr (rank == 2) {
        layout.stride = mapping.stride(0);
      }
      if constexpr (rank > 2) {
        if (mapping.stride(rank - 2) != mapping.extents().extent(rank - 1))
          Kokkos::abort(
              "Invalid conversion from layout_right_padded to LayoutRight");
      }
    }
    return layout;
  }
#ifdef KOKKOS_COMPILER_INTEL
  __builtin_unreachable();
#endif
}

template <class MappingType, class LayoutType, size_t... Idx>
KOKKOS_INLINE_FUNCTION auto mapping_from_layout_type_impl(
    LayoutType layout, std::index_sequence<Idx...>) {
  using index_type   = typename MappingType::index_type;
  using extents_type = typename MappingType::extents_type;
  if constexpr (std::is_same_v<typename MappingType::layout_type,
                               layout_left> ||
                std::is_same_v<typename MappingType::layout_type,
                               layout_right>) {
    return MappingType{
        extents_type{dextents<index_type, MappingType::extents_type::rank()>{
            layout.dimension[Idx]...}}};
  } else {
    if (layout.stride == KOKKOS_IMPL_CTOR_DEFAULT_ARG ||
        extents_type::rank() < 2) {
      return MappingType{
          extents_type{dextents<index_type, MappingType::extents_type::rank()>{
              layout.dimension[Idx]...}}};
    } else {
      if constexpr (std::is_same_v<LayoutType, LayoutRight> &&
                    extents_type::rank() > 2) {
        size_t product_of_dimensions = 1;
        for (size_t r = 1; r < extents_type::rank(); r++)
          product_of_dimensions *= layout.dimension[r];
        if (product_of_dimensions != layout.stride)
          Kokkos::abort(
              "Invalid conversion from LayoutRight to layout_right_padded");
      } else {
        return MappingType{
            extents_type{
                dextents<index_type, MappingType::extents_type::rank()>{
                    layout.dimension[Idx]...}},
            layout.stride};
      }
    }
  }
}
template <class MappingType, size_t... Idx>
KOKKOS_INLINE_FUNCTION auto mapping_from_layout_type_impl(
    LayoutStride layout, std::index_sequence<Idx...>) {
  static_assert(
      std::is_same_v<typename MappingType::layout_type, layout_stride>);
  using index_type = typename MappingType::index_type;
  index_type strides[MappingType::extents_type::rank()] = {
      layout.stride[Idx]...};
  return MappingType{
      mdspan_non_standard_tag(),
      static_cast<typename MappingType::extents_type>(
          dextents<index_type, MappingType::extents_type::rank()>{
              layout.dimension[Idx]...}),
      strides};
}

// specialization for rank 0 to avoid empty array
template <class MappingType>
KOKKOS_INLINE_FUNCTION auto mapping_from_layout_type_impl(
    LayoutStride, std::index_sequence<>) {
  return MappingType{};
}

template <class MappingType, class LayoutType>
KOKKOS_INLINE_FUNCTION auto mapping_from_layout_type(LayoutType layout) {
  return mapping_from_layout_type_impl<MappingType>(
      layout, std::make_index_sequence<MappingType::extents_type::rank()>());
}

template <class MDSpanType, class VM>
KOKKOS_INLINE_FUNCTION auto mapping_from_view_mapping(const VM &view_mapping) {
  using mapping_type = typename MDSpanType::mapping_type;
  using extents_type = typename mapping_type::extents_type;

  // std::span is not available in C++17 (our current requirements),
  // so we need to use the std::array constructor for layout mappings.
  // FIXME When C++20 is available, we can use std::span here instead
  std::size_t strides[VM::Rank];
  view_mapping.stride_fill(&strides[0]);
  if constexpr (std::is_same_v<typename mapping_type::layout_type,
                               Kokkos::layout_stride>) {
    return mapping_type(Kokkos::mdspan_non_standard,
                        extents_from_view_mapping<extents_type>(view_mapping),
                        strides);
  } else if constexpr (VM::Rank > 1 &&
                       std::is_same_v<typename mapping_type::layout_type,
                                      Kokkos::Experimental::layout_left_padded<
                                          Kokkos::dynamic_extent>>) {
    return mapping_type(extents_from_view_mapping<extents_type>(view_mapping),
                        strides[1]);
  } else if constexpr (VM::Rank > 1 &&
                       std::is_same_v<typename mapping_type::layout_type,
                                      Kokkos::Experimental::layout_right_padded<
                                          Kokkos::dynamic_extent>>) {
    return mapping_type(extents_from_view_mapping<extents_type>(view_mapping),
                        strides[VM::Rank - 2]);
  } else {
    return mapping_type(extents_from_view_mapping<extents_type>(view_mapping));
  }
#ifdef KOKKOS_COMPILER_INTEL
  __builtin_unreachable();
#endif
}

}  // namespace Kokkos::Impl

#endif  // KOKKOS_EXPERIMENTAL_MDSPAN_LAYOUT_HPP

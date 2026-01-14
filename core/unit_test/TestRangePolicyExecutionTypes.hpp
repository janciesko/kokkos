// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

namespace Test {

enum nesting_level {
  zero /*no nesting*/,
  one /*team-level nesting*/,
  two /*thread-level nesting*/,
  three /*vector-level nesting*/,
  four /*any other nesting level serializes execution*/
};

namespace Experimental {
class Thread_handle;
class Vector_handle;
class Serial_handle;
}  // namespace Experimental

using t_h = Experimental::Thread_handle;
using v_h = Experimental::Vector_handle;
using s_h = Experimental::Serial_handle;

template <typename T>
concept isTeamHandle = std::is_same_v<T, Kokkos::TeamPolicy<>::member_type>;
template <typename T>
concept isThreadHandle = std::is_same_v<T, t_h>;
template <typename T>
concept isVectorHandle = std::is_same_v<T, v_h>;
template <typename T>
concept isSerialHandle = std::is_same_v<T, s_h>;

template <typename T>
concept isTeamOrExecSpace = isTeamHandle<T> || Kokkos::is_execution_space_v<T>;

template <nesting_level Level, class ExecType, class X, class Y>
KOKKOS_INLINE_FUNCTION void sum_views(const ExecType& handle, const X& x,
                                      const Y& y) {
  assert(isTeamHandle<ExecType>() || isThreadHandle<ExecType>() ||
         isVectorHandle<ExecType>() || isSerialHandle<ExecType>());
  if constexpr (Level == nesting_level::zero || Level == nesting_level::one) {
    auto policy = Kokkos::RangePolicy(handle, 0, x.extent(0));
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(const int& i) { x(i) += y(i); });
  } else if constexpr (Level == nesting_level::two) {
    assert(isTeamHandle<ExecType>() || isThreadHandle<ExecType>() ||
           isVectorHandle<ExecType>());
    auto policy = Kokkos::RangePolicy(handle, 0, x.extent(0));
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(const ExecType& handle) {
          sum_views<nesting_level::one>(
              handle,
              Kokkos::subview(x,
                              handle.get_resouce_id(),  // get the right exec
                                                        // resource handle index
                              Kokkos::ALL(), Kokkos::ALL()),
              Kokkos::subview(y, handle.get_resouce_id(), Kokkos::ALL(),
                              Kokkos::ALL()));
        });
  } else if constexpr (Level == nesting_level::three) {
    assert(isTeamHandle<ExecType>() || isThreadHandle<ExecType>());
    auto policy = Kokkos::RangePolicy(handle, 0, x.extent(0));
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(const ExecType& handle) {
          sum_views<nesting_level::two>(
              handle,
              Kokkos::subview(x,
                              handle.get_resouce_id(),  // get the right exec
                                                        // resource handle index
                              Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()),
              Kokkos::subview(y, handle.get_resouce_id(), Kokkos::ALL(),
                              Kokkos::ALL(), Kokkos::ALL()));
        });
  } else if constexpr (Level == nesting_level::four) {
    assert(isTeamHandle<ExecType>());
    auto policy = Kokkos::RangePolicy(handle, 0, x.extent(0));
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(const ExecType& handle) {
          sum_views<nesting_level::three>(
              handle,
              Kokkos::subview(
                  x, handle.get_resouce_id(),  // get the right exec resource
                                               // handle index
                  Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()),
              Kokkos::subview(y, handle.get_resouce_id(), Kokkos::ALL(),
                              Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()));
        });
  }
}

template <int>
void test_self_similar_range_policy_computation();

template <>
void test_self_similar_range_policy_computation<nesting_level::zero>() {
  int dim0 = 7;  // e.g. some work per team

  Kokkos::View<float*> v_x("v_x", dim0), v_y("v_y", dim0);
  Kokkos::deep_copy(v_x, 1);
  Kokkos::deep_copy(v_y, 2);

  // No nesting (nesting level 0)
  sum_views<nesting_level::zero>(Kokkos::DefaultExecutionSpace(), v_x, v_y);

  // check
  size_t result = 0;
  Kokkos::parallel_reduce(
      "Check", v_x.extent(0),
      KOKKOS_LAMBDA(int i, size_t& val) { val += v_x(i); }, result);

  ASSERT_EQ(result, size_t(3) * v_x.extent(0));
}

template <>
void test_self_similar_range_policy_computation<nesting_level::one>() {
  int dim0 = 5;  // e.g. num teams
  int dim1 = 7;  // e.g. some work per team

  Kokkos::View<float**> M_x("M_x", dim0, dim1), M_y("M_y", dim0, dim1);
  Kokkos::deep_copy(M_x, 1);
  Kokkos::deep_copy(M_y, 2);

  using handle_t = typename Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for(
      "apxyFromTeam", Kokkos::TeamPolicy(dim0, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const handle_t& handle) {
        sum_views<nesting_level::one>(
            handle, Kokkos::subview(M_x, handle.league_rank(), Kokkos::ALL()),
            Kokkos::subview(M_y, handle.league_rank(), Kokkos::ALL()));
      });

  // check
  size_t result = 0;
  Kokkos::parallel_reduce(
      "Check", M_x.extent(0),
      KOKKOS_LAMBDA(int i, size_t& val) {
        for (int j = 0; j < M_x.extent_int(1); j++) val += M_x(i, j);
      },
      result);

  ASSERT_EQ(result, size_t(3) * M_x.extent(0) * M_x.extent(1));
}

#if 0
template <>
void test_self_similar_range_policy_computation<nesting_level::two>() {
  int dim0 = 7; //e.g. num teams
  int dim1 = 5; //e.g. num threads per team
  int dim2 = 5; //e.g. some work per thread

  Kokkos::View<float***> M_x("M_x", dim0, dim1, dim2), M_y("M_y", dim0, dim1, dim2);
  Kokkos::deep_copy(M_x, 1);
  Kokkos::deep_copy(M_y, 2);

  // Nesting level 2
  using team_t = typename Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for(
      "apxyFromTeam", Kokkos::TeamPolicy(num_teams, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const team_t& team) {
        // call sum_views(TeamHandle) (nesting level 2)
        sum_views<nesting_level::two>(team, Kokkos::subview(M_x, team.league_rank(), Kokkos::ALL(), Kokkos::ALL()),
                  Kokkos::subview(M_y, team.league_rank(), Kokkos::ALL(), Kokkos::ALL()));
      });
  // check
  size_t result = 0;
  Kokkos::parallel_reduce(
      "Check", M_x.extent(0),
      KOKKOS_LAMBDA(int i, size_t& val) {
        for (int j = 0; j < M_x.extent_int(1); j++)
          for (int k = 0; k < M_x.extent_int(2); k++) val += M_x(i, j, k);
      },
      result);
  ASSERT_EQ(result, size_t(3) * M_x.extent(0) * M_x.extent(1));
}

template <>
void test_self_similar_range_policy_computation<nesting_level::three>() {
  int dim0 = 7; //e.g. num teams
  int dim1 = 5; //e.g. num threads per team
  int dim2 = 5; //e.g. some work per thread
  int dim3 = 6; //e.g. some work per vector lane

  Kokkos::View<float****> M_x("M_x", dim0, dim1, dim2, dim3), M_y("M_y", dim0, dim1, dim2, dim3);
  Kokkos::deep_copy(M_x, 1);
  Kokkos::deep_copy(M_y, 2);

  // Nesting level 3
  using team_t = typename Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for(
      "apxyFromTeam", Kokkos::TeamPolicy(num_teams, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const team_t& team) {
        // call sum_views(TeamHandle) (nesting level 3)
        sum_views<nesting_level::three>(team, Kokkos::subview(M_x, team.league_rank(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()),
                  Kokkos::subview(M_y, team.league_rank(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()));
      });
  
  // check
  size_t result = 0;
  Kokkos::parallel_reduce(
      "Check", M_x.extent(0),
      KOKKOS_LAMBDA(int i, size_t& val) {
        for (int j = 0; j < M_x.extent_int(1); j++)
          for (int k = 0; k < M_x.extent_int(2); k++)
            for (int l = 0; l < M_x.extent_int(3); l++) val += M_x(i, j, k, l);
      },
      result);
  ASSERT_EQ(result, size_t(3) * M_x.extent(0) * M_x.extent(1));
}

template <>
void test_self_similar_range_policy_computation<nesting_level::four>() {
  int dim0 = 7; //e.g. num teams
  int dim1 = 5; //e.g. num threads per team
  int dim2 = 5; //e.g. some work per thread
  int dim3 = 6; //e.g. some work per vector lane
  int dim4 = 7; //e.g. some work for serial execution

  Kokkos::View<float*****> M_x("M_x", dim0, dim1, dim2, dim3, dim4), M_y("M_y", dim0, dim1, dim2, dim3, dim4);
  Kokkos::deep_copy(M_x, 1);
  Kokkos::deep_copy(M_y, 2);

  // Nesting level 4
  using team_t = typename Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for(
      "apxyFromTeam", Kokkos::TeamPolicy(num_teams, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const team_t& team) {
        // call sum_views(TeamHandle) (nesting level 4)
        sum_views<nesting_level::four>(team, Kokkos::subview(M_x, team.league_rank(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()),
                  Kokkos::subview(M_y, team.league_rank(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()));
      });
  // check
  size_t result = 0;
  Kokkos::parallel_reduce(
      "Check", M_x.extent(0),
      KOKKOS_LAMBDA(int i, size_t& val) {
        for (int j = 0; j < M_x.extent_int(1); j++)
          for (int k = 0; k < M_x.extent_int(2); k++)
            for (int l = 0; l < M_x.extent_int(3); l++)
              for (int m = 0; m < M_x.extent_int(4); m++) val += M_x(i, j, k, l, m);
      },
      result);
  ASSERT_EQ(result, size_t(3) * M_x.extent(0) * M_x.extent(1));
}

#endif

TEST(TEST_CATEGORY, self_similar_range_policy_computation) {
  test_self_similar_range_policy_computation<nesting_level::zero>();
  test_self_similar_range_policy_computation<nesting_level::one>();
#if 0
  test_self_similar_range_policy_computation<nesting_level::two>();
  test_self_similar_range_policy_computation<nesting_level::three>();
  test_self_similar_range_policy_computation<nesting_level::four>();
#endif
}

}  // namespace Test

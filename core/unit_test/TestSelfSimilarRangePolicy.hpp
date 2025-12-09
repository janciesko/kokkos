// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

namespace Test {

template <class Policy>
KOKKOS_INLINE_FUNCTION int check_runtime_inputs(
    Policy& p, const typename Policy::index_type expected_begin,
    const typename Policy::index_type expected_end,
    const typename Policy::index_type chunk_size = 0) {
  int nerrs = 0;

  if (p.begin() != expected_begin) ++nerrs;
  if (p.end() != expected_end) ++nerrs;

  auto p2 = p.set_chunk_size(chunk_size);
  if constexpr (Kokkos::ExecutionSpace<typename Policy::execution_type>)
    if (p2.chunk_size() != chunk_size) ++nerrs;

  return nerrs;
}

void test_self_similar_range_policy_runtime() {
  using IndexType = typename Kokkos::DefaultExecutionSpace::size_type;

  IndexType beg        = 5;
  IndexType end        = 15;
  IndexType chunk_size = 10;

  auto p_execspace =
      Kokkos::RangePolicy(Kokkos::DefaultExecutionSpace(), beg, end);
  auto nerrs_exec_space =
      check_runtime_inputs(p_execspace, beg, end, chunk_size);
  ASSERT_EQ(nerrs_exec_space, 0);

  int nerrs_team_handle;
  using team_t = typename Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_reduce(
      "check_runtime", Kokkos::TeamPolicy(1, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const team_t& team, int& nerrs) {
        auto p_teamhandle = Kokkos::RangePolicy(team, beg, end);
        auto tvr          = Kokkos::TeamVectorRange(team, beg, end);
        nerrs = check_runtime_inputs(p_teamhandle, tvr.start, tvr.end);
      },
      nerrs_team_handle);
  ASSERT_EQ(nerrs_team_handle, 0);
}

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

enum nesting_level { one, two, three };

template <int>
struct Self_similar_range_policy;

template <>
struct Self_similar_range_policy<nesting_level::one> {
  struct Tag_1 {};
  struct Tag_2 {};

  using range_policy_1_t = Kokkos::RangePolicy<TEST_EXECSPACE, Tag_1>;
  using range_policy_2_t = Kokkos::RangePolicy<TEST_EXECSPACE, Tag_2>;

  using View_1D = Kokkos::View<float*>;
  using View_2D = Kokkos::View<float**>;
  View_1D v_x, v_y;
  View_2D M_x, M_y;

  // sum_views over a team (nesting level 0 or 1)
  template <class X, class Y>
  KOKKOS_INLINE_FUNCTION void sum_views(const auto& handle, const X& x,
                                        const Y& y) const {
    auto policy = Kokkos::RangePolicy(handle, 0, x.extent(0));
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(const int& i) { x(i) += y(i); });
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const auto& team) const {
    sum_views(team, Kokkos::subview(M_x, team.league_rank(), Kokkos::ALL()),
              Kokkos::subview(M_y, team.league_rank(), Kokkos::ALL()));
  };

  KOKKOS_INLINE_FUNCTION
  void operator()(const Tag_1&, size_t i, size_t& red) const { red += v_x(i); }

  KOKKOS_INLINE_FUNCTION
  void operator()(const Tag_2&, size_t i, size_t& red) const {
    for (int j = 0; j < M_x.extent_int(1); j++) red += M_x(i, j);
  }

  Self_similar_range_policy() { run(); };

  void run() {
    int N         = 7;
    int num_teams = 5;

    v_x = View_1D("v_x", N);
    v_y = View_1D("v_y", N);
    M_x = View_2D("M_x", num_teams, N);
    M_y = View_2D("M_y", num_teams, N);

    Kokkos::deep_copy(v_x, 1);
    Kokkos::deep_copy(v_y, 2);
    Kokkos::deep_copy(M_x, 1);
    Kokkos::deep_copy(M_y, 2);

    // Call sum_views(ExecSpace)
    sum_views(Kokkos::DefaultExecutionSpace(), v_x, v_y);

    // call sum_views(TeamHandle, nested)
    Kokkos::parallel_for("apxyFromTeam",
                         Kokkos::TeamPolicy(num_teams, Kokkos::AUTO()), *this);

    size_t result = 0;
    Kokkos::parallel_reduce("Check1", range_policy_1_t(0, v_x.extent(0)), *this,
                            result);
    ASSERT_EQ(result, size_t(3) * v_x.extent(0));
    Kokkos::parallel_reduce("Check2", range_policy_2_t(0, M_x.extent(0)), *this,
                            result);
    ASSERT_EQ(result, size_t(3) * M_x.extent(0) * M_x.extent(1));
  }
};

#if 0  // Protype of nested levels


template <>
struct Self_similar_range_policy<nesting_level::two> {
  struct Tag_0 {};
  struct Tag_1 {};
  struct Tag_2 {};
  using range_policy_0_t = Kokkos::RangePolicy<Kokkos::TeamPolicy<>::member_type, Tag_0>;
  using range_policy_1_t = Kokkos::RangePolicy<Thread_handle_type, Tag_1>;
  using range_policy_2_t = Kokkos::RangePolicy<TEST_EXECSPACE, Tag_2>; //Check

  using View_3D = Kokkos::View<float***>;
  View_3D M_x, M_y;

  // Formulation 1 (implicitly captures this ptr)
  // sum_views over a team (nesting level 2)
  template <class X, class Y>
  KOKKOS_INLINE_FUNCTION void sum_views(const auto& handle,
                                        size_t i) const {
    static_assert(isThreadHandle<decltype(handle)>);
    
    auto policy = range_policy_1_t(handle, 0, M_y.extent(1));
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(handle, size_t j) {
          for (size_t k = 0; k < M_y.extent(2); k++) {
            M_x(i, j, k) += M_y(i, j, k);
          }
        });

    // Thread team local ops, e.g.:
    // single(PerThread(handle),[&] () {
    //  some code here
    // });
  }

  // Formulation 2 (implicitly captures this ptr)
  // sum_views via operator over a team of threads (nesting level 2)
  KOKKOS_INLINE_FUNCTION
  void operator()(Tag_1, const auto& handle) const {
    static_assert(isThreadHandle<decltype(handle)>);
    // RangePolicy-specialized-on-ThreadHandle type (constructed from
    // RangePolicy-specialized-on-TeamHandle type ) The ctor overload would
    // return a policy semantically equivalent to TeamThreadRange

    auto policy = range_policy_1_t(handle, 0, M_x.extent(0)/*, M_x.extent(1)*/);
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(handle, size_t i, size_t j) {
          for (size_t k = 0; k < M_y.extent(2); k++) {
            M_x(i, j, k) += M_y(i, j, k);
          }
        });

    // Thread team local ops, e.g.:
    // single(PerThread(handle),[&] () {
    //  some code here
    // });
  }

  // sum_views over a league (nesting level 1)
  KOKKOS_INLINE_FUNCTION
  void operator()(const Tag_0&, const auto & handle) const {
    static_assert(isTeamHandle<decltype(handle)>);

    // RangePolicy-specialized-on-TeamHandle type
    auto policy = range_policy_0_t(handle, 0, M_x.extent(0));
    Kokkos::parallel_for("Nesting level 1", policy, *this);

    // OR another way of expressing the above is
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(const isTeamHandle& handle, size_t i) {
          sum_views(handle, i);
        });

    // Team local ops h, e.g.:
    // handle.team_barrier();
    // single(PerTeam(handle),[&] () {
    //  some code here
    // });
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const Tag_2&, size_t i, size_t& red) const {
    for (int j = 0; j < M_x.extent_int(1); j++)
      for (int k = 0; k < M_x.extent_int(2); k++) red += M_x(i, j, k);
  }

  Self_similar_range_policy() { run(); };

  void run() {
    int N         = 3;
    int num_teams = 5;
    int team_size = 5;

    M_x = View_3D("M_x", num_teams, team_size, N);
    M_y = View_3D("M_y", num_teams, team_size, N);

    Kokkos::deep_copy(M_x, 1);
    Kokkos::deep_copy(M_y, 2);

    // call sum_views(TeamHandle, nested)
    Kokkos::parallel_for("apxyFromTeam",
                         Kokkos::TeamPolicy(num_teams, Kokkos::AUTO()), *this);

    size_t result = 0;
    Kokkos::parallel_reduce("Check2", range_policy_2_t(0, M_x.extent(0)), *this,
                            result);
    ASSERT_EQ(result,
              size_t(3) * M_x.extent(0) * M_x.extent(1) * M_x.extent(2));
  }
};

template <>
struct Self_similar_range_policy<nesting_level::three> {
  struct Tag_0 {};
  struct Tag_1 {};
  struct Tag_2 {};
  struct Tag_3 {};  
  using range_policy_0_t = Kokkos::RangePolicy<Kokkos::TeamPolicy<>::member_type, Tag_0>;
  using range_policy_1_t = Kokkos::RangePolicy<Thread_handle_type, Tag_1>;
  using range_policy_1_t = Kokkos::RangePolicy<Vector_handle_type, Tag_2>;
  using range_policy_2_t = Kokkos::RangePolicy<TEST_EXECSPACE, Tag_3>; //Check

  using View_3D = Kokkos::View<float***>;
  View_3D M_x, M_y;

  // Formulation 1 (implicitly captures this ptr)
  // sum_views over a team (nesting level 3)
  template <class X, class Y>
  KOKKOS_INLINE_FUNCTION void sum_views(const auto& handle,
                                        size_t i, size_t j) const {
    static_assert(isVectorHandle<decltype(handle)>);
    auto policy = range_policy_1_t(handle, 0, M_y.extent(1));
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(handle, size_t k) {
            M_x(i, j, k) += M_y(i, j, k);
        });
    // Vector local ops (what that is)
  }

  // Formulation 1 (implicitly captures this ptr)
  // sum_views over a team (nesting level 2)
  template <class X, class Y>
  KOKKOS_INLINE_FUNCTION void sum_views(const auto& handle,
                                        size_t i, ) const {
    static_assert(isThreadHandle<decltype(handle)>);
    auto policy = range_policy_1_t(handle, 0, M_y.extent(1));
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(handle, size_t j) {
            sum_views(handle, i, j);
        });

    // Thread team local ops, e.g.:
    // single(PerThread(handle),[&] () {
    //  some code here
    // });
  }

  // Formulation 2 (implicitly captures this ptr)
  // sum_views via operator over a team of threads (nesting level 3)
  KOKKOS_INLINE_FUNCTION
  void operator()(const Tag_2&, const auto& handle) const {
    static_assert(isVectorHandle<decltype(handle)>);
    // RangePolicy-specialized-on-VectorHandle type (constructed from
    // RangePolicy-specialized-on-ThreadHandle type ) The ctor overload would
    // return a policy semantically equivalent to ThreadVectorRange

    auto policy = range_policy_2_t(handle, 0, M_x.extent(2));
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(handle, size_t i, size_t j, size_t kj) {
            M_x(i, j, k) += M_y(i, j, k);
        });
  }

  // Formulation 2 (implicitly captures this ptr)
  // sum_views via operator over a team of threads (nesting level 2)
  KOKKOS_INLINE_FUNCTION
  void operator()(const Tag_1&, const auto& handle) const {
    static_assert(isThreadHandle<decltype(handle)>);
    // RangePolicy-specialized-on-ThreadHandle type (constructed from
    // RangePolicy-specialized-on-TeamHandle type ) The ctor overload would
    // return a policy semantically equivalent to TeamThreadRange

    auto policy = range_policy_1_t(handle, 0, M_x.extent(1));
    Kokkos::parallel_for(policy, *this);

    // Thread team local ops, e.g.:
    // single(PerThread(handle),[&] () {
    //  some code here
    // });
  }

  // sum_views over a league (nesting level 1)
  KOKKOS_INLINE_FUNCTION
  void operator()(const Tag_0&, const auto& handle) const {
    static_assert(isTeamHandle<decltype(handle)>);
    // RangePolicy-specialized-on-TeamHandle type
    auto policy = range_policy_0_t(handle, 0, M_x.extent(0));
    Kokkos::parallel_for(policy, *this);

    // OR another way of expressing the above is
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(const auto& handle, size_t i) {
          sum_views(handle, i);
        });

    // Thread league local ops, e.g.:
    // single(PerTeam(team),[&] () {
    //  some code here
    // });
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const Tag_3&, size_t i, size_t& red) const {
    for (int j = 0; j < M_x.extent_int(1); j++)
      for (int k = 0; k < M_x.extent_int(2); k++) red += M_x(i, j, k);
  }

  Self_similar_range_policy() { run(); };

  void run() {
    int N          = 3;
    int num_teams  = 5;
    int team_size  = 5;

    M_x = View_3D("M_x", num_teams, team_size, N);
    M_y = View_3D("M_y", num_teams, team_size, N);

    Kokkos::deep_copy(M_x, 1);
    Kokkos::deep_copy(M_y, 2);

    // call sum_views(TeamHandle, nested)
    Kokkos::parallel_for("apxyFromTeam",
                         Kokkos::TeamPolicy(num_teams, Kokkos::AUTO()), *this);

    size_t result = 0;
    Kokkos::parallel_reduce("Check2", range_policy_2_t(0, M_x.extent(0)), *this,
                            result);
    ASSERT_EQ(result,
              size_t(3) * M_x.extent(0) * M_x.extent(1) * M_x.extent(2));
  }
};

#endif

TEST(TEST_CATEGORY, self_similar_range_policy_runtime) {
  test_self_similar_range_policy_runtime();
}

TEST(TEST_CATEGORY, self_similar_range_policy_computation) {
  // Nesting level 1
  Self_similar_range_policy<nesting_level::one> test_1;

#if 0
  // Nesting level 2
  Self_similar_range_policy<nesting_level::two> test_2;

  // Nesting level 3
  Self_similar_range_policy<nesting_level::three> test_3;
#endif
}

}  // namespace Test

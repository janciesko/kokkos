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

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <type_traits>

#include "KokkosExecutionEnvironmentNeverInitializedFixture.hpp"

namespace {

using ExecutionEnvironmentNonInitializedOrFinalized_DeathTest =
    KokkosExecutionEnvironmentNeverInitialized;

struct NonTrivial {
  KOKKOS_FUNCTION NonTrivial() {}
};
static_assert(!std::is_trivially_default_constructible_v<NonTrivial>);

TEST_F(ExecutionEnvironmentNonInitializedOrFinalized_DeathTest,
       default_constructed_views) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  auto make_views = [] {
    Kokkos::View<int> v0;
    Kokkos::View<float*> v1;
    Kokkos::View<NonTrivial**> v2;
    return std::make_tuple(v0, v1, v2);
  };
  EXPECT_EXIT(
      {
        { auto views = make_views(); }
        std::exit(EXIT_SUCCESS);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS), "");
  EXPECT_EXIT(
      {
        {
          Kokkos::initialize();
          auto views =
              make_views();  // views outlive the Kokkos execution environment
          Kokkos::finalize();
        }
        std::exit(EXIT_SUCCESS);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS), "");
  EXPECT_EXIT(
      {
        {
          Kokkos::initialize();
          Kokkos::finalize();
          auto views = make_views();
        }
        std::exit(EXIT_SUCCESS);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}

TEST_F(ExecutionEnvironmentNonInitializedOrFinalized_DeathTest, views) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  EXPECT_EXIT(
      {
        {
          Kokkos::View<int*> v;
          Kokkos::initialize();
          v = Kokkos::View<int*>("v", 10);
          v = Kokkos::View<int*>();
          Kokkos::finalize();
        }
        std::exit(EXIT_SUCCESS);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS), "");
  EXPECT_EXIT(
      {
        {
          Kokkos::initialize();
          Kokkos::View<int*> v("v", 10);
          v = {};  // assign default constructed view
          Kokkos::finalize();
        }
        std::exit(EXIT_SUCCESS);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS), "");
  EXPECT_DEATH(
      {
        Kokkos::initialize();
        Kokkos::View<int*> v("v", 0);
        Kokkos::finalize();
      },
      "Kokkos allocation \"v\" is being deallocated after Kokkos::finalize was "
      "called");

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL) || defined(KOKKOS_ENABLE_OPENACC)
  std::string matcher = "Kokkos contract violation.*";
#else
  std::string matcher =
      "Constructing View and initializing data with uninitialized execution "
      "space";
#endif
  EXPECT_DEATH({ Kokkos::View<int*> v("v", 0); }, matcher);
  EXPECT_DEATH(
      {
        Kokkos::initialize();
        Kokkos::finalize();
        Kokkos::View<int*> v("v", 0);
      },
      matcher);
}

TEST_F(ExecutionEnvironmentNonInitializedOrFinalized_DeathTest,
       c_style_memory_management) {
// FIXME_THREADS: Checking for calls to kokkos_malloc, kokkos_realloc,
// kokkos_free before initialize or after finalize is currently disabled
// for the Threads backend. Refer issue #7944.
#ifdef KOKKOS_ENABLE_THREADS
  GTEST_SKIP()
      << "skipping since initializing Threads backend calls kokkos_malloc";
#endif
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  EXPECT_DEATH(
      { [[maybe_unused]] void* ptr = Kokkos::kokkos_malloc(1); },
      "Kokkos ERROR: attempting to perform C-style memory management via "
      "kokkos_malloc\\(\\) \\*\\*before\\*\\* Kokkos::initialize\\(\\) was "
      "called");
  EXPECT_DEATH(
      {
        Kokkos::initialize();
        Kokkos::finalize();
        [[maybe_unused]] void* ptr = Kokkos::kokkos_malloc(1);
      },
      "Kokkos ERROR: attempting to perform C-style memory management via "
      "kokkos_malloc\\(\\) \\*\\*after\\*\\* Kokkos::finalize\\(\\) was "
      "called");
  EXPECT_DEATH(
      {
        Kokkos::initialize();
        void* ptr = Kokkos::kokkos_malloc(1);
        Kokkos::finalize();
        Kokkos::kokkos_free(ptr);
      },
      "Kokkos ERROR: attempting to perform C-style memory management via "
      "kokkos_free\\(\\) \\*\\*after\\*\\* Kokkos::finalize\\(\\) was called");
  EXPECT_DEATH(
      {
        Kokkos::initialize();
        void* prev = Kokkos::kokkos_malloc(1);
        Kokkos::finalize();
        [[maybe_unused]] void* next = Kokkos::kokkos_realloc(prev, 2);
      },
      "Kokkos ERROR: attempting to perform C-style memory management via "
      "kokkos_realloc\\(\\) \\*\\*after\\*\\* Kokkos::finalize\\(\\) was "
      "called");
  EXPECT_DEATH(
      {
        // Take a fake pointer
        void* ptr = reinterpret_cast<void*>(0x8BADF00D);
        Kokkos::kokkos_free(ptr);
      },
      "Kokkos ERROR: attempting to perform C-style memory management via "
      "kokkos_free\\(\\) \\*\\*before\\*\\* Kokkos::initialize\\(\\) was "
      "called");
  EXPECT_DEATH(
      {
        Kokkos::initialize();
        Kokkos::finalize();
        // Take a fake pointer
        void* ptr = reinterpret_cast<void*>(0xB105F00D);
        Kokkos::kokkos_free(ptr);
      },
      "Kokkos ERROR: attempting to perform C-style memory management via "
      "kokkos_free\\(\\) \\*\\*after\\*\\* Kokkos::finalize\\(\\) was called");
}

namespace Tested_APIs {

template <typename Type>
class EmptyReduceFunctor {
 public:
  using size_type = typename Kokkos::DefaultExecutionSpace::size_type;
  KOKKOS_INLINE_FUNCTION
  void join(Type&, const Type&) const {}
  KOKKOS_INLINE_FUNCTION
  void operator()(size_type, Type&) const {}
  KOKKOS_INLINE_FUNCTION
  void final(Type&) const {}
};

// Ctor with "String, policy and functor"
void parallel_for_1() {
  Kokkos::RangePolicy<> policy(0, 0);
  Kokkos::parallel_for("parallel_for", policy, KOKKOS_LAMBDA(int){});
}
void parallel_for_2() {
  Kokkos::RangePolicy<> policy(0, 0);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(int){});
}

// Ctor with "No Return Argument", non-fencing
void parallel_reduce_1() {
  using functor_type = EmptyReduceFunctor<float>;
  Kokkos::parallel_reduce("parallel_reduce", 0, functor_type{});
  Kokkos::fence();
}
void parallel_reduce_2() {
  using functor_type = EmptyReduceFunctor<float>;
  Kokkos::parallel_reduce(0, functor_type{});
  Kokkos::fence();
}

// Ctor with "ReturnValue is scalar or array"
void parallel_reduce_3() {
  float x;
  Kokkos::RangePolicy<> policy(0, 0);
  Kokkos::parallel_reduce("parallel_reduce", policy,
                          KOKKOS_LAMBDA(int, float&){}, x);
}
void parallel_reduce_4() {
  float x;
  Kokkos::RangePolicy<> policy(0, 0);
  Kokkos::parallel_reduce(policy, KOKKOS_LAMBDA(int, float&){}, x);
}

// Ctor with "ReturnValue as View or Reducer"
void parallel_reduce_5() {
  Kokkos::View<float> x{"x"};
  Kokkos::RangePolicy<> policy(0, 0);
  Kokkos::parallel_reduce("parallel_reduce", policy,
                          KOKKOS_LAMBDA(int, float&){}, x);
}
void parallel_reduce_6() {
  Kokkos::View<float> x{"x"};
  Kokkos::RangePolicy<> policy(0, 0);
  Kokkos::parallel_reduce(policy, KOKKOS_LAMBDA(int, float&){}, x);
}

// Ctor with "Is_execution_policy<ExecutionPolicy>, no return val", non-fencing
void parallel_scan_1() {
  Kokkos::RangePolicy<> policy(0, 0);
  Kokkos::parallel_scan("parallel_scan", policy,
                        KOKKOS_LAMBDA(int, float&, bool){});
  Kokkos::fence();
}
void parallel_scan_2() {
  Kokkos::RangePolicy<> policy(0, 0);
  Kokkos::parallel_scan(policy, KOKKOS_LAMBDA(int, float&, bool){});
  Kokkos::fence();
}

// Ctor with "Is_execution_policy<ExecutionPolicy>, return val"
void parallel_scan_3() {
  float x;
  Kokkos::RangePolicy<> policy(0, 0);
  Kokkos::parallel_scan("parallel_scan", policy,
                        KOKKOS_LAMBDA(int, float&, bool){}, x);
}
void parallel_scan_4() {
  Kokkos::RangePolicy<> policy(0, 0);
  Kokkos::parallel_scan(policy, KOKKOS_LAMBDA(int, float&, bool){});
}
}  // namespace Tested_APIs

using namespace ::testing;

TEST_F(ExecutionEnvironmentNonInitializedOrFinalized_DeathTest,
       parallel_for_1) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  std::string matcher                     = "Kokkos contract violation.*";
  EXPECT_DEATH({ Tested_APIs::parallel_for_1(); }, matcher);
  EXPECT_DEATH(
      {
        Kokkos::initialize();
        Kokkos::finalize();
        Tested_APIs::parallel_for_1();
      },
      matcher);
}

TEST_F(ExecutionEnvironmentNonInitializedOrFinalized_DeathTest,
       parallel_for_2) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  std::string matcher                     = "Kokkos contract violation.*";
  EXPECT_DEATH({ Tested_APIs::parallel_for_2(); }, matcher);
  EXPECT_DEATH(
      {
        Kokkos::initialize();
        Kokkos::finalize();
        Tested_APIs::parallel_for_2();
      },
      matcher);
}

TEST_F(ExecutionEnvironmentNonInitializedOrFinalized_DeathTest,
       parallel_reduce_1) {
  std::string matcher = "Kokkos contract violation.*";
  EXPECT_DEATH({ Tested_APIs::parallel_reduce_1(); }, matcher);
  EXPECT_DEATH(
      {
        Kokkos::initialize();
        Kokkos::finalize();
        Tested_APIs::parallel_reduce_1();
      },
      matcher);
}

TEST_F(ExecutionEnvironmentNonInitializedOrFinalized_DeathTest,
       parallel_reduce_2) {
  std::string matcher = "Kokkos contract violation.*";
  EXPECT_DEATH({ Tested_APIs::parallel_reduce_2(); }, matcher);
  EXPECT_DEATH(
      {
        Kokkos::initialize();
        Kokkos::finalize();
        Tested_APIs::parallel_reduce_2();
      },
      matcher);
}

TEST_F(ExecutionEnvironmentNonInitializedOrFinalized_DeathTest,
       parallel_reduce_3) {
  std::string matcher = "Kokkos contract violation.*";
  EXPECT_DEATH({ Tested_APIs::parallel_reduce_3(); }, matcher);
  EXPECT_DEATH(
      {
        Kokkos::initialize();
        Kokkos::finalize();
        Tested_APIs::parallel_reduce_3();
      },
      matcher);
}

TEST_F(ExecutionEnvironmentNonInitializedOrFinalized_DeathTest,
       parallel_reduce_4) {
  std::string matcher = "Kokkos contract violation.*";
  EXPECT_DEATH({ Tested_APIs::parallel_reduce_4(); }, matcher);
  EXPECT_DEATH(
      {
        Kokkos::initialize();
        Kokkos::finalize();
        Tested_APIs::parallel_reduce_4();
      },
      matcher);
}

TEST_F(ExecutionEnvironmentNonInitializedOrFinalized_DeathTest,
       parallel_reduce_5) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL) || defined(KOKKOS_ENABLE_OPENACC)
  std::string matcher = "Kokkos contract violation.*";
#else
  std::string matcher =
      "Constructing View and initializing data with uninitialized execution "
      "space";
#endif
  EXPECT_DEATH({ Tested_APIs::parallel_reduce_5(); }, matcher);
  EXPECT_DEATH(
      {
        Kokkos::initialize();
        Kokkos::finalize();
        Tested_APIs::parallel_reduce_5();
      },
      matcher);
}

TEST_F(ExecutionEnvironmentNonInitializedOrFinalized_DeathTest,
       parallel_reduce_6) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL) || defined(KOKKOS_ENABLE_OPENACC)
  std::string matcher = "Kokkos contract violation.*";
#else
  std::string matcher =
      "Constructing View and initializing data with uninitialized execution "
      "space";
#endif
  EXPECT_DEATH({ Tested_APIs::parallel_reduce_6(); }, matcher);
  EXPECT_DEATH(
      {
        Kokkos::initialize();
        Kokkos::finalize();
        Tested_APIs::parallel_reduce_6();
      },
      matcher);
}

TEST_F(ExecutionEnvironmentNonInitializedOrFinalized_DeathTest,
       parallel_scan_1) {
  std::string matcher = "Kokkos contract violation.*";
  EXPECT_DEATH({ Tested_APIs::parallel_scan_1(); }, matcher);
  EXPECT_DEATH(
      {
        Kokkos::initialize();
        Kokkos::finalize();
        Tested_APIs::parallel_scan_1();
      },
      matcher);
}

TEST_F(ExecutionEnvironmentNonInitializedOrFinalized_DeathTest,
       parallel_scan_2) {
  std::string matcher = "Kokkos contract violation.*";
  EXPECT_DEATH({ Tested_APIs::parallel_scan_2(); }, matcher);
  EXPECT_DEATH(
      {
        Kokkos::initialize();
        Kokkos::finalize();
        Tested_APIs::parallel_scan_2();
      },
      matcher);
}
TEST_F(ExecutionEnvironmentNonInitializedOrFinalized_DeathTest,
       parallel_scan_3) {
  std::string matcher = "Kokkos contract violation.*";
  EXPECT_DEATH({ Tested_APIs::parallel_scan_3(); }, matcher);
  EXPECT_DEATH(
      {
        Kokkos::initialize();
        Kokkos::finalize();
        Tested_APIs::parallel_scan_3();
      },
      matcher);
}
TEST_F(ExecutionEnvironmentNonInitializedOrFinalized_DeathTest,
       parallel_scan_4) {
  std::string matcher = "Kokkos contract violation.*";
  EXPECT_DEATH({ Tested_APIs::parallel_scan_4(); }, matcher);
  EXPECT_DEATH(
      {
        Kokkos::initialize();
        Kokkos::finalize();
        Tested_APIs::parallel_scan_4();
      },
      matcher);
}
}  // namespace

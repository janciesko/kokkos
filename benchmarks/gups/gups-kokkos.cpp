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

#include "Kokkos_Core.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <sys/time.h>

#define HLINE "-------------------------------------------------------------\n"

#if defined(KOKKOS_ENABLE_CUDA)
using GUPSHostArray   = Kokkos::View<int64_t*, Kokkos::CudaSpace>::HostMirror;
using GUPSDeviceArray = Kokkos::View<int64_t*, Kokkos::CudaSpace>;
#else
using GUPSHostArray   = Kokkos::View<int64_t*, Kokkos::HostSpace>::HostMirror;
using GUPSDeviceArray = Kokkos::View<int64_t*, Kokkos::HostSpace>;
#endif

using GUPSIndex = int;

double now() {
  struct timeval now;
  gettimeofday(&now, nullptr);
  return (double)now.tv_sec + ((double)now.tv_usec * 1.0e-6);
}

void randomize_indices(GUPSHostArray& indices, GUPSDeviceArray& dev_indices,
                       const int64_t dataCount) {
  for (GUPSIndex i = 0; i < indices.extent(0); ++i) {
    indices[i] = lrand48() % dataCount;
  }
  Kokkos::deep_copy(dev_indices, indices);
}

void run_gups(GUPSDeviceArray& indices, GUPSDeviceArray& data,
              const int64_t datum, const bool performAtomics, const int64_t num_teams,  const int64_t team_size, const int64_t vec_len) {

  auto policy =
      Kokkos::TeamPolicy<>(num_teams, team_size, vec_len);
  using team_t = Kokkos::TeamPolicy<>::member_type;

  if (performAtomics) {
//    Kokkos::parallel_for(
//        "bench-gups-atomic",  indices.extent(0),
//        KOKKOS_LAMBDA(const GUPSIndex i) {
//          Kokkos::atomic_fetch_xor(&data[indices[i]], datum);
//       });
//
  const int64_t iters_per_team = indices.extent(0) / num_teams;
  const int64_t iters_per_thread= iters_per_team / team_size;

  Kokkos::parallel_for(
    "bench-gups-non-atomic", policy,
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
      const int64_t first_i = team.league_rank() * iters_per_team;
      const int64_t last_i  = first_i + iters_per_team < indices.extent(0)
                                     ? first_i + iters_per_team
                                     : indices.extent(0);
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, first_i, last_i), [&](const int64_t j){
        const int64_t first_thread_i = team.team_rank() * iters_per_thread;
        const int64_t last_thread_i  = first_thread_i + iters_per_thread< last_i
                                     ? first_thread_i + iters_per_thread
                                     : last_i;
              Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(team,last_thread_i), [=](const int64_t i) {
                Kokkos::atomic_fetch_xor(&data[indices[first_thread_i + i]], datum);
              });
          });
      });
  } else {
//    Kokkos::parallel_for(
//      "bench-gups-non-atomic", indices.extent(0),
//      KOKKOS_LAMBDA(const GUPSIndex i) { data[indices[i]] ^= datum; });
//
//        

  const int64_t iters_per_team = indices.extent(0) / num_teams;
  const int64_t iters_per_thread= iters_per_team / team_size;

  Kokkos::parallel_for(
    "bench-gups-non-atomic", policy,
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
      const int64_t first_i = team.league_rank() * iters_per_team;
      const int64_t last_i  = first_i + iters_per_team < indices.extent(0)
                                     ? first_i + iters_per_team
                                     : indices.extent(0);
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, first_i, last_i), [&](const int64_t j){
        const int64_t first_thread_i = team.team_rank() * iters_per_thread;
        const int64_t last_thread_i  = first_thread_i + iters_per_thread< last_i
                                     ? first_thread_i + iters_per_thread
                                     : last_i;
              Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(team,last_thread_i), [=](const int64_t i) {
                Kokkos::atomic_fetch_xor(&data[indices[first_thread_i + i]], datum);
              });
          });
      });

  }
  Kokkos::fence();
}

int run_benchmark(const GUPSIndex indicesCount, const GUPSIndex dataCount,
                  const int repeats, const bool useAtomics, const int64_t num_teams, const int64_t team_size, const int64_t veclen) {


  GUPSDeviceArray dev_indices("indices", indicesCount);
  GUPSDeviceArray dev_data("data", dataCount);
  int64_t datum = -1;

  GUPSHostArray indices = Kokkos::create_mirror_view(dev_indices);
  GUPSHostArray data    = Kokkos::create_mirror_view(dev_data);

  double gupsTime = 0.0;

//  printf("Initializing Views...\n");

#if defined(KOKKOS_HAVE_OPENMP)
  Kokkos::parallel_for(
      "init-data", Kokkos::RangePolicy<Kokkos::OpenMP>(0, dataCount),
#else
  Kokkos::parallel_for(
      "init-data", Kokkos::RangePolicy<Kokkos::Serial>(0, dataCount),
#endif
      KOKKOS_LAMBDA(const int i) { data[i] = 10101010101; });

#if defined(KOKKOS_HAVE_OPENMP)
  Kokkos::parallel_for(
      "init-indices", Kokkos::RangePolicy<Kokkos::OpenMP>(0, indicesCount),
#else
  Kokkos::parallel_for(
      "init-indices", Kokkos::RangePolicy<Kokkos::Serial>(0, indicesCount),
#endif
      KOKKOS_LAMBDA(const int i) { indices[i] = 0; });

  Kokkos::deep_copy(dev_data, data);
  Kokkos::deep_copy(dev_indices, indices);
  double start;

  for (GUPSIndex k = 0; k < repeats; ++k) {
    randomize_indices(indices, dev_indices, data.extent(0));

    start = now();
    run_gups(dev_indices, dev_data, datum, useAtomics,num_teams, team_size, veclen);
    gupsTime += now() - start;
  }

  Kokkos::deep_copy(indices, dev_indices);
  Kokkos::deep_copy(data, dev_data);

//  printf(HLINE);
//    printf("- Elements:      %15" PRIu64 " (%12.4f MB)\n",
//         static_cast<uint64_t>(dataCount),
//         1.0e-6 * ((double)dataCount * (double)sizeof(int64_t)));
//  printf("- Indices:       %15" PRIu64 " (%12.4f MB)\n",
//         static_cast<uint64_t>(indicesCount),
//         1.0e-6 * ((double)indicesCount * (double)sizeof(int64_t)));
//  printf(" - Atomics:      %15s\n", (useAtomics ? "Yes" : "No"));
//  printf("Benchmark kernels will be performed for %d iterations.\n", repeats);
//
  // datacount, datasize, idxcount, idxsize, gups
  printf("Hopper,%lu,%lu,%lu,%lu,%.5f,%lu,%.5f,%s,%.5f\n",
      num_teams,
      team_size,
      veclen,
      static_cast<uint64_t>(dataCount),
      1.0e-6 * ((double)dataCount * (double)sizeof(int64_t)),
      static_cast<uint64_t>(indicesCount),
      1.0e-6 * ((double)indicesCount * (double)sizeof(int64_t)),
      (useAtomics ? "Yes" : "No"),
      (1.0e-9 * ((double)repeats) * (double)dev_indices.extent(0)) / gupsTime);

  return 0;
}

int main(int argc, char* argv[]) {
  srand48(1010101);
  
  Kokkos::initialize(argc, argv);
  
  int64_t indices = 8192;
  int64_t data    = 33554432; //256MB
  int64_t repeats = 10;
  int64_t nTeams  = 32;
  int64_t nThrPerTeam = 32;
  int64_t vlen = 1;
  bool useAtomics = false;
  
    indices = argc > 1 ? atoi(argv[1]) : indices;
  data = argc > 2 ? atoi(argv[2]):data;
  repeats = argc > 3 ? atoi (argv[3]):data;	  
  useAtomics = argc > 4 ? atoi (argv[4]):useAtomics;
  nTeams = argc > 5 ? atoi (argv[5]):nTeams;
  nThrPerTeam= argc > 6 ? atoi (argv[6]): nThrPerTeam;
  vlen = argc > 7 ? atoi (argv[7]): vlen;

  const int rc = run_benchmark(indices, data, repeats, useAtomics,nTeams,nThrPerTeam, vlen);

  Kokkos::finalize();

  return rc;
}


#!/bin/bash

KOKKOS_INTALL="@CMAKE_BINARY_DIR@"
SPACK_TEST_SOURCE="@CMAKE_CURRENT_BINARY_DIR@/source"
SPACK_TEST_BUILD="@CMAKE_CURRENT_BINARY_DIR@/build"

cd "${SMOKE_TEST_BUILD}"
rm -rf CMake*

cmake "${SPACK_TEST_SOURCE}" -D Kokkos_ROOT:PATH="@CMAKE_BINARY_DIR@/lib/cmake/Kokkos"


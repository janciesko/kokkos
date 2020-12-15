file(COPY ${PACKAGE_SOURCE_DIR}/scripts/spack_test/CMakeLists.txt.in DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/source)
file(RENAME ${CMAKE_CURRENT_BINARY_DIR}/source/CMakeLists.txt.in ${CMAKE_CURRENT_BINARY_DIR}/source/CMakeLists.txt)

set(TEST_FILE "./example/tutorial/01_hello_world/hello_world.cpp")

# Second copy the source files needed to the build area
file(COPY ${PACKAGE_SOURCE_DIR}/${TEST_FILE}$ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/source)

# Third write a configure file that can be invoked to test the library installation
configure_file(${PACKAGE_SOURCE_DIR}/spack_test/run_test.sh.in ${CMAKE_CURRENT_BINARY_DIR}/run_test.sh @ONLY)

add_test(NAME SPACK_TEST COMMAND /bin/bash ${CMAKE_CURRENT_BINARY_DIR}/run_test.sh)

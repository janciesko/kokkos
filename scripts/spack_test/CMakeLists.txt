#Copy CMakeLists.txt.in to Spack test directory
set(RELATIVE_PATH "scripts/spack_test/")
file(COPY ${SPACK_PACKAGE_TEST_ROOT_DIR}/${RELATIVE_PATH}/CMakeLists.txt.in DESTINATION ${SPACK_PACKAGE_TEST_ROOT_DIR}/source)
file(RENAME ${SPACK_PACKAGE_TEST_ROOT_DIR}/source/CMakeLists.txt.in ${SPACK_PACKAGE_TEST_ROOT_DIR}/source/CMakeLists.txt)

#parse test_list.def and copy those tests that need it

#Copy test source to Spack test directory
set(TEST_FILE ${SPACK_PACKAGE_SOURCE_DIR}/example/tutorial/01_hello_world/hello_world.cpp)
file(COPY ${TEST_FILE}$ DESTINATION ${SPACK_PACKAGE_SOURCE_DIR}/sources)

#Configure test generator script with correct paths from Spack
configure_file(${SPACK_PACKAGE_SOURCE_DIR}${RELATIVE_PATH}/spack_test/run_test.sh.in ${SPACK_PACKAGE_SOURCE_DIR}/sources/run_test.sh @ONLY)

#Run test generator script
add_test(NAME SPACK_TEST COMMAND /bin/bash ${CURRENT_TEST_DIR}/sources/run_test.sh)

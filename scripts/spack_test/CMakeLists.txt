project(SpackTestGen)
#Copy CMakeLists.txt.in to Spack test directory
set(RELATIVE_PATH "scripts/spack_test")
set(ABOLUTE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${RELATIVE_PATH}")
message(${ABOLUTE_PATH})
message("=============")
file(COPY "${ABSOLUTE_PATH}/CMakeLists.txt.in" DESTINATION "${ABSOLUTE_PATH}/out/")
file(RENAME "${ABSOLUTE_PATH}/out/CMakeLists.txt.in" "${ABSOLUTE_PATH}/out/CMakeLists.txt")

#parse test_list.def and copy those tests that need it

#Copy test source to Spack test directory
set(TEST_FILE ${SPACK_PACKAGE_SOURCE_DIR}/example/tutorial/01_hello_world/hello_world.cpp)
file(COPY ${TEST_FILE} DESTINATION ${ABSOLUTE_PATH}/out)

#Configure test generator script with correct paths from Spack
configure_file(${ABSOLUTE_PATH}/run_test.sh.in ${ABSOLUTE_PATH}/out/run_test.sh @ONLY)

#Run test generator script
#add_test(NAME SPACK_TEST COMMAND /bin/bash ${CURRENT_TEST_DIR}/sources/run_test.sh)

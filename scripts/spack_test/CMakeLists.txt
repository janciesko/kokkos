project(SpackTestGen)
#Copy CMakeLists.txt.in to Spack test directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt.in DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/out/)
file(RENAME ${CMAKE_CURRENT_SOURCE_DIR}/out/CMakeLists.txt.in ${CMAKE_CURRENT_SOURCE_DIR}/out/CMakeLists.txt)

#parse test_list.def and copy those tests that need it

#Copy test source to Spack test directory
set(TEST_FILE ${SPACK_PACKAGE_SOURCE_DIR}/example/tutorial/01_hello_world/hello_world.cpp)
file(COPY ${TEST_FILE} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/out)

#Configure test generator script with correct paths from Spack
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_test.sh.in ${CMAKE_CURRENT_SOURCE_DIR}/out/run_test.sh @ONLY)

#Run test generator script
#add_test(NAME SPACK_TEST COMMAND /bin/bash ${CURRENT_TEST_DIR}/sources/run_test.sh)

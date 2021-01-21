cmake_minimum_required(VERSION 3.12)
project(SpackTestGen)
#Copy CMakeLists.txt.in to Spack test directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt.in DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/out/)
file(RENAME ${CMAKE_CURRENT_SOURCE_DIR}/out/CMakeLists.txt.in ${CMAKE_CURRENT_SOURCE_DIR}/out/CMakeLists.txt)

#parse test_list.def and copy those tests that need it
set(TEST_LIST_DEF "${CMAKE_CURRENT_SOURCE_DIR}/test_list.def")
FILE(READ "${TEST_LIST_DEF}" TEST_FILES)
STRING(REGEX REPLACE "\n" ";" TEST_FILES "${TEST_FILES}")

#Copy test source to Spack test directory
foreach (TEST_FILE in ${TEST_FILES})
	set(TEST_FILE_SRC ${SPACK_PACKAGE_SOURCE_DIR}/${TEST_FILE})
	file(COPY ${TEST_FILE_SRC} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/out)
endforeach()

#Configure test generator script with correct paths from Spack
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_test.sh.in ${CMAKE_CURRENT_SOURCE_DIR}/out/run_test.sh @ONLY)

#Run test generator script
#add_test(NAME SPACK_TEST COMMAND /bin/bash ${CURRENT_TEST_DIR}/sources/run_test.sh)

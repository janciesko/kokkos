cmake_minimum_required(VERSION 3.14)
project(SpackTestGen)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt.in DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/out/)
file(RENAME ${CMAKE_CURRENT_SOURCE_DIR}/out/CMakeLists.txt.in ${CMAKE_CURRENT_SOURCE_DIR}/out/CMakeLists.txt)
set(TEST_LIST_DEF ${CMAKE_CURRENT_SOURCE_DIR}/test_list.def)
FILE(READ ${TEST_LIST_DEF} TEST_FILES)

#Copy test source to Spack test directory
foreach (TEST_FILE IN LISTS ${TEST_FILES})
#  message("XXXXXX:" ${TEST_FILE})
  file(COPY ${SPACK_PACKAGE_SOURCE_DIR}/${TEST_FILE} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/out)
endforeach()

#Configure test generator script with correct paths from Spack
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_test.sh.in ${CMAKE_CURRENT_SOURCE_DIR}/out/run_test.sh @ONLY)

#Run test generator script
#add_test(NAME SPACK_TEST COMMAND /bin/bash ${CURRENT_TEST_DIR}/sources/run_test.sh)

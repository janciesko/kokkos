cmake_minimum_required(VERSION 3.14)
project(SpackTestGen)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt.in DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/out/)
file(RENAME ${CMAKE_CURRENT_SOURCE_DIR}/out/CMakeLists.txt.in ${CMAKE_CURRENT_SOURCE_DIR}/out/CMakeLists.txt)
set(TEST_LIST_DEF ${CMAKE_CURRENT_SOURCE_DIR}/test_list.def)
file(STRINGS ${TEST_LIST_DEF} TEST_FILES)

#Copy test source to Spack test directory
foreach (TEST_FILE ${TEST_FILES})
  set(TEST_FILE_LOCATION ${SPACK_PACKAGE_SOURCE_DIR}/${TEST_FILE})
  file(COPY ${SPACK_PACKAGE_SOURCE_DIR}/${TEST_FILE} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/out)
endforeach()

#Configure test generator script with correct paths from Spack
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/make.sh.in ${CMAKE_CURRENT_SOURCE_DIR}/out/make.sh @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run.sh.in ${CMAKE_CURRENT_SOURCE_DIR}/out/run.sh @ONLY)

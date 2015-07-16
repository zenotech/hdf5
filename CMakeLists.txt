cmake_minimum_required(VERSION 2.8)

project(hdf5)

find_package(HDF5 REQUIRED COMPONENTS C HL)
find_package(Boost REQUIRED COMPONENTS system)
find_package(MPI REQUIRED)

set (CMAKE_CXX_FLAGS "-std=c++11")

include_directories( ${MPI_CXX_INCLUDE_PATH} 
    ${MPI_C_INCLUDE_PATH}
    ${HDF5_INCLUDE_DIRS}
    ${boost_INCLUDE_DIRS}
    ${boost_LIBRARY_DIRS})

include_directories(. hdf5)

set(HDF5_HEADERS 
    hdfattribute.hpp
    hdfdataset.hpp
    hdffile.hpp
    hdfgroup.hpp
    slab.hpp
    hdf5/traits.hpp)

add_executable(hdf5_tests test-hdf.cpp ${HDF5_HEADERS})
target_link_libraries(hdf5_tests ${HDF5_LIBRARIES} ${Boost_LIBRARIES})


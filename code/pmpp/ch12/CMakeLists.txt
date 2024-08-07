cmake_minimum_required(VERSION 3.0)

project(merge_sort)

include_directories(include)
include_directories(../include)

add_executable(test_merge_sort test_merge_sort.cu merge_sort.cu merge_circular_buffer_sort.cu)
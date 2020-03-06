#
# Copyright INAOS GmbH, Thalwil, 2019.
# Copyright Francesc Alted, 2019.
#
# All rights reserved
#
# This software is the confidential and proprietary information of INAOS GmbH
# and Francesc Alted ("Confidential Information"). You shall not disclose such Confidential
# Information and shall use it only in accordance with the terms of the license agreement.
#
cmake_minimum_required (VERSION 3.12)
project(minjugg)

include_directories("${CMAKE_CURRENT_LIST_DIR}/include")

set(SRC ${CMAKE_CURRENT_LIST_DIR}/src)
set(CMAKE_CXX_STANDARD 14)
add_library(minjugg ${SRC}/tinyexpr.c ${SRC}/minjugg.c)
add_library(minjuggutil ${SRC}/minjuggutil.cpp)

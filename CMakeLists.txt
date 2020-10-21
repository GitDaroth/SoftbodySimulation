cmake_minimum_required(VERSION 3.17)

project(SoftbodySimulation 
	VERSION 0.0.1
	LANGUAGES CXX CUDA)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt5 COMPONENTS Core Widgets Gui REQUIRED)

add_subdirectory(SoftbodyPhysics)
add_subdirectory(SoftbodySimulation)
add_subdirectory(MeshSampler)
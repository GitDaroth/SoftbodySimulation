set(miscFiles
	include/ParticleDragger.h
	src/ParticleDragger.cpp
	src/main.cpp)

set(cameraFiles
	include/Camera/Camera.h
	src/Camera/Camera.cpp
	include/Camera/CameraController.h
	src/Camera/CameraController.cpp)

set(scenarioFiles
	include/Scenario/Scenario.h
	src/Scenario/Scenario.cpp
	include/Scenario/ObstacleCourseScenario.h
	src/Scenario/ObstacleCourseScenario.cpp
	include/Scenario/BlockScenario.h
	src/Scenario/BlockScenario.cpp
	include/Scenario/MovingWallScenario.h
	src/Scenario/MovingWallScenario.cpp
	include/Scenario/ShapeTypeComparisonScenario.h
	src/Scenario/ShapeTypeComparisonScenario.cpp
	include/Scenario/SpinningBoxesScenario.h
	src/Scenario/SpinningBoxesScenario.cpp
	include/Scenario/StairsScenario.h
	src/Scenario/StairsScenario.cpp
	include/Scenario/WashingMachineScenario.h
	src/Scenario/WashingMachineScenario.cpp)

set(widgetFiles
	include/Widget/MainWindow.h
	include/Widget/MainWindow.ui
	src/Widget/MainWindow.cpp
	include/Widget/ControlPanel.h
	include/Widget/ControlPanel.ui
	src/Widget/ControlPanel.cpp
	include/Widget/OpenGLWidget.h
	src/Widget/OpenGLWidget.cpp)

set(shaderFiles
	assets/shaders/shapeMesh.frag
	assets/shaders/shapeMesh.vert
	assets/shaders/skybox.frag
	assets/shaders/skybox.vert
	assets/shaders/sphereImpostor.frag
	assets/shaders/sphereImpostor.geom
	assets/shaders/sphereImpostor.vert
	assets/shaders/texture.frag
	assets/shaders/texture.vert)

source_group("" FILES ${miscFiles})
source_group("\\Camera" FILES ${cameraFiles})
source_group("\\Scenario" FILES ${scenarioFiles})
source_group("\\Widget" FILES ${widgetFiles})
source_group("\\Shader" FILES ${shaderFiles})

add_executable(SoftbodySimulation 
	${miscFiles}
	${cameraFiles}
	${scenarioFiles} 
	${widgetFiles}
	${shaderFiles})

set_property(TARGET SoftbodySimulation PROPERTY CUDA_ARCHITECTURES 35 50 72)

target_link_libraries(SoftbodySimulation PRIVATE SoftbodyPhysics Qt5::Widgets)
target_include_directories(SoftbodySimulation PUBLIC "include")


add_custom_target(copyAssets)
set_target_properties(copyAssets PROPERTIES FOLDER PostBuild)

FILE(TO_NATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/assets" source)
FILE(TO_NATIVE_PATH "${CMAKE_CURRENT_BINARY_DIR}/assets" destination)

add_custom_command(
	TARGET copyAssets POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_directory ${source} ${destination}
	DEPENDS ${destination}
	COMMENT "copy assets folder from ${source} to ${destination}")

add_dependencies(SoftbodySimulation copyAssets)
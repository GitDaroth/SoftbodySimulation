set(miscFiles
	include/PBDSolver.h
	src/PBDSolver.cpp
	include/PBDParticle.h
	src/PBDParticle.cpp
	include/PBDParameterData.h)

set(cudaFiles
	include/CUDA/ConstructGrid.cuh
	src/CUDA/ConstructGrid.cu
	include/CUDA/CorrectParticles.cuh
	src/CUDA/CorrectParticles.cu
	include/CUDA/Integrate.cuh
	src/CUDA/Integrate.cu
	include/CUDA/RaytestParticles.cuh
	src/CUDA/RaytestParticles.cu
	include/CUDA/SkinMesh.cuh
	src/CUDA/SkinMesh.cu
	include/CUDA/SolveConstraints.cuh
	src/CUDA/SolveConstraints.cu
	include/CUDA/UpdateShapes.cuh
	src/CUDA/UpdateShapes.cu
	include/CUDA/ParticleData.h
	include/CUDA/ShapeData.h
	include/CUDA/ColliderData.h
	include/CUDA/VertexData.h)

set(constraintFiles
	include/Constraint/Constraint.h
	src/Constraint/Constraint.cpp
	include/Constraint/BoxConstraint.h
	src/Constraint/BoxConstraint.cpp
	include/Constraint/PlaneConstraint.h
	src/Constraint/PlaneConstraint.cpp
	include/Constraint/ShapeConstraint.h
	src/Constraint/ShapeConstraint.cpp
	include/Constraint/RigidShapeConstraint.h
	src/Constraint/RigidShapeConstraint.cpp
	include/Constraint/SoftShapeLinearConstraint.h
	src/Constraint/SoftShapeLinearConstraint.cpp
	include/Constraint/SoftShapeQuadraticConstraint.h
	src/Constraint/SoftShapeQuadraticConstraint.cpp)

source_group("" FILES ${miscFiles})
source_group("\\CUDA" FILES ${cudaFiles})
source_group("\\Constraint" FILES ${constraintFiles})

add_library(SoftbodyPhysics STATIC 
	${miscFiles}
	${cudaFiles}
	${constraintFiles})

set_property(TARGET SoftbodyPhysics PROPERTY CUDA_ARCHITECTURES 35 50 72)

target_link_libraries(SoftbodyPhysics PUBLIC Qt5::Core Qt5::Gui)
target_include_directories(SoftbodyPhysics PUBLIC "include" "3rdParty" "3rdParty/CUDA" "3rdParty/tinyobjectloader")
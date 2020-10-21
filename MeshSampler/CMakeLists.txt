add_executable(MeshSampler main.cpp tiny_obj_loader.h)
target_link_libraries(MeshSampler PUBLIC Qt5::Core Qt5::Gui)
set_property(TARGET MeshSampler PROPERTY CUDA_ARCHITECTURES 35 50 72)
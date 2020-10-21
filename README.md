# SoftbodySimulation
Interactive real-time soft-body simulation parallelized on a GPU with CUDA.

![obstacle_course](/showcase/obstacle_course.gif)
![skinning_vs_particles](/showcase/skinning_vs_particles.gif)

**Features:**
- PBD (Position Based Dynamics) method
- Soft-bodies from mesh and point data
- Box constraints (boundary or obstacle)
- Plane constraints
- Shape matching constraints (rigid, soft linear, soft quadratic)
- Cluster shape matching
- Mesh skinning according to particle positions
- Particle collisions
- Friction
- Different scenarios
- Particle dragging

## Build Instructions
For Visual Studio 2019:
```
git clone https://github.com/GitDaroth/SoftbodySimulation
cd SoftbodySimulation
cmake_generate_VS2019.bat
```
Open the generated Visual Studio solution in the "build" folder and build the "SoftbodySimulation" target.

Don't forget to copy the needed dll files from your Qt5 installation next to your executable:
```
QT5_INSTALLATION_PATH/bin/Qt5Core.dll
QT5_INSTALLATION_PATH/bin/Qt5Gui.dll
QT5_INSTALLATION_PATH/bin/Qt5Widgets.dll
QT5_INSTALLATION_PATH/plugins/platforms/qwindows.dll
QT5_INSTALLATION_PATH/plugins/styles/qwindowsvistastyle.dll
QT5_INSTALLATION_PATH/plugins/imageformats/qjpeg.dll
```

## Dependencies
**SoftbodyPhysics:**
- [Qt5 (Modules: Core, Gui)](https://www.qt.io)
- [CUDA](https://developer.nvidia.com/cuda-downloads)
- [Eigen](https://github.com/libigl/eigen)
- [unsupported/Eigen](https://github.com/libigl/eigen/tree/master/unsupported)
- [tinyobjectloader](https://github.com/tinyobjloader/tinyobjloader)

**SoftbodySimulation:**
- SoftbodyPhysics
- [Qt5 (Modules: Widgets)](https://www.qt.io)
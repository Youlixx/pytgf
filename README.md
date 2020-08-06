# pytgf - Simple python tile based game framework
A simple implementation of a tile engine, including a physical and graphical engine based on OpenGL.
The python package embbed the following functions:
- a complete event pipeline, for pure event-driven programming.
- multithreaded collision detection and exact resolution.
- rendering of the world and GUI with or without a window.
- OpenGL and GLSL code free. If you don't want to do advanced rendering things, you don't have to know how OpenGL works.

## Requirement
Your GPU should support OpenGL 3.3.0 and GLSL 3.30 (unless you create custom shaders compatible with previous versions).

## Installation
This package can be installed using `pip install pytgf`.

## Dependencies
The library requires the following packages (the latest stable versions should be used):
 - moderngl
 - pyglet
 - numpy

### Optional packages
Other usefull packages:
 - imageio (library for reading and writing a wide range of image, video, scientific, and volumetric data formats)

## License
This library is available under the [MIT license](LICENSE.md).
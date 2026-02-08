# pmoo - parallel multi-objective optimization

## Project requirements

The project was verified to work on Ubuntu 22, 24 and Windows 11 (with windows specific instructions)

- [CMake](https://cmake.org/)

- [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit)

- CUDA capable GPU

## Recommended

- [VS Code](https://code.visualstudio.com/)

- [CMake Tools extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools)

## Cmake profiles

The available CMake profiles are appropriately named for the target operating systems (Windows or Linux), with both Debug and Release configurations provided.

You might be required to add or modify `"CMAKE_CUDA_ARCHITECTURES"` in `"cacheVariables"` to align with your GPU environment. For example modified Windows profile specifying 86 value for RTX 3050 Ti:
```
{
    "name": "winrelease",
    "displayName": "WindowsRelease",
    "generator": "Ninja",
    "binaryDir": "${sourceDir}/build",
    "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
        "CMAKE_CXX_FLAGS_INIT": "-O3 /openmp:llvm",
        "CMAKE_CUDA_SEPARABLE_COMPILATION": "ON",
        "CMAKE_CUDA_FLAGS_INIT": "-Xcompiler=/openmp:llvm",
        "CMAKE_CUDA_ARCHITECTURES": "86"
    }
}
```

CMake presets was verified for version 8 but should work on older versions, the same should be applicable for CMake version defined in `CMakeLists.txt`

## How to build

### CLI

1. To build the solution using CLI you should check the available profile names, have the CLI pointed at the root project directory and specify the profile name in the `cmake --preset [name]` command, eg. `cmake --preset linuxrelease`
2. Run `cmake --build [fir]` where the default value is going to be `cmake --build build`

### VS Code
 1. Using CMake Tools extension for VS Code select the configuration profile and load it.
 2. Run the default build

## Windows specific instructions

- run the VS Code or Cmake through the **`x64 Native Tools Command Prompt for VS 2022`** or different versions provided by the Visual Studio

> **Note:** This could be skipped if a proper environment is set up (We were not able to create it ourselves)

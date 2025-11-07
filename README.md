# Mechanics Benchmarking

A C++ application for benchmarking and validating mechanics solver algorithms with different optimization strategies.

## Overview

This project implements and benchmarks various mechanics solver algorithms for computational mechanics problems. It supports multiple solver implementations including reference, grid-based, and transposed solvers, with performance optimizations using SIMD operations via Google Highway.

## Prerequisites

- **CMake** 3.10 or higher
- **C++ Compiler** with C++20 support (GCC, Clang, or MSVC)
- **Ninja** build system
- **vcpkg** package manager
- **OpenMP** for parallel computation

## Dependencies

The project uses the following libraries (managed by vcpkg):
- `argparse` - Command-line argument parsing
- `nlohmann-json` - JSON file handling
- `highway` - SIMD operations for performance optimization

## Building the Project

### 1. Setup vcpkg

If you haven't already, bootstrap vcpkg:

```bash
cd vcpkg
./bootstrap-vcpkg.sh  # On Linux/macOS
```

Set the `VCPKG_ROOT` environment variable:

```bash
export VCPKG_ROOT="$PWD/vcpkg"
```

### 2. Configure the Project

The project uses CMake presets. Choose either Debug or Release configuration:

**Debug Build** (with sanitizers and debug symbols):
```bash
cmake --preset Debug
```

**Release Build** (optimized for performance):
```bash
cmake --preset Release
```

This will:
- Install dependencies via vcpkg
- Generate build files in `build/Debug/` or `build/Release/`
- Enable compile commands export for IDE integration

### 3. Build

After configuration, build the project:

```bash
cmake --build build/Debug
```

or for Release:

```bash
cmake --build build/Release
```

The executable `mechanize` will be created in the respective build directory.

## Usage

The `mechanize` program requires three main arguments:

```bash
./build/Release/mechanize --alg <algorithm> --params <params.json> --problem <problem.json> [OPTIONS]
```

### Required Arguments

- `--alg` - Algorithm to run (e.g., `reference`, `grid`, `transposed`)
- `--params` - Path to JSON parameters file
- `--problem` - Path to JSON problem definition file

### Operations (choose one)

- `--validate` - Validate algorithm correctness against reference implementation
- `--benchmark` - Benchmark algorithm performance
- `--run [FILE]` - Run algorithm and optionally output results to FILE

### Optional Arguments

- `--verbose` - Enable verbose output
- `--double` - Use double precision instead of single precision

### Example Commands

**Validate an algorithm:**
```bash
./build/Release/mechanize --alg grid --params examples/params_validation.json --problem examples/problem_validation.json --validate
```

**Benchmark performance:**
```bash
./build/Release/mechanize --alg grid --params examples/params_benchmark.json --problem examples/problem_2d_medium.json --benchmark
```

**Run with output:**
```bash
./build/Release/mechanize --alg transposed --params examples/params_base.json --problem examples/problem_3d_medium.json --run output.json --verbose
```

## Project Structure

```
.
├── CMakeLists.txt              # Build configuration
├── CMakePresets.json           # CMake presets for Debug/Release
├── vcpkg.json                  # Dependency manifest
├── src/
│   ├── main.cpp                # Application entry point
│   ├── algorithms.cpp/h        # Algorithm dispatcher
│   ├── grid.cpp/h              # Grid data structure
│   ├── problem.h               # Problem definition
│   ├── mechanics_solver.h      # Solver interface
│   ├── agent_distributor.h     # Agent distribution logic
│   └── solvers/                # Solver implementations
│       ├── base_solver.cpp/h
│       ├── reference_solver.cpp/h
│       ├── grid_solver.cpp/h
│       └── transposed_solver.cpp/h
└── examples/                   # Example parameter and problem files
    ├── params_*.json           # Parameter configurations
    └── problem_*.json          # Problem definitions
```

## Build Configurations

### Debug Configuration
- Compiler warnings: `-Wall -Wextra -Wpedantic`
- Sanitizers enabled:
  - AddressSanitizer
  - LeakSanitizer
  - UndefinedBehaviorSanitizer
  - Pointer comparison/subtract sanitizers
- Native architecture optimizations: `-march=native`

### Release Configuration
- Compiler warnings: `-Wall -Wextra -Wpedantic`
- Full optimizations enabled
- Debug symbols included (`-g`)
- Native architecture optimizations: `-march=native`

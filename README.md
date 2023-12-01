# Parallel BFS and K-Means Clustering with OpenMP and MPI

## Overview

This project demonstrates parallel implementations of Breadth-First Search (BFS) and K-Means Clustering algorithms in C using OpenMP and MPI. The combination of OpenMP and MPI allows for efficient parallelization on shared-memory (multi-core) and distributed-memory (cluster) systems.

## Table of Contents

- [Parallel BFS and K-Means Clustering with OpenMP and MPI](#parallel-bfs-and-k-means-clustering-with-openmp-and-mpi)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Building](#building)
  - [Usage](#usage)
  - [Breadth-First Search](#breadth-first-search)
  - [K-Means Clustering](#k-means-clustering)
  - [License](#license)

## Prerequisites

Ensure the following dependencies are installed:

- OpenMP: Compiler with OpenMP support (e.g., GCC)
- MPI: MPI library (e.g., Open MPI)

## Building

Compile the project using the build task:

```
terminal -> Run Build Task
```

This will generate executable files.

NOTE: Make sure you have vscode with openmp and mpi support

## Usage

Run the programs using MPI commands. For example:

```bash
mpiexec -np 8 ./bfs_mpi.exe
mpiexec -np 4 ./kmeans_mpi.exe
```

Adjust the number of processes (`-np`) based on your computing environment.

## Breadth-First Search

The BFS algorithm explores the given graph in parallel using MPI. The input graph should be in a specific format (e.g., adjacency matrix) as specified.


## K-Means Clustering

The K-Means algorithm performs parallel clustering of data using MPI. The input data should be in a suitable format (e.g., space-separated values) as specified in `input_data.txt`.


## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute the code according to the terms of the license.


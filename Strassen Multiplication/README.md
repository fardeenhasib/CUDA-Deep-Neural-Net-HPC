# Strassen Matrix Multiplication on CUDA

This project implements Strassen matrix multiplication on CUDA, a parallel computing platform that allows using GPU for general purpose programming. The project compares the performance of Strassen algorithm with the naïve matrix multiplication algorithm on different matrix sizes and terminal matrix sizes.

## How to run the code

- Load the intelcuda/2020a module. For some unknown reason, using the latest CUDA version results in weird outputs. Using an older version of CUDA fixes this issue.
    - `module load intelcuda/2020a
- Compile the Strassen_cuda.cu file using the following command:
    - `nvcc -o strassen_cuda.exe strassen_cuda.cu`
- Run the executable using the following command:
    - `./strassen_cuda.exe k k_bar`
    - Where `k > k_bar > 1`

## Results and analysis

The project plots the execution time, speed up, and efficiency of Strassen algorithm versus matrix size for a fixed terminal matrix size (`k_bar = 5`). The plots show that the parallel performance increases as the matrix size increases, because Strassen algorithm has more advantage over naïve algorithm for larger matrices, and also because using more threads reduces the memory read/write and serial operations.

The project also shows the effect of varying `k_bar`, which is the size of the submatrices that are multiplied using CUDA device. The results show that choosing smaller `k_bar` values results in faster execution time, because choosing larger `k_bar` values means performing matrix multiplication on smaller matrices one at a time using CUDA device, which hinders utilizing the threads properly. Also, combining the results of small chunks requires more memory operations, which reduces parallel performance.

The project uses cached (shared memory) matrix multiplication on CUDA device instead of the naïve (global memory) version, which significantly improves the runtime of Strassen algorithm.

## References

The project uses the following sources as references:

- [Programming with CUDA: Matrix Multiplication](^1^)
- [Code from the "CUDA Crash Course" YouTube series by CoffeeBeforeArch](^2^)¹²[12]
- [Parallelizing Strassen’s matrix multiplication using OpenMP, MPI and CUDA](^3^)¹³[13]

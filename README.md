Parallelized Mini-Batch Gradient Descent for Linear Regression (OpenMP)

📌 Overview

This project implements Linear Regression using Mini-Batch Gradient Descent, parallelized with OpenMP to accelerate training on large datasets. The gradient computation for each mini-batch is executed concurrently using multi-threading, significantly reducing overall training time on multi-core CPUs.
The project demonstrates how classical machine learning algorithms can be efficiently parallelized using shared-memory parallel programming.

🚀 Features

Linear Regression implemented from scratch in C/C++
Mini-Batch Gradient Descent optimization
Parallel gradient computation using OpenMP
Efficient utilization of multi-core CPUs
Thread-safe gradient aggregation
Scalable and performance-oriented design

🧠 Why OpenMP?

OpenMP provides a simple and efficient way to parallelize loops and shared-memory computations. In this project:
Each mini-batch is processed in parallel
Gradient calculations are distributed across threads
Synchronization ensures correct parameter updates
This results in faster convergence compared to sequential mini-batch gradient descent.

🏗️ Parallel Architecture

Dataset is divided into mini-batches
Each mini-batch is split across multiple threads
Gradients are computed in parallel using #pragma omp parallel for
Partial gradients are reduced safely
Model parameters are updated iteratively

🛠️ Technologies Used

Language: C / C++
Parallelization: OpenMP
Compiler: GCC / Clang (with OpenMP support)

📊 Performance Highlights

Significant speedup over sequential implementation
Improved CPU utilization
Scales with number of cores

📚 Learning Outcomes

Practical understanding of Mini-Batch Gradient Descent
Hands-on experience with OpenMP parallelism
Reduction operations and thread synchronization
Performance optimization for ML algorithms

🔮 Future Enhancements

Dynamic scheduling strategies
Cache-aware optimizations
Support for multi-variable regression
Comparison with MPI and CUDA implementations
Performance benchmarking and visualization

✨ Author

Sreyas Krishnan

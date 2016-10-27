Histogram project for Programming Massively Parrallel Hardware.

histMain         -- Controls testing and benchmarking

setup.cu.h       -- Contains histogram parameters, that are set before usage.

histDataGen.cu.h -- Generates arbitrarily large data arrays, of type T.

Host.cu.h        -- Kernel wrappers

Kernels.cu.h     -- Cuda kernels for computing large scale histograms

radix.cu         -- A test program where we are using Radix sort in CUB.
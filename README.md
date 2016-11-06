Histogram project for Programming Massively Parrallel Hardware.

setup.cu.h        -- Contains histogram parameters, that are set before usage.

Host.cu.h         -- Kernel wrappers

Kernels.cu.h      -- Cuda kernels for computing large scale histograms

NaiveHistTest.cu  -- Testing and benchmarking programs for the histogram kernels.

StreamHistTest.cu -- Testing and benchmarking programs for async streaming.

sequential/...    -- A CPU implementation using OpenMP and some array helper
                     functions.

To test the histogram do

```
make histtest
./histtest <data_size> <bins>
```

To test the streaming do
```
make stream
./stream <data_size> <bins>
```

Note that an upper bound for the data size (at the moment) is the size of a 32 bit unsigned integer.

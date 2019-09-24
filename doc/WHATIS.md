# What is IronArray?

IronArray is both a data container and a computational engine optimized for numerical data.

The data container can be compressed and decompressed transparently.  It is based on the Blosc2 format and leverages the Caterva library for fast manipulation of multidimensional data.

The computational engine is based on LLVM and it is carefully tuned to the get most performance out of IronArray compressed containers.

The ultimate goal of IronArray is to be able to deal with compressed datasets at the speed of traditional, uncompressed datasets.

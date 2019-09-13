# Error propagation in matrix multiplication

An error propagation in matrix-matrix multiplication is found. It ocurrs when two linspaces of 100x100 (in general, squared matrix) elements between -1 and 1 (in general, from -x to x) are multiplied. In the iron array multiplication, blocks of 10x10 elements are used.
To reproduce it, the three algorithms created to perform the multiplications are in the next section.

## Matrix-matrix multiplication Algorithms

### Basic algorithm

```C
int mult_c(const double *a, const double *b, double *c, const int I, const int J, const int K) {

    for (int i = 0; i < I; ++i) {
        for (int j = 0; j < J; ++j) {
            double sum = 0;
            for (int k = 0; k < K; ++k) {
                sum = sum + a[i * K + k] * b[k * J + j];
            }
            c[i * J + j] = sum;
        }
    }

    return 0;
}
```

### MKL algorithm

```C
int mult_mkl(const double *a, const double *b, double *c, const int I, const int J, const int K) {

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, I, J, K,
        1.0, a, (int) K, b, (int) J, 0.0, c, (int) J);

    return 0;
}
```

### Iarray algorithm

```C
int mult_iarray(iarray_context_t *ctx, iarray_container_t *a, int64_t *bshape_a,
    iarray_container_t *b, int64_t *bshape_b, iarray_container_t *c) {

    INA_SUCCEED(iarray_linalg_matmul(ctx, a, b, c, bshape_a, bshape_b, IARRAY_OPERATOR_GENERAL));

    return 0;
}
```

## Results

The results obtained when comparing the multiplications with each other are:

```C
Error percentage (C - MKL): 0.1501
Error percentage (C - iarray): 0.3882
Error percentage (MKL - iarray): 0.3805
```

We believe that the problem is in the order of operations. Due to the blocking nature of our matmul implementation, it is not possible to change this order. Therefore, we must be prepared to see strong differences in some data distributions, such as the one shown here. For now, this has not been seen in other data distributions.

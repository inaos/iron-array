# Linear algebra in ironArray

The intention of this document is to collect some ideas on how to implement
linear algebra algorithms using chunked (tiled) data (See https://docs.dask.org/en/latest/array-api.html#linear-algebra).

## Matrix multiplication

The algorithm is described at https://en.wikipedia.org/wiki/Block_matrix#Block_matrix_multiplication.

As you can see, this algorithm can be implemented at the block level and
therefore use compression parallelism.

## Cholesky, LU and QR decompositions

The chunked algorithms for these decompositions can be found at https://arxiv.org/pdf/0709.1272.pdf.

In order to implement these cases, ironArray must be able to add chunks in a
disorderly way since they are not calculated sequentially.

## Solve triangular matrix

The algorithm to solve the equation $Ax=b$, assuming $a$ is a triangular matrix with $j\times j$ blocks, is:

    x[j] = SOLVE(A[j,j], b[j] - sum(matmul([A[j,i], x[i]) for i in range(j)) ) 


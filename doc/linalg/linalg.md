# Linear algebra in ironArray

The intention of this document is to collect some ideas on how to implement
linear algebra algorithms using chunked (tiled) data.

Our goal should be to implement most of the algorithms available in dask.
(See https://docs.dask.org/en/latest/array-api.html#linear-algebra)

## Algorithms

Except in the matrix multiplication algorithm, the chunks (if the algorithm is implemented
at the chunk level) or the blocks (if it is implemented at the block level) must be square.

### Matrix multiplication

The algorithm is described at https://en.wikipedia.org/wiki/Block_matrix#Block_matrix_multiplication.

As you can see, this algorithm can be implemented at the block level and
therefore use compression parallelism.

#### Complete intermediate matrix in expressions

f we want to introduce matrix multiplication within the expression evaluator,
we either have to redo many calculations (less performace) or we have to
calculate intermediate matrices (more memory).


### Cholesky, LU and QR decompositions

The chunked algorithms for these decompositions can be found at https://arxiv.org/pdf/0709.1272.pdf.

In order to implement these cases, ironArray must be able to add chunks in a
disorderly way since they are not calculated sequentially.

### Solve triangular matrix

The algorithm to solve the equation $Ax=b$, assuming $a$ is a triangular matrix with
$vchunks \times hchunks$ blocks, is:

    for j in range(vchunks):
        x[j] = solve( A[j,j], b[j] - sum( matmul([A[j,i], x[i]) for i in range(j) ) ) 

### Solve general matrix

The algorithm is based in the LU decomposition and the solve triangular matrix
algorithms. It can be found at https://en.wikipedia.org/wiki/LU_decomposition#Solving_linear_equations,

### Matrix inverse

The algorithm is described at https://en.wikipedia.org/wiki/LU_decomposition#Inverting_a_matrix.

## Roadmap

1. Implement the matrix multiplication at the block level.
2. Allow ironArray (at Caterva level?) to append disordered chunks.
3. Implement LU (also Chloselsky and QR?) decomposition.
4. Implement the triangular and general solver.
5. Implement the matrix inverse.

//
// Created by Francesc Alted on 25/09/2018.
//

/*
  Example program demonstrating how to execute an expression with super-chunks as operands.
  This is the version for using frames (either in-memory or on-disk) backing the super-chunks.

  To compile this program:

  $ gcc -O3 gemm-caterva.c -o gemm-caterva -lblosc

  To run:

  $ ./gemm-caterva memory
  ...
  $ ./gemm-caterva disk
  ...

*/

#include <libiarray/iarray.h>

#define KB (1024.)
#define MB (1024 * KB)

#define N (1000)   /* array size is (N * N) */
#define NELEM (N * N)
#define NELEM_BYTES (NELEM*sizeof(double))
#define NTHREADS 1

/* Simple matrix-matrix multiplication for square matrices */
int simple_matmul(size_t n, double const *a, double const *b, double *c)
{
    size_t i, j, k;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            double t = 0.0;
            for (k = 0; k < n; ++k)
                t += a[i*n+k] * b[k*n+j];
            c[i*n+j] = t;
        }
    }
    return 0;
}


/* Check that the values of a super-chunk are equal to a C matrix */
int test_mat_equal(double *c1, double *c2) {
    for (int nelem=0; nelem < NELEM; nelem++) {
        double vdiff = fabs((c1[nelem] - c2[nelem]) / c1[nelem]);
        if (vdiff > 1e-6) {
            printf("%f, %f\n", c1[nelem], c2[nelem]);
            printf("Values differ in (%d nelem) (diff: %f)\n", nelem, vdiff);
            return 0;
        }
    }
    return 1;
}

static double *mat_x = NULL;
static double *mat_y = NULL;
static double *mat_out = NULL;

int main(int argc, char** argv)
{
    ina_stopwatch_t *w = NULL;
    iarray_context_t *ctx = NULL;
    const char *mat_x_name = NULL;
    const char *mat_y_name = NULL;
    const char *mat_out_name = NULL;

    INA_MUST_SUCCEED(iarray_init());

    INA_OPTS(opt,
        INA_OPT_FLAG("p", "persistence", "Use persistent containers")
    );

    if (INA_SUCCEED(ina_opt_isset("p"))) {
        mat_x_name = "mat_x";
        mat_y_name = "mat_y";
        mat_out_name = "mat_out";
    }

    iarray_config_t config;
    ina_mem_set(&config, 0, sizeof(iarray_config_t));
    config.compression_codec = IARRAY_COMPRESSION_LZ4;
    config.compression_level = 5;
    config.max_num_threads = NTHREADS;
    config.flags = IARRAY_EXPR_EVAL_CHUNK;

    iarray_ctx_new(&config, &ctx);

    double elapsed_sec;
    INA_STOPWATCH_NEW(1, -1, &w);

    mat_x = (double*)ina_mem_alloc((sizeof(double)*NELEM));
    mat_y = (double*)ina_mem_alloc((sizeof(double)*NELEM));
    mat_out = (double*)ina_mem_alloc((sizeof(double)*NELEM));

    INA_STOPWATCH_START(w);
    double incx = 10. / NELEM;
    for (int i = 0; i < NELEM; i++) {
        mat_x[i] = i * incx;
        mat_y[i] = i * incx;
    }
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
    printf("Time for filling X and Y matrices: %.3g s, %.1f MB/s\n",
        elapsed_sec, (sizeof(mat_x) + sizeof(mat_y)) / (elapsed_sec * MB));

    
    /* Compute naive matrix-matrix multiplication */
    INA_STOPWATCH_START(w);
    simple_matmul(N, mat_x, mat_y, mat_out);
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
    printf("Time for multiplying two matrices (pure C): %.3g s, %.1f MB/s\n",
        elapsed_sec, (sizeof(mat_x) * 3) / (elapsed_sec * MB));

  
    iarray_dtshape_t shape;
    shape.ndim = 2;
    shape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    shape.dims[0] = N;
    shape.dims[1] = N;

    iarray_container_t *con_x;
    iarray_container_t *con_y;

    INA_STOPWATCH_START(w);
    iarray_from_buffer(ctx, &shape, IARRAY_DATA_TYPE_DOUBLE, mat_x, N, IARRAY_STORAGE_ROW_WISE, mat_x_name, 0, &con_x);
    iarray_from_buffer(ctx, &shape, IARRAY_DATA_TYPE_DOUBLE, mat_y, N, IARRAY_STORAGE_ROW_WISE, mat_y_name, 0, &con_y);
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));

    size_t nbytes = 0;
    size_t cbytes = 0;
    iarray_container_info(con_x, &nbytes, &cbytes);
    printf("Time for filling X and Y iarray-containers: %.3g s, %.1f MB/s\n",
        elapsed_sec, (nbytes * 2) / (elapsed_sec * MB));
    printf("Compression for X iarray-container: %.1f MB -> %.1f MB (%.1fx)\n",
        (nbytes / MB), (cbytes / MB),
        ((double)nbytes / cbytes));

    iarray_to_buffer(ctx, con_x, IARRAY_DATA_TYPE_DOUBLE, mat_x, NELEM_BYTES, IARRAY_STORAGE_ROW_WISE);
    iarray_to_buffer(ctx, con_y, IARRAY_DATA_TYPE_DOUBLE, mat_y, NELEM_BYTES, IARRAY_STORAGE_ROW_WISE);
    if (!test_mat_equal(mat_x, mat_y)) {
        return EXIT_FAILURE; /* FIXME: error handling */
    } 
    
    iarray_container_t *con_out;
    iarray_container_new(ctx, &shape, IARRAY_DATA_TYPE_DOUBLE, mat_out_name, 0, &con_out);

    INA_STOPWATCH_START(w);
    iarray_gemm(con_x, con_y, con_out); /* FIXME: error handling */
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
    
    iarray_container_info(con_out, &nbytes, &cbytes);
    printf("\n");
    printf("Time for multiplying two matrices (iarray):  %.3g s, %.1f MB/s\n",
        elapsed_sec, (nbytes * 3) / (elapsed_sec * MB));
    printf("Compression for OUT values: %.1f MB -> %.1f MB (%.1fx)\n",
            (nbytes/MB), (cbytes/MB),
            (1.*nbytes) / cbytes);

    /* Check that we are getting the same results than through manual computation */
    ina_mem_set(mat_out, 0, NELEM_BYTES);
    iarray_to_buffer(ctx, con_out, IARRAY_DATA_TYPE_DOUBLE, mat_out, NELEM_BYTES, IARRAY_STORAGE_ROW_WISE);
    if (!test_mat_equal(mat_out, mat_out)) {
        return EXIT_FAILURE; /* FIXME: error-handling */
    }

    iarray_container_free(ctx, &con_x);
    iarray_container_free(ctx, &con_y);
    iarray_container_free(ctx, &con_out);

    ina_mem_free(mat_x);
    ina_mem_free(mat_y);
    ina_mem_free(mat_out);

    INA_STOPWATCH_FREE(&w);

    return EXIT_SUCCESS;
}

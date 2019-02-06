/*
 * Copyright INAOS GmbH, Thalwil, 2018.
 * Copyright Francesc Alted, 2018.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of INAOS GmbH
 * and Francesc Alted ("Confidential Information"). You shall not disclose such Confidential
 * Information and shall use it only in accordance with the terms of the license agreement.
 *
 */

#include <libiarray/iarray.h>
#include <iarray_private.h>

#define NELEM_BYTES(nelem) (nelem * sizeof(double))
#define NTHREADS 1

INA_BENCH_DATA(matmul) {
    iarray_context_t *ctx;
    iarray_config_t config;
    ina_stopwatch_t *w;

    const char *mat_x_name;
    const char *mat_y_name;
    const char *mat_out_name;

    uint64_t nbytes;
    uint64_t cbytes;
    double nbytes_mb;
    double cbytes_mb;

    uint64_t shape_x_0;
    uint64_t shape_x_1;

    uint64_t pshape_x_0;
    uint64_t pshape_x_1;

    uint64_t bshape_x_0;
    uint64_t bshape_x_1;

    uint64_t size_x;

    uint64_t shape_y_0;
    uint64_t shape_y_1;

    uint64_t pshape_y_0;
    uint64_t pshape_y_1;

    uint64_t bshape_y_0;
    uint64_t bshape_y_1;

    uint64_t size_y;

    uint64_t shape_out_0;
    uint64_t shape_out_1;
    
    uint64_t pshape_out_0;
    uint64_t pshape_out_1;

    uint64_t size_out;

    uint64_t flops;

    int flags;
    iarray_store_properties_t *mat_x_prop;
    iarray_store_properties_t *mat_y_prop;
    iarray_store_properties_t *mat_out_prop;

    iarray_container_t *con_x;
    iarray_container_t *con_y;

    double *mat_x;
    double *mat_y;
    double *mat_out;
    double *mat_res;
};

static ina_rc_t _init_message(struct matmul_data *data)
{
    INA_BENCH_MSG("Matrix X has a shape of (%lld, %lld) with a partition of (%lld, %lld)",
        data->shape_x_0, data->shape_x_1, data->pshape_x_0, data->pshape_x_1);
    INA_BENCH_MSG("Matrix Y has a shape of (%lld, %lld) with a partition of (%lld, %lld)",
        data->shape_y_0, data->shape_y_1, data->pshape_y_0, data->pshape_y_1);

    INA_BENCH_MSG("Working set for the 4 uncompressed matrices: %.1f MB",
        (data->size_x + data->size_y + data->size_out * 2) * sizeof(double) / (double)_IARRAY_SIZE_MB);

    return INA_SUCCESS;
}

static ina_rc_t _set_matrix_config(struct matmul_data *data, 
    uint64_t shape_x_0, 
    uint64_t shape_x_1,
    uint64_t pshape_x_0,
    uint64_t pshape_x_1,
    uint64_t bshape_x_0,
    uint64_t bshape_x_1,
    uint64_t shape_y_0,
    uint64_t shape_y_1,
    uint64_t pshape_y_0,
    uint64_t pshape_y_1,
    uint64_t bshape_y_0,
    uint64_t bshape_y_1)
{
    data->shape_x_0 = shape_x_0;
    data->shape_x_1 = shape_x_1;

    data->pshape_x_0 = pshape_x_0;
    data->pshape_x_1 = pshape_x_1;

    data->bshape_x_0 = bshape_x_0;
    data->bshape_x_1 = bshape_x_1;

    data->size_x = shape_x_0 * shape_x_1;

    data->shape_y_0 = shape_y_0;
    data->shape_y_1 = shape_y_1;

    data->pshape_y_0 = bshape_y_0;
    data->bshape_y_1 = bshape_y_1;
    data->size_y = shape_y_0 * shape_y_1;

    data->shape_out_0 = shape_x_0;
    data->shape_out_1 = shape_y_1;

    data->pshape_out_0 = data->bshape_x_0;
    data->pshape_out_1 = data->bshape_y_1;

    data->size_out = data->shape_out_0 * data->shape_out_1;

    data->flops = (2 * data->shape_x_1 - 1) * data->shape_x_0 * data->shape_y_1;

    data->flags = 0;
    data->mat_x_prop = NULL;
    data->mat_y_prop = NULL;
    data->mat_out_prop = NULL;

    return INA_SUCCESS;
}

static ina_rc_t _setup_matrices(struct matmul_data *data, int allocated)
{
    double elapsed_sec = 0;

    INA_MUST_SUCCEED(iarray_context_new(&data->config, &data->ctx));

    data->mat_x = (double *)ina_mem_alloc((sizeof(double) * data->size_x));
    data->mat_y = (double *)ina_mem_alloc((sizeof(double) * data->size_y));

    INA_STOPWATCH_START(data->w);
    double incx = 10. / data->size_x;
    for (uint64_t i = 0; i < data->size_x; i++) {
        data->mat_x[i] = i * incx;
    }
    double incy = 10. / data->size_y;
    for (uint64_t i = 0; i < data->size_y; i++) {
        data->mat_y[i] = i * incy;
    }
    INA_STOPWATCH_STOP(data->w);

    INA_MUST_SUCCEED(ina_stopwatch_duration(data->w, &elapsed_sec));
    printf("Time for filling X and Y matrices: %.3g s, %.1f MB/s\n",
        elapsed_sec, NELEM_BYTES(data->size_x + data->size_y) / (elapsed_sec * _IARRAY_SIZE_MB));

    iarray_dtshape_t xdtshape;
    xdtshape.ndim = 2;
    xdtshape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    
    xdtshape.shape[0] = data->shape_x_0;
    xdtshape.pshape[1] = data->pshape_x_1;
    xdtshape.shape[0] = data->shape_x_0;
    xdtshape.pshape[1] = data->pshape_x_1;
    
    iarray_dtshape_t ydtshape;
    ydtshape.ndim = 2;
    ydtshape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    ydtshape.shape[0] = data->shape_y_0;
    ydtshape.pshape[1] = data->pshape_y_1;
    ydtshape.shape[0] = data->shape_y_0;
    ydtshape.pshape[1] = data->pshape_y_1;

    INA_STOPWATCH_START(data->w);
    INA_MUST_SUCCEED(iarray_from_buffer(data->ctx, &xdtshape, data->mat_x, data->size_x, &data->mat_x_prop, data->flags, &data->con_x));
    INA_MUST_SUCCEED(iarray_from_buffer(data->ctx, &ydtshape, data->mat_y, data->size_y, &data->mat_y_prop, data->flags, &data->con_y));
    INA_STOPWATCH_STOP(data->w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(data->w, &elapsed_sec));

    iarray_container_info(data->con_x, &data->nbytes, &data->cbytes);
    INA_BENCH_MSG("Time for filling X and Y iarray-containers: %.3g s, %.1f MB/s",
        elapsed_sec, NELEM_BYTES(data->size_x + data->size_y) / (elapsed_sec * _IARRAY_SIZE_MB));
    data->nbytes_mb = ((double)data->nbytes / _IARRAY_SIZE_MB);
    data->cbytes_mb = ((double)data->cbytes / _IARRAY_SIZE_MB);
    INA_BENCH_MSG("Compression for X iarray-container: %.1f MB -> %.1f MB (%.1fx)",
        data->nbytes_mb, data->cbytes_mb, ((double)data->nbytes / data->cbytes));

    if (allocated == 0) {
        INA_MUST_SUCCEED(iarray_to_buffer(data->ctx, data->con_x, data->mat_x, NELEM_BYTES(data->size_x)));
        INA_MUST_SUCCEED(iarray_to_buffer(data->ctx, data->con_y, data->mat_y, NELEM_BYTES(data->size_y)));
    }

    data->mat_out = (double*)ina_mem_alloc((sizeof(double) * data->size_out));
    data->mat_res = (double*)ina_mem_alloc((sizeof(double) * data->size_out));

    return INA_SUCCESS;
}

static ina_rc_t _teardown_matrices(struct matmul_data *data)
{
    iarray_container_free(data->ctx, &data->con_x);
    iarray_container_free(data->ctx, &data->con_y);

    iarray_context_free(&data->ctx);

    ina_mem_free(data->mat_x);
    ina_mem_free(data->mat_y);
    ina_mem_free(data->mat_out);
    ina_mem_free(data->mat_res);

    return INA_SUCCESS;
}

INA_BENCH_SETUP(matmul)
{
    ina_bench_set_scale_label("matrix_result_size");
    ina_bench_set_precision(0);

    INA_STOPWATCH_NEW(-1, -1, &data->w);

    INA_MUST_SUCCEED(iarray_init());

    data->config = IARRAY_CONFIG_DEFAULTS;
    data->config.compression_codec = IARRAY_COMPRESSION_LZ4;
    data->config.compression_level = 5;
    data->config.max_num_threads = NTHREADS;
    data->config.eval_flags = IARRAY_EXPR_EVAL_CHUNK;

    INA_BENCH_MSG("Measuring time for multiplying matrices X and Y");
}

INA_BENCH_TEARDOWN(matmul)
{
    // FIXME: compare
    INA_STOPWATCH_FREE(&data->w);
    iarray_destroy();
}

INA_BENCH_BEGIN(matmul, native_mkl)
{
    _set_matrix_config(data, 4000, 5000, 500, 750, 1000,
        1200, 5000, 3000, 400, 510, 1200, 1100);
}

INA_BENCH_SCALE(matmul) {
    data->shape_x_0 = data->shape_x_0 + (100 * ina_bench_get_iteration());
    ina_bench_set_scale(data->size_out);
}

INA_BENCH(matmul, native_mkl, 2)
{
    double elapsed_sec = 0;

    _init_message(data);

    _setup_matrices(data, 0);
    
    /* Compute MKL matrix-matrix multiplication */
    INA_STOPWATCH_START(data->w);
    ina_bench_stopwatch_start();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)data->shape_x_0, (int)data->shape_y_1, (int)data->shape_x_1,
        1.0, data->mat_x, (int)data->shape_x_1, data->mat_y, (int)data->shape_y_1, 0.0, data->mat_res, (int)data->shape_y_1);
    ina_bench_set_int64(ina_bench_stopwatch_stop());
    INA_STOPWATCH_STOP(data->w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(data->w, &elapsed_sec));

    INA_BENCH_MSG("Time for multiplying two matrices (pure C, MKL): %.3g s, %.1f GFLOPs\n",
        elapsed_sec, data->flops / (elapsed_sec * 10e9));
}

INA_BENCH_END(matmul, native_mkl)
{
    _teardown_matrices(data);
}

INA_BENCH_BEGIN(matmul, linalg_matmul)
{
    _set_matrix_config(data, 4000, 5000, 500, 750, 1000,
        1200, 5000, 3000, 400, 510, 1200, 1100);
}

INA_BENCH(matmul, linalg_matmul, 2)
{
    double elapsed_sec = 0;

    _init_message(data);

    _setup_matrices(data, 0);

    iarray_dtshape_t outdtshape;
    outdtshape.ndim = 2;
    outdtshape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    
    outdtshape.shape[0] = data->shape_out_0;
    outdtshape.pshape[1] = data->pshape_out_1;
    outdtshape.shape[0] = data->shape_out_0;
    outdtshape.pshape[1] = data->pshape_out_1;

    uint64_t bshape_x[] = { data->bshape_x_0, data->bshape_x_1 };
    uint64_t bshape_y[] = { data->bshape_y_0, data->bshape_y_1 };

    iarray_container_t *con_out;
    iarray_container_new(data->ctx, &outdtshape, &data->mat_out_prop, 0, &con_out);

    INA_STOPWATCH_START(data->w);
    ina_bench_stopwatch_start();
    iarray_linalg_matmul(data->ctx, data->con_x, data->con_y, con_out, bshape_x, bshape_y, IARRAY_OPERATOR_GENERAL); /* FIXME: error handling */
    ina_bench_set_int64(ina_bench_stopwatch_stop());
    INA_STOPWATCH_STOP(data->w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(data->w, &elapsed_sec));

    iarray_container_info(con_out, &data->nbytes, &data->cbytes);
    INA_BENCH_MSG("Time for multiplying two matrices (iarray):  %.3g s, %.1f GFLOPs",
        elapsed_sec, data->flops / (elapsed_sec * 10e9));

    data->nbytes_mb = ((double)data->nbytes / _IARRAY_SIZE_MB);
    data->cbytes_mb = ((double)data->cbytes / _IARRAY_SIZE_MB);
    INA_BENCH_MSG("Compression for OUT values: %.1f MB -> %.1f MB (%.1fx)",
        data->nbytes_mb, data->cbytes_mb, (1.*data->nbytes) / data->cbytes);
}

INA_BENCH_END(matmul, linalg_matmul)
{
    _teardown_matrices(data);
}


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


int main(int argc, char** argv)
{
    int n_threads = 1;
    if (argc != 1) {
        n_threads = atoi(argv[0]);
    }

    printf("Nthreads: %d\n", n_threads);

    iarray_init();
    ina_stopwatch_t *w = NULL;
    double elapsed_sec = 0;
    INA_STOPWATCH_NEW(-1, -1, &w);

    // Define example parameters
    int8_t ndim = 2;
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    iarray_storage_type_t storage_format = IARRAY_STORAGE_BLOSC;


    int64_t shape_x[] = {8192, 8192};
    int64_t shape_y[] = {8192, 8192};

    int64_t size_x = 1;
    int64_t size_y = 1;
    for (int i = 0; i < ndim; ++i) {
        size_x *= shape_x[i];
        size_y *= shape_y[i];
    }

    int64_t cshape_x[] = {4096, 4096};
    int64_t cshape_y[] = {4096, 4096};
    int64_t cshape_z[] = {4096, 4096};

    int64_t bshape_x[] = {1024, 1024};
    int64_t bshape_y[] = {1024, 1024};
    int64_t bshape_z[] = {1024, 1024};

    // Create context
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.max_num_threads = n_threads;
    iarray_context_t *ctx;
    IARRAY_RETURN_IF_FAILED(iarray_context_new(&cfg, &ctx));

    // Create dtshape
    iarray_dtshape_t dtshape_x;
    dtshape_x.ndim = ndim;
    dtshape_x.dtype = dtype;
    for (int i = 0; i < ndim; ++i) {
        dtshape_x.shape[i] = shape_x[i];
    }

    iarray_storage_t store_x;
    store_x.backend = storage_format;
    store_x.enforce_frame = false;
    store_x.filename = NULL;
    for (int i = 0; i < ndim; ++i) {
        store_x.chunkshape[i] = cshape_x[i];
        store_x.blockshape[i] = bshape_x[i];
    }
    iarray_container_t *c_x;
    IARRAY_RETURN_IF_FAILED(iarray_linspace(ctx, &dtshape_x, size_x, 0, 1, &store_x, 0, &c_x));


    iarray_dtshape_t dtshape_y;
    dtshape_y.ndim = ndim;
    dtshape_y.dtype = dtype;
    for (int i = 0; i < ndim; ++i) {
        dtshape_y.shape[i] = shape_y[i];
    }
    iarray_storage_t store_y;
    store_y.backend = storage_format;
    store_y.enforce_frame = false;
    store_y.filename = NULL;
    for (int i = 0; i < ndim; ++i) {
        store_y.chunkshape[i] = cshape_y[i];
        store_y.blockshape[i] = bshape_y[i];
    }
    iarray_container_t *c_y;
    IARRAY_RETURN_IF_FAILED(iarray_linspace(ctx, &dtshape_y, size_y, 0, 1, &store_y, 0, &c_y));


    iarray_storage_t store_z;
    store_z.backend = storage_format;
    store_z.enforce_frame = false;
    store_z.filename = NULL;
    for (int i = 0; i < ndim; ++i) {
        store_z.chunkshape[i] = cshape_z[i];
        store_z.blockshape[i] = bshape_z[i];
    }
    iarray_container_t *c_z_parallel;

    INA_STOPWATCH_START(w);
    IARRAY_RETURN_IF_FAILED(iarray_linalg_parallel_matmul(ctx, c_x, c_y, &store_z, &c_z_parallel));
    INA_STOPWATCH_STOP(w);
    IARRAY_RETURN_IF_FAILED(ina_stopwatch_duration(w, &elapsed_sec));

    printf("Time parallel version: %.4f\n", elapsed_sec);

    iarray_container_t *c_z_old;
    iarray_dtshape_t dtshape_z = {0};
    dtshape_z.ndim = ndim;
    dtshape_z.dtype = dtype;
    dtshape_z.shape[0] = shape_x[0];
    dtshape_z.shape[1] = shape_y[1];

    INA_RETURN_IF_FAILED(iarray_container_new(ctx, &dtshape_z, &store_z, 0, &c_z_old));

    int64_t bshape_a[2] = {cshape_z[0], 350};
    int64_t bshape_b[2] = {350, cshape_z[1]};
    INA_STOPWATCH_START(w);
    IARRAY_RETURN_IF_FAILED(iarray_linalg_matmul(ctx, c_x, c_y, c_z_old, bshape_a, bshape_b, IARRAY_OPERATOR_GENERAL));
    INA_STOPWATCH_STOP(w);
    IARRAY_RETURN_IF_FAILED(ina_stopwatch_duration(w, &elapsed_sec));

    printf("Time single-thread version: %.4f\n", elapsed_sec);

    // MKL dgemm
    iarray_container_t *a = c_x;
    size_t size_a = a->catarr->nitems * a->catarr->itemsize;
    uint8_t *buffer_a = malloc(size_a);
    IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, a, buffer_a, size_a));

    iarray_container_t *b = c_y;
    size_t size_b = b->catarr->nitems * b->catarr->itemsize;
    uint8_t *buffer_b = malloc(size_b);
    IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, b, buffer_b, size_b));

    int m = a->dtshape->shape[0];
    int k = a->dtshape->shape[1];
    int n = b->dtshape->shape[1];

    int ld_a = k;
    int ld_b = n;
    int ld_c = n;

    iarray_container_t *c = c_z_parallel;
    size_t size_c = c->catarr->nitems * c->catarr->itemsize;
    uint8_t *buffer_c = malloc(size_c);

    INA_STOPWATCH_START(w);
    if (c->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) m, (int) n, (int) k,
                    1.0, (double *) buffer_a, ld_a, (double *) buffer_b, ld_b, 0.0, (double *) buffer_c,
                    ld_c);
    } else {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) m, (int) n, (int) k,
                    1.0f, (float *) buffer_a, ld_a, (float *) buffer_b, ld_b, 0.0f, (float *) buffer_c,
                    ld_c);
    }
    INA_STOPWATCH_STOP(w);
    IARRAY_RETURN_IF_FAILED(ina_stopwatch_duration(w, &elapsed_sec));

    printf("Time MKL version: %.4f\n", elapsed_sec);

    // Testing
    IARRAY_RETURN_IF_FAILED(iarray_container_almost_equal(ctx, c_z_parallel, c_z_old));


    // Free allocated memory
    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);
    iarray_container_free(ctx, &c_z_parallel);
    iarray_container_free(ctx, &c_z_old);
    iarray_context_free(&ctx);
    INA_STOPWATCH_FREE(&w);

    iarray_destroy();

    return INA_SUCCESS;
}

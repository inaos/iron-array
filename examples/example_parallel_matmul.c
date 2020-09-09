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

int main(void)
{
    int n_threads = 4;

    iarray_init();
    ina_stopwatch_t *w = NULL;
    double elapsed_sec = 0;
    INA_STOPWATCH_NEW(-1, -1, &w);

    // Define example parameters
    int8_t ndim = 2;
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int64_t shape_x[] = {4000, 4000};
    int64_t shape_y[] = {4000, 4000};

    int64_t size_x = 1;
    int64_t size_y = 1;
    for (int i = 0; i < ndim; ++i) {
        size_x *= shape_x[i];
        size_y *= shape_y[i];
    }

    int64_t cshape_x[] = {1000, 1000};
    int64_t cshape_y[] = {1000, 1000};
    int64_t cshape_z[] = {1000, 1000};
    
    int64_t bshape_x[] = {250, 250};
    int64_t bshape_y[] = {250, 250};
    int64_t bshape_z[] = {250, 250};

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
    store_x.backend = IARRAY_STORAGE_BLOSC;
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
    store_y.backend = IARRAY_STORAGE_BLOSC;
    store_y.enforce_frame = false;
    store_y.filename = NULL;
    for (int i = 0; i < ndim; ++i) {
        store_y.chunkshape[i] = cshape_y[i];
        store_y.blockshape[i] = bshape_y[i];
    }
    iarray_container_t *c_y;
    IARRAY_RETURN_IF_FAILED(iarray_linspace(ctx, &dtshape_y, size_y, 0, 1, &store_y, 0, &c_y));


    iarray_storage_t store_z;
    store_z.backend = IARRAY_STORAGE_BLOSC;
    store_z.enforce_frame = false;
    store_z.filename = NULL;
    for (int i = 0; i < ndim; ++i) {
        store_z.chunkshape[i] = cshape_z[i];
        store_z.blockshape[i] = bshape_z[i];
    }
    iarray_container_t *c_z;

    INA_STOPWATCH_START(w);
    IARRAY_RETURN_IF_FAILED(iarray_linalg_parallel_matmul(ctx, c_x, c_y, &store_z, &c_z));
    INA_STOPWATCH_STOP(w);
    IARRAY_RETURN_IF_FAILED(ina_stopwatch_duration(w, &elapsed_sec));

    printf("Time mkl (C): %.4f\n", elapsed_sec);

    // Testing
    int64_t c_size = 1;
    for (int i = 0; i < c_z->dtshape->ndim; ++i) {
        c_size *= c_z->dtshape->shape[i];
    }

    int64_t buffer_size = c_size * c_z->catarr->itemsize;
    double *buffer = ina_mem_alloc(buffer_size);

    IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, c_z, buffer, buffer_size));

    for (int i = 0; i < c_size; ++i) {
        // printf("%f - ", buffer[i]);
    }

    INA_MEM_FREE_SAFE(buffer);


    // Free allocated memory
    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);
    iarray_container_free(ctx, &c_z);
    iarray_context_free(&ctx);
    INA_STOPWATCH_FREE(&w);

    iarray_destroy();

    return INA_SUCCESS;
}

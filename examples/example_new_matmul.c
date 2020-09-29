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
        n_threads = atoi(argv[1]);
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

    int64_t max_cache = 64  * 1024 * 1024;

    int64_t sh_in = 4096;
    int64_t cs_in = 1000;
    int64_t bs_in = 512;

    int64_t sh = 4096;
    int64_t cs = 512;
    int64_t bs = 64;

    int64_t shape_x[] = {sh_in, sh_in};
    int64_t shape_y[] = {sh_in, sh_in};

    int64_t size_x = 1;
    int64_t size_y = 1;
    for (int i = 0; i < ndim; ++i) {
        size_x *= shape_x[i];
        size_y *= shape_y[i];
    }

    int64_t cshape_x[] = {cs_in, cs_in};
    int64_t cshape_y[] = {cs_in, cs_in};
    int64_t cshape_z[] = {cs, cs};

    int64_t bshape_x[] = {bs_in, bs_in};
    int64_t bshape_y[] = {bs_in, bs_in};
    int64_t bshape_z[] = {bs, bs};

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
    IARRAY_RETURN_IF_FAILED(iarray_arange(ctx, &dtshape_x, 0, size_x, 1, &store_x, 0, &c_x));


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
    IARRAY_RETURN_IF_FAILED(iarray_arange(ctx, &dtshape_y, 0, size_y, 1, &store_y, 0, &c_y));


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
    // IARRAY_RETURN_IF_FAILED(iarray_linalg_parallel_matmul(ctx, c_x, c_y, &store_z, &c_z_parallel));
    INA_STOPWATCH_STOP(w);
    IARRAY_RETURN_IF_FAILED(ina_stopwatch_duration(w, &elapsed_sec));

    printf("Time parallel version: %.4f\n", elapsed_sec);

    iarray_container_t *c_z_parallel2;

    INA_STOPWATCH_START(w);
    // IARRAY_RETURN_IF_FAILED(iarray_linalg_parallel_matmul2(ctx, c_x, c_y, max_cache, &store_z, &c_z_parallel2));
    INA_STOPWATCH_STOP(w);
    IARRAY_RETURN_IF_FAILED(ina_stopwatch_duration(w, &elapsed_sec));

    printf("Time parallel version 2: %.4f\n", elapsed_sec);

    iarray_container_t *c_z_parallel3;

    INA_STOPWATCH_START(w);
    // IARRAY_RETURN_IF_FAILED(iarray_linalg_parallel_matmul3(ctx, c_x, c_y, &store_z, &c_z_parallel3));
    INA_STOPWATCH_STOP(w);
    IARRAY_RETURN_IF_FAILED(ina_stopwatch_duration(w, &elapsed_sec));

    printf("Time parallel version 3: %.4f\n", elapsed_sec);

    iarray_container_t *c_z_parallel4;

    INA_STOPWATCH_START(w);
    IARRAY_RETURN_IF_FAILED(iarray_linalg_parallel_matmul4(ctx, c_x, c_y, max_cache, &store_z, &c_z_parallel4));
    INA_STOPWATCH_STOP(w);
    IARRAY_RETURN_IF_FAILED(ina_stopwatch_duration(w, &elapsed_sec));

    printf("Time parallel version 4: %.4f\n", elapsed_sec);

    iarray_container_t *c_z_parallel5;

    INA_STOPWATCH_START(w);
    IARRAY_RETURN_IF_FAILED(iarray_linalg_parallel_matmul5(ctx, c_x, c_y, &store_z, &c_z_parallel5));
    INA_STOPWATCH_STOP(w);
    IARRAY_RETURN_IF_FAILED(ina_stopwatch_duration(w, &elapsed_sec));

    printf("Time parallel version 5: %.4f\n", elapsed_sec);

    // iarray_container_almost_equal(ctx, c_z_parallel, c_z_parallel3);
    // iarray_container_almost_equal(ctx, c_z_parallel2, c_z_parallel3);
    // iarray_container_almost_equal(ctx, c_z_parallel3, c_z_parallel4);
    iarray_container_almost_equal(ctx, c_z_parallel4, c_z_parallel5);

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);
    // iarray_container_free(ctx, &c_z_parallel);
    // iarray_container_free(ctx, &c_z_parallel2);
    // iarray_container_free(ctx, &c_z_parallel3);
    iarray_container_free(ctx, &c_z_parallel4);
    iarray_container_free(ctx, &c_z_parallel5);
    iarray_context_free(&ctx);
    INA_STOPWATCH_FREE(&w);

    iarray_destroy();

    return INA_SUCCESS;
}

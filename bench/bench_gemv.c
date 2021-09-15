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


int main(void)
{
    iarray_init();
    ina_stopwatch_t *w = NULL;
    double elapsed_sec = 0;
    INA_STOPWATCH_NEW(-1, -1, &w);

    int n_threads = 1;
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim_x = 2;
    int8_t ndim_y = 1;
    int8_t ndim_z = 1;

    int64_t shape_x[] = {1000000, 140};
    int64_t shape_y[] = {140};
    int64_t shape_z[] = {1000000};

//    int64_t size_x = shape_x[0] * shape_x[1];
//    int64_t size_y = shape_y[0];
//    int64_t size_z = shape_z[0];

    //int64_t cshape_x[] = {5000, 140};
    int64_t cshape_y[] = {140};
    int64_t cshape_z[] = {5000};

    //int64_t bshape_x[] = {200, 140};
    int64_t bshape_y[] = {140};
    int64_t bshape_z[] = {200};

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.max_num_threads = n_threads;
    cfg.compression_level = 9;
    cfg.btune = false;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;

    iarray_context_t *ctx;
    IARRAY_FAIL_IF_ERROR(iarray_context_new(&cfg, &ctx));

    iarray_dtshape_t dtshape_x;
    dtshape_x.ndim = ndim_x;
    dtshape_x.dtype = dtype;
    for (int i = 0; i < ndim_x; ++i) {
        dtshape_x.shape[i] = shape_x[i];
    }

//    iarray_storage_t store_x;
//    store_x.backend = IARRAY_STORAGE_BLOSC;
//    store_x.contiguous = false;
//    store_x.urlpath = "arr.iarr";
//    for (int i = 0; i < ndim_x; ++i) {
//        store_x.chunkshape[i] = cshape_x[i];
//        store_x.blockshape[i] = bshape_x[i];
//    }
//    IARRAY_FAIL_IF_ERROR(iarray_linspace(ctx, &dtshape_x, 3, 2, &store_x, 0, &c_x));
    iarray_container_t *c_x;
    int niter = 100;
    //IARRAY_FAIL_IF_ERROR(iarray_container_load(ctx, "../examples/arr.iarr", &c_x));
    IARRAY_FAIL_IF_ERROR(iarray_container_open(ctx, "../examples/arr.iarr", &c_x));

    iarray_dtshape_t dtshape_y;
    dtshape_y.ndim = ndim_y;
    dtshape_y.dtype = dtype;
    for (int i = 0; i < ndim_y; ++i) {
        dtshape_y.shape[i] = shape_y[i];
    }
    iarray_storage_t store_y;
    store_y.backend = IARRAY_STORAGE_BLOSC;
    store_y.contiguous = false;
    store_y.urlpath = NULL;
    for (int i = 0; i < ndim_y; ++i) {
        store_y.chunkshape[i] = cshape_y[i];
        store_y.blockshape[i] = bshape_y[i];
    }
    iarray_container_t *c_y;
    //IARRAY_FAIL_IF_ERROR(iarray_linspace(ctx, &dtshape_y, 2, 10, &store_y, 0, &c_y));
    IARRAY_FAIL_IF_ERROR(iarray_zeros(ctx, &dtshape_y, &store_y, 0, &c_y));

    iarray_dtshape_t dtshape_z;
    dtshape_z.ndim = ndim_z;
    dtshape_z.dtype = dtype;
    for (int i = 0; i < ndim_z; ++i) {
        dtshape_z.shape[i] = shape_z[i];
    }
    iarray_storage_t store_z;
    store_z.backend = IARRAY_STORAGE_BLOSC;
    store_z.contiguous = false;
    store_z.urlpath = NULL;
    for (int i = 0; i < ndim_z; ++i) {
        store_z.chunkshape[i] = cshape_z[i];
        store_z.blockshape[i] = bshape_z[i];
    }
    iarray_container_t *c_z;

    INA_STOPWATCH_START(w);
    for (int j = 0; j < niter; j++) {
        if (INA_FAILED(iarray_opt_gemv(ctx, c_x, c_y, false, &store_z, &c_z))) {
            fprintf(stderr, "Error in linalg_gemv1: %s\n", ina_err_strerror(ina_err_get_rc()));
            goto fail;
        }
    }
    INA_STOPWATCH_STOP(w);
    IARRAY_FAIL_IF_ERROR(ina_stopwatch_duration(w, &elapsed_sec));

    printf("Time iarray gemv: %.4f\n", elapsed_sec);

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);
    iarray_container_free(ctx, &c_z);
    iarray_context_free(&ctx);

    INA_STOPWATCH_FREE(&w);
    iarray_destroy();

    return INA_SUCCESS;
    fail:
    return (int)ina_err_get_rc();
}

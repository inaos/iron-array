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
#include <math.h>
#include "iarray_private.h"



int main(void) {

    iarray_init();
    ina_stopwatch_t *w = NULL;
    double elapsed_sec = 0;
    INA_STOPWATCH_NEW(-1, -1, &w);

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.btune = false;
    cfg.max_num_threads = 4;

    iarray_context_t *ctx;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &ctx));
    iarray_random_ctx_t *rnd_ctx;
    INA_TEST_ASSERT_SUCCEED(iarray_random_ctx_new(ctx, 1234, IARRAY_RANDOM_RNG_MRG32K3A, &rnd_ctx));

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t xndim = 2;
    int64_t xshape[] = {12 * 1024, 12 * 1024};
    int64_t xcshape[] = {6 * 1024, 6 * 1024};
    int64_t xbshape[] = {256, 256};

    // Define iarray container x
    iarray_dtshape_t xdtshape;
    xdtshape.ndim = xndim;
    xdtshape.dtype = dtype;
    for (int i = 0; i < xdtshape.ndim; ++i) {
        xdtshape.shape[i] = xshape[i];
    }

    iarray_storage_t xstore;
    xstore.urlpath = NULL;
    xstore.contiguous = true;
    for (int i = 0; i < xdtshape.ndim; ++i) {
        xstore.chunkshape[i] = xcshape[i];
        xstore.blockshape[i] = xbshape[i];
    }
    int64_t nelem = 1;
    for (int i = 0; i < xndim; ++i) {
        nelem *= xshape[i];
    }

    iarray_random_dist_set_param(rnd_ctx, IARRAY_RANDOM_DIST_PARAM_A, 2.);
    iarray_random_dist_set_param(rnd_ctx, IARRAY_RANDOM_DIST_PARAM_B, 3.);

    iarray_container_t *c_x;
    INA_STOPWATCH_START(w);
    // INA_TEST_ASSERT_SUCCEED(iarray_arange(ctx, &xdtshape, 1, 2, &xstore, &c_x));
    // INA_TEST_ASSERT_SUCCEED(iarray_random_uniform(ctx, &xdtshape, rnd_ctx, &xstore, 0, &c_x));

    INA_STOPWATCH_STOP(w);
    IARRAY_RETURN_IF_FAILED(ina_stopwatch_duration(w, &elapsed_sec));
    printf("Time: %.4f\n", elapsed_sec);

    iarray_container_t *c_y;
    INA_STOPWATCH_START(w);
    // INA_TEST_ASSERT_SUCCEED(iarray_linspace(ctx, &xdtshape, 1, 2, &xstore, &c_y));
    // INA_TEST_ASSERT_SUCCEED(iarray_random_uniform(ctx, &xdtshape, rnd_ctx, &xstore, &c_y));
    INA_STOPWATCH_STOP(w);
    IARRAY_RETURN_IF_FAILED(ina_stopwatch_duration(w, &elapsed_sec));
    printf("Time with prefilter: %.4f\n", elapsed_sec);

    int64_t size = nelem * sizeof(double);

    double *buf_x = ina_mem_alloc(size);
    double *buf_y = ina_mem_alloc(size);

    IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, c_x, buf_x, size));
    IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, c_y, buf_y, size));

    double atol = 1e-14;
    double rtol = 1e-14;
    for (int i = 0; i < nelem; ++i) {
        double a = buf_x[i];
        double b = buf_y[i];
        if (fabs(a - b) > atol + rtol * fabs(b)) {
            printf("ERRROR at %d (%f - %f)\n", i, a, b);
            return -1;
        }
    }

    INA_MEM_FREE_SAFE(buf_x);
    INA_MEM_FREE_SAFE(buf_y);

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);

    iarray_random_ctx_free(ctx, &rnd_ctx);
    iarray_context_free(&ctx);

    iarray_destroy();

    return INA_SUCCESS;
}
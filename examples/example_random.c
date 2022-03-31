/*
 * Copyright ironArray SL 2021.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of ironArray SL
 * ("Confidential Information"). You shall not disclose such Confidential
 * Information and shall use it only in accordance with the terms of the license agreement.
 *
 */

#include <libiarray/iarray.h>
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

//    int64_t xshape[] = {8 * 1000, 8 * 1000};
//    int64_t xcshape[] = {1000, 1000};
//    int64_t xbshape[] = {250, 250};
    int64_t xshape[] = {8, 8};
    int64_t xcshape[] = {4, 4};
    int64_t xbshape[] = {3, 3};

    int64_t size = 1;
    for (int i = 0; i < xndim; ++i) {
        size *= xshape[i];
    }

    //Define iarray container x
    iarray_dtshape_t dtshape;
    dtshape.ndim = xndim;
    dtshape.dtype = dtype;
    for (int i = 0; i < dtshape.ndim; ++i) {
        dtshape.shape[i] = xshape[i];
    }

    iarray_storage_t store;
    store.urlpath = NULL;
    store.contiguous = true;
    for (int i = 0; i < dtshape.ndim; ++i) {
        store.chunkshape[i] = xcshape[i];
        store.blockshape[i] = xbshape[i];
    }

    iarray_container_t *x;
    INA_STOPWATCH_START(w);
    INA_TEST_ASSERT_SUCCEED(iarray_random_rand(ctx, &dtshape, rnd_ctx, &store, &x));
    INA_STOPWATCH_STOP(w);
    IARRAY_RETURN_IF_FAILED(ina_stopwatch_duration(w, &elapsed_sec));
    printf("Time: %.4f\n", elapsed_sec);

    iarray_container_t *y;
    INA_STOPWATCH_START(w);
    INA_TEST_ASSERT_SUCCEED(iarray_random_rand(ctx, &dtshape, rnd_ctx, &store, &y));
    INA_STOPWATCH_STOP(w);
    IARRAY_RETURN_IF_FAILED(ina_stopwatch_duration(w, &elapsed_sec));
    printf("Time prefilter: %.4f\n", elapsed_sec);

    bool res = false;

    iarray_random_kstest(ctx, x, y, &res);

    printf("Res: %d\n", res);


    int64_t bufsize = size * (int64_t)sizeof(double);
    double *buf = ina_mem_alloc(bufsize);

    iarray_to_buffer(ctx, x, buf, bufsize);

    for (int i = 0; i < size; ++i) {
        if (i % xshape[0] == 0) {
            printf("\n");
        }
        printf("%8.2f", buf[i]);
    }
    INA_MEM_FREE_SAFE(buf);

    iarray_container_free(ctx, &x);
    iarray_container_free(ctx, &y);

    iarray_random_ctx_free(ctx, &rnd_ctx);
    iarray_context_free(&ctx);

    INA_STOPWATCH_FREE(&w);

    iarray_destroy();

    return INA_SUCCESS;
}
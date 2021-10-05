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

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t xndim = 2;
    int64_t xshape[] = {10, 10};
    int64_t xcshape[] = {5, 2};
    int64_t xbshape[] = {3, 2};

    //Define iarray container x
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

    iarray_container_t *c_x;
    INA_STOPWATCH_START(w);

    INA_TEST_ASSERT_SUCCEED(iarray_tri(ctx, &xdtshape, -1, &xstore, &c_x));
    INA_STOPWATCH_STOP(w);
    IARRAY_RETURN_IF_FAILED(ina_stopwatch_duration(w, &elapsed_sec));
    printf("Time eye: %.4f\n", elapsed_sec);

    int64_t size = nelem * sizeof(double);

    double *buf_x = ina_mem_alloc(size);

    IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, c_x, buf_x, size));

    for (int i = 0; i < xshape[0]; ++i) {
        for (int j = 0; j < xshape[1]; ++j) {
            printf("%8.4f", buf_x[j + i * xshape[1]]);
        }
        printf("\n");
    }

    INA_MEM_FREE_SAFE(buf_x);

    iarray_container_free(ctx, &c_x);

    iarray_context_free(&ctx);

    INA_STOPWATCH_FREE(&w);

    iarray_destroy();

    return INA_SUCCESS;
}
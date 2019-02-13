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



int main(int argc, char **argv)
{
    ina_stopwatch_t *w;
    double elapsed_slice, elapsed_view;
    INA_STOPWATCH_NEW(-1, -1, &w);

    uint64_t shape_x[] = {3000, 3000};
    uint64_t pshape_x[] = {20, 20};
    uint64_t offset_x[] = {0, 0};

    uint64_t pshape_y[] = {10, 10};
    uint64_t pshape_z[] = {10, 10};

    int64_t start[] = {10, 1000};
    int64_t stop[] = {2400, 2600};

    iarray_context_t *ctx;

    iarray_config_t config = IARRAY_CONFIG_DEFAULTS;
    config.compression_level = 5;
    config.compression_codec = 1;
    config.max_num_threads = 2;

    INA_MUST_SUCCEED(iarray_context_new(&config, &ctx));

    iarray_dtshape_t dtshape_x;
    dtshape_x.dtype = IARRAY_DATA_TYPE_DOUBLE;
    dtshape_x.ndim = 2;
    uint64_t size_x = 1;
    for (int i = 0; i < dtshape_x.ndim; ++i) {
        dtshape_x.shape[i] = shape_x[i];
        dtshape_x.pshape[i] = pshape_x[i];
        dtshape_x.offset[i] = offset_x[i];
        size_x *= shape_x[i];
    }

    iarray_container_t *c_x;
    INA_MUST_SUCCEED(iarray_arange(ctx, &dtshape_x, 0, size_x, 1, NULL, 0, &c_x));

    INA_STOPWATCH_START(w);
    iarray_container_t *c_y;
    INA_MUST_SUCCEED(iarray_get_slice(ctx, c_x, start, stop, pshape_y, NULL, 0, 0, &c_y));
    INA_STOPWATCH_STOP(w);

    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_slice));
    printf("Time for get_slice: %f\n", elapsed_slice);

    INA_STOPWATCH_START(w);
    iarray_container_t *c_z;
    INA_MUST_SUCCEED(iarray_get_slice(ctx, c_x, start, stop, pshape_z, NULL, 0, 1, &c_z));
    INA_STOPWATCH_STOP(w);

    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_view));
    printf("Time for get_slice (view): %f\n", elapsed_view);

    printf("Speed-up: %f", elapsed_slice / elapsed_view);

    INA_STOPWATCH_FREE(&w);

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);
    iarray_container_free(ctx, &c_z);

    iarray_context_free(&ctx);
    return 0;
}
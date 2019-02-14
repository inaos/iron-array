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
#include <src/iarray_private.h>

int main(int argc, char **argv)
{
    ina_stopwatch_t *w;
    double elapsed_slice, elapsed_view;
    INA_STOPWATCH_NEW(-1, -1, &w);

    uint64_t shape_x[] = {10, 10};
    uint64_t pshape_x[] = {2, 2};
    uint64_t offset_x[] = {0, 0};

    uint64_t pshape_y[] = {2, 2};
    uint64_t pshape_z[] = {2, 2};

    int64_t start[] = {1, 3};
    int64_t stop[] = {5, 7};

    uint64_t bshape[] = {2, 2};
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

    printf("Speed-up: %f\n", elapsed_slice / elapsed_view);

    INA_STOPWATCH_FREE(&w);

    iarray_iter_read_block_t *iter_y;

    iarray_iter_read_block_new(ctx, c_y, &iter_y, bshape);

    for (iarray_iter_read_block_init(iter_y); !iarray_iter_read_block_finished(iter_y); iarray_iter_read_block_next(iter_y)) {
        iarray_iter_read_block_value_t value;
        iarray_iter_read_block_value(iter_y, &value);

        uint64_t bsize = 1;
        for (int i = 0; i < c_y->dtshape->ndim; ++i) {
            bsize *= value.block_shape[i];
        }

        for (int i = 0; i < bsize; ++i) {
            //printf("%f\n", ((double *) value.pointer)[i]);
        }
    }

    iarray_iter_read_block_free(iter_y);

    iarray_iter_read_block_t *iter_z;

    iarray_iter_read_block_new(ctx, c_z, &iter_z, bshape);

    for (iarray_iter_read_block_init(iter_z); !iarray_iter_read_block_finished(iter_z); iarray_iter_read_block_next(iter_z)) {
        iarray_iter_read_block_value_t value;
        iarray_iter_read_block_value(iter_z, &value);

        uint64_t bsize = 1;
        for (int i = 0; i < c_z->dtshape->ndim; ++i) {
            bsize *= value.block_shape[i];
        }

        for (int i = 0; i < bsize; ++i) {
            printf("%f\n", ((double *) value.pointer)[i]);
        }
    }

    iarray_iter_read_block_free(iter_z);
    
    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);
    iarray_container_free(ctx, &c_z);

    iarray_context_free(&ctx);
    return 0;
}
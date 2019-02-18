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

int main()
{
    ina_stopwatch_t *w;
    double elapsed, elapsed_view;
    INA_STOPWATCH_NEW(-1, -1, &w);

    int dtype = IARRAY_DATA_TYPE_FLOAT;

    uint64_t shape_x[] = {10, 10, 10};
    uint64_t pshape_x[] = {2, 2, 2};
    uint8_t ndim_x = 3;

    uint64_t pshape_y[] = {2, 1, 2};
    uint64_t pshape_z[] = {2, 1, 2};

    uint64_t shape_mul[] = {4, 4};
    uint64_t pshape_mul[] = {2, 2};
    uint8_t ndim_mul = 2;

    int64_t start[] = {1, 3, 3};
    int64_t stop[] = {5, 4, 7};

    uint64_t bshape[] = {2, 2};
    iarray_context_t *ctx;

    iarray_config_t config = IARRAY_CONFIG_DEFAULTS;
    config.compression_level = 5;
    config.compression_codec = 1;
    config.max_num_threads = 2;

    INA_MUST_SUCCEED(iarray_context_new(&config, &ctx));

    iarray_dtshape_t dtshape_x;
    dtshape_x.dtype = dtype;
    dtshape_x.ndim = ndim_x;
    uint64_t size_x = 1;
    for (int i = 0; i < dtshape_x.ndim; ++i) {
        dtshape_x.shape[i] = shape_x[i];
        dtshape_x.pshape[i] = pshape_x[i];
        size_x *= shape_x[i];
    }

    iarray_container_t *c_x;
    INA_MUST_SUCCEED(iarray_arange(ctx, &dtshape_x, 0, size_x, 1, NULL, 0, &c_x));

    INA_STOPWATCH_START(w);
    iarray_container_t *c_y;
    INA_MUST_SUCCEED(iarray_get_slice(ctx, c_x, start, stop, pshape_y, NULL, 0, false, &c_y));
    INA_MUST_SUCCEED(iarray_squeeze(ctx, c_y));
    INA_STOPWATCH_STOP(w);

    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed));
    printf("Time for get_slice: %f\n", elapsed);

    INA_STOPWATCH_START(w);
    iarray_container_t *c_z;
    INA_MUST_SUCCEED(iarray_get_slice(ctx, c_x, start, stop, pshape_z, NULL, 0, true, &c_z));
    iarray_squeeze(ctx, c_z);
    INA_STOPWATCH_STOP(w);

    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_view));
    printf("Time for get_slice (view): %f\n", elapsed_view);

    printf("Speed-up: %f\n", elapsed / elapsed_view);

    iarray_iter_read_block_t *iter_y;
    iarray_iter_read_block_t *iter_z;

    iarray_iter_read_block_new(ctx, c_y, &iter_y, bshape);
    iarray_iter_read_block_new(ctx, c_z, &iter_z, bshape);

    for (iarray_iter_read_block_init(iter_y),
        iarray_iter_read_block_init(iter_z);
         !iarray_iter_read_block_finished(iter_y);
         iarray_iter_read_block_next(iter_y),
         iarray_iter_read_block_next(iter_z)) {
        iarray_iter_read_block_value_t value_y;
        iarray_iter_read_block_value(iter_y, &value_y);
        iarray_iter_read_block_value_t value_z;
        iarray_iter_read_block_value(iter_z, &value_z);

        uint64_t bsize = 1;
        for (int i = 0; i < c_y->dtshape->ndim; ++i) {
            bsize *= value_y.block_shape[i];
        }

        for (uint64_t i = 0; i < bsize; ++i) {
            switch (dtype) {
                case IARRAY_DATA_TYPE_DOUBLE:
                    INA_TEST_ASSERT_EQUAL_FLOATING(((double *) value_y.pointer)[i], ((double *) value_z.pointer)[i]);
                    break;
                case IARRAY_DATA_TYPE_FLOAT:
                    INA_TEST_ASSERT_EQUAL_FLOATING(((float *) value_y.pointer)[i], ((float *) value_z.pointer)[i]);
                    break;
            }
        }
    }

    iarray_iter_read_block_free(iter_y);
    iarray_iter_read_block_free(iter_z);

    iarray_dtshape_t dtshape_mul;

    dtshape_mul.dtype = dtype;
    dtshape_mul.ndim = ndim_mul;
    for (int i = 0; i < dtshape_mul.ndim; ++i) {
        dtshape_mul.shape[i] = shape_mul[i];
        dtshape_mul.pshape[i] = pshape_mul[i];
    }

    iarray_container_t *c_mul;
    iarray_container_t *c_mul_view;

    INA_MUST_SUCCEED(iarray_container_new(ctx, &dtshape_mul, NULL, 0, &c_mul));
    INA_MUST_SUCCEED(iarray_container_new(ctx, &dtshape_mul, NULL, 0, &c_mul_view));

    INA_STOPWATCH_START(w);
    INA_MUST_SUCCEED(iarray_linalg_matmul(ctx, c_y, c_y, c_mul, bshape, bshape, IARRAY_OPERATOR_GENERAL));
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed));
    printf("Time for matrix-matrix mult: %f\n", elapsed);

    INA_STOPWATCH_START(w);
    INA_MUST_SUCCEED(iarray_linalg_matmul(ctx, c_z, c_z, c_mul_view, bshape, bshape, IARRAY_OPERATOR_GENERAL));
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_view));
    printf("Time for matrix-matrix mult (view): %f\n", elapsed_view);

    printf("Speed-up: %f\n", elapsed / elapsed_view);

    iarray_iter_read_block_t *iter_mul;
    iarray_iter_read_block_t *iter_mul_view;

    iarray_iter_read_block_new(ctx, c_mul, &iter_mul, bshape);
    iarray_iter_read_block_new(ctx, c_mul_view, &iter_mul_view, bshape);

    for (iarray_iter_read_block_init(iter_mul),
             iarray_iter_read_block_init(iter_mul_view);
         !iarray_iter_read_block_finished(iter_mul);
         iarray_iter_read_block_next(iter_mul),
             iarray_iter_read_block_next(iter_mul_view)) {
        iarray_iter_read_block_value_t value_mul;
        iarray_iter_read_block_value(iter_mul, &value_mul);
        iarray_iter_read_block_value_t value_mul_view;
        iarray_iter_read_block_value(iter_mul_view, &value_mul_view);

        uint64_t bsize = 1;
        for (int i = 0; i < c_mul->dtshape->ndim; ++i) {
            bsize *= value_mul.block_shape[i];
        }

        for (uint64_t i = 0; i < bsize; ++i) {
            switch (dtype) {
                case IARRAY_DATA_TYPE_DOUBLE:
                    INA_TEST_ASSERT_EQUAL_FLOATING(((double *) value_mul.pointer)[i], ((double *) value_mul_view.pointer)[i]);
                    break;
                case IARRAY_DATA_TYPE_FLOAT:
                    INA_TEST_ASSERT_EQUAL_FLOATING(((float *) value_mul.pointer)[i], ((float *) value_mul_view.pointer)[i]);
                    break;
            }
        }
    }

    iarray_iter_read_block_free(iter_mul);
    iarray_iter_read_block_free(iter_mul_view);

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);
    iarray_container_free(ctx, &c_z);
    iarray_container_free(ctx, &c_mul);
    iarray_container_free(ctx, &c_mul_view);

    iarray_context_free(&ctx);

    INA_STOPWATCH_FREE(&w);

    return 0;
}

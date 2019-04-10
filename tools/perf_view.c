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

int main(int argc, char *argv[])
{
    ina_stopwatch_t *w;
    double elapsed, elapsed_view;
    INA_STOPWATCH_NEW(-1, -1, &w);

    int dtype = IARRAY_DATA_TYPE_FLOAT;
    int typesize = sizeof(float);
    if ((argc > 1) && (strcmp(argv[1], "double") == 0)) {
        dtype = IARRAY_DATA_TYPE_DOUBLE;
        typesize = sizeof(double);
    }

    int64_t shape_x[] = {10, 10, 10};
    int64_t pshape_x[] = {2, 2, 2};
    int8_t ndim_x = 3;

    int64_t pshape_y[] = {2, 1, 2};
    int64_t pshape_z[] = {2, 1, 2};

    int64_t shape_mul[] = {4, 4};
    int64_t pshape_mul[] = {2, 2};
    int8_t ndim_mul = 2;

    int64_t start[] = {1, 3, 3};
    int64_t stop[] = {5, 4, 7};

    int64_t bshape[] = {2, 2};
    iarray_context_t *ctx;

    iarray_config_t config = IARRAY_CONFIG_DEFAULTS;
    config.compression_level = 5;
    config.compression_codec = 1;
    config.max_num_threads = 2;

    INA_MUST_SUCCEED(iarray_context_new(&config, &ctx));

    iarray_dtshape_t dtshape_x;
    dtshape_x.dtype = dtype;
    dtshape_x.ndim = ndim_x;
    size_t size_x = 1;
    for (int i = 0; i < dtshape_x.ndim; ++i) {
        dtshape_x.shape[i] = shape_x[i];
        dtshape_x.pshape[i] = pshape_x[i];
        size_x *= shape_x[i];
    }

    iarray_container_t *c_x;
    INA_MUST_SUCCEED(iarray_arange(ctx, &dtshape_x, 0., (double)size_x, 1., NULL, 0, &c_x));

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

    iarray_iter_read_block2_t *iter_y;
    iarray_iter_read_block2_t *iter_z;

    iarray_iter_read_block2_value_t value_y;
    iarray_iter_read_block2_value_t value_z;

    iarray_iter_read_block_new(ctx, &iter_y, c_y, bshape, &value_y);
    iarray_iter_read_block_new(ctx, &iter_z, c_z, bshape, &value_z);

    while (iarray_iter_read_block_has_next(iter_y)) {
        iarray_iter_read_block_next(iter_y);
        iarray_iter_read_block_next(iter_z);
        
        for (int64_t i = 0; i < value_y.block_size; ++i) {
            switch (dtype) {
                case IARRAY_DATA_TYPE_DOUBLE:
                    INA_TEST_ASSERT_EQUAL_FLOATING(((double *) value_y.pointer)[i], ((double *) value_z.pointer)[i]);
                    break;
                case IARRAY_DATA_TYPE_FLOAT:
                    INA_TEST_ASSERT_EQUAL_FLOATING(((float *) value_y.pointer)[i], ((float *) value_z.pointer)[i]);
                    break;
                default:
                    return INA_ERR_EXCEEDED;
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

    iarray_iter_read_t *iter_mul;
    iarray_iter_read_t *iter_mul_view;

    iarray_iter_read_new(ctx, c_mul, &iter_mul);
    iarray_iter_read_new(ctx, c_mul_view, &iter_mul_view);

    for (iarray_iter_read_init(iter_mul),
             iarray_iter_read_init(iter_mul_view);
         !iarray_iter_read_finished(iter_mul);
         iarray_iter_read_next(iter_mul),
             iarray_iter_read_next(iter_mul_view)) {
        iarray_iter_read_value_t value_mul;
        iarray_iter_read_value(iter_mul, &value_mul);
        iarray_iter_read_value_t value_mul_view;
        iarray_iter_read_value(iter_mul_view, &value_mul_view);

        switch (dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                INA_TEST_ASSERT_EQUAL_FLOATING(((double *) value_mul.pointer)[0], ((double *) value_mul_view.pointer)[0]);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                INA_TEST_ASSERT_EQUAL_FLOATING(((float *) value_mul.pointer)[0], ((float *) value_mul_view.pointer)[0]);
                break;
            default:
                return INA_ERR_EXCEEDED;
        }
    }

    iarray_iter_read_free(iter_mul);
    iarray_iter_read_free(iter_mul_view);

    uint64_t size = 1;
    for (int i = 0; i < c_y->dtshape->ndim; ++i) {
        size *= c_y->dtshape->shape[i];
    }

    uint8_t *buffer_y = ina_mem_alloc(size * typesize);
    INA_MUST_SUCCEED(iarray_to_buffer(ctx, c_y, buffer_y, size * typesize));

    uint8_t *buffer_z = ina_mem_alloc(size * typesize);
    INA_MUST_SUCCEED(iarray_to_buffer(ctx, c_z, buffer_z, size * typesize));

    for (uint64_t i = 0; i < size; ++i) {
        switch (dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                INA_TEST_ASSERT_EQUAL_FLOATING(((double *) buffer_y)[i], ((double *) buffer_z)[i]);

                break;
            case IARRAY_DATA_TYPE_FLOAT:
                INA_TEST_ASSERT_EQUAL_FLOATING(((float *) buffer_y)[i], ((float *) buffer_z)[i]);
                break;
            default:
                return INA_ERR_EXCEEDED;
        }
    }

    ina_mem_free(buffer_y);
    ina_mem_free(buffer_z);

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);
    iarray_container_free(ctx, &c_z);
    iarray_container_free(ctx, &c_mul);
    iarray_container_free(ctx, &c_mul_view);

    iarray_context_free(&ctx);

    INA_STOPWATCH_FREE(&w);

    return 0;
}

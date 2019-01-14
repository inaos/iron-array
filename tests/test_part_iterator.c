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

#include <tests/iarray_test.h>

static ina_rc_t test_part_iterator(iarray_context_t *ctx, iarray_data_type_t dtype, size_t type_size, uint8_t ndim,
                                    const uint64_t *shape, const uint64_t *pshape) {

    // Create dtshape
    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    uint64_t size = 1;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
        xdtshape.pshape[i] = pshape[i];
        size *= shape[i];
    }

    iarray_container_t *c_x;

    iarray_container_new(ctx, &xdtshape, NULL, 0, &c_x);

    // Start Iterator
    iarray_iter_part_t *I;
    iarray_iter_part_new(ctx, c_x, &I);

    for (iarray_iter_part_init(ctx, I); !iarray_iter_part_finished(ctx, I); iarray_iter_part_next(ctx, I)) {

        iarray_itr_part_value_t val;
        iarray_iter_part_value(ctx, I, &val);

        uint64_t part_size = 1;
        for (int i = 0; i < ndim; ++i) {
            part_size *= val.part_shape[i];
        }

        uint8_t *data = malloc(part_size * type_size);

        if(dtype == IARRAY_DATA_TYPE_DOUBLE) {
            for (uint64_t i = 0; i < part_size; ++i) {
                ( (double *)data)[i] = (double) val.nelem * part_size + i;
            }
        } else {
            for (uint64_t i = 0; i < part_size; ++i) {
                ( (float *)data)[i] = (float) val.nelem * part_size + i;
            }
        }
        memcpy(val.pointer, &data[0], part_size * type_size);
        free(data);
    }

    iarray_iter_part_free(ctx, I);

    // Testing

    // Start Iterator
    iarray_iter_block_read_t *I2;
    iarray_iter_block_read_new(ctx, c_x, &I2, pshape);

    for (iarray_iter_block_read_init(ctx, I2);
         !iarray_iter_block_read_finished(ctx, I2);
         iarray_iter_block_read_next(ctx, I2)) {

        iarray_iter_block_read_value_t val;
        iarray_iter_block_read_value(ctx, I2, &val);

        uint64_t block_size = 1;
        for (int i = 0; i < ndim; ++i) {
            block_size *= val.block_shape[i];
        }

        if(dtype == IARRAY_DATA_TYPE_DOUBLE) {
            for (uint64_t i = 0; i < block_size; ++i) {
                INA_TEST_ASSERT_EQUAL_FLOATING(((double *)val.pointer)[i], (double) val.nelem * block_size + i);
            }
        } else {
            for (uint64_t i = 0; i < block_size; ++i) {
                INA_TEST_ASSERT_EQUAL_FLOATING(((float *)val.pointer)[i], (float) val.nelem * block_size + i);
            }
        }
    }

    iarray_iter_block_read_free(ctx, I2);

    // Free
    iarray_container_free(ctx, &c_x);

    return INA_SUCCESS;
}

INA_TEST_DATA(part_iterator) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(part_iterator) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.flags = IARRAY_EXPR_EVAL_CHUNK;

    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(part_iterator) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(part_iterator, double_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    uint8_t ndim = 2;
    uint64_t shape[] = {10, 10};
    uint64_t pshape[] = {2, 2};

    INA_TEST_ASSERT_SUCCEED(test_part_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}


INA_TEST_FIXTURE(part_iterator, float_3) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    uint8_t ndim = 3;
    uint64_t shape[] = {120, 131, 155};
    uint64_t pshape[] = {23, 32, 35};

    INA_TEST_ASSERT_SUCCEED(test_part_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}


INA_TEST_FIXTURE(part_iterator, double_4) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    uint8_t ndim = 4;
    uint64_t shape[] = {80, 64, 80, 99};
    uint64_t pshape[] = {11, 8, 12, 21};

    INA_TEST_ASSERT_SUCCEED(test_part_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}

INA_TEST_FIXTURE(part_iterator, float_5) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    uint8_t ndim = 5;
    uint64_t shape[] = {40, 26, 35, 23, 21};
    uint64_t pshape[] = {5, 8, 10, 7, 9};

    INA_TEST_ASSERT_SUCCEED(test_part_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}

INA_TEST_FIXTURE(part_iterator, double_6) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    uint8_t ndim = 6;
    uint64_t shape[] = {12, 13, 21, 19, 13, 15};
    uint64_t pshape[] = {5, 4, 7, 3, 4, 12};

    INA_TEST_ASSERT_SUCCEED(test_part_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}

INA_TEST_FIXTURE(part_iterator, float_7) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    uint8_t ndim = 7;
    uint64_t shape[] = {10, 8, 6, 7, 13, 9, 10};
    uint64_t pshape[] = {2, 3, 1, 3, 2, 4, 5};

    INA_TEST_ASSERT_SUCCEED(test_part_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}

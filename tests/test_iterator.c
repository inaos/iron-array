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


static ina_rc_t test_iterator(iarray_context_t *ctx, iarray_data_type_t dtype, int32_t type_size, int8_t ndim,
                              const int64_t *shape, const int64_t *pshape) {

    // Create dtshape
    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
        xdtshape.pshape[i] = pshape[i];
    }

    iarray_container_t *c_x;

    INA_TEST_ASSERT_SUCCEED(iarray_container_new(ctx, &xdtshape, NULL, 0, &c_x));

    // Start Iterator
    iarray_iter_write_t *I;
    iarray_iter_write_new(ctx, c_x, &I);

    for (iarray_iter_write_init(I); !iarray_iter_write_finished(I); iarray_iter_write_next(I)) {

        iarray_iter_write_value_t val;
        iarray_iter_write_value(I, &val);

        if(dtype == IARRAY_DATA_TYPE_DOUBLE) {
            double value = (double) val.nelem;
            memcpy(val.pointer, &value, type_size);
        } else {
            float value = (float) val.nelem;
            memcpy(val.pointer, &value, type_size);
        }
    }

    iarray_iter_write_free(I);


    // Assert iterator reading it
    iarray_iter_read_t *I2;
    iarray_iter_read_new(ctx, c_x, &I2);

    for (iarray_iter_read_init(I2); !iarray_iter_read_finished(I2); iarray_iter_read_next(I2)) {

        iarray_iter_read_value_t val;
        iarray_iter_read_value(I2, &val);

        if(dtype == IARRAY_DATA_TYPE_DOUBLE) {
            double value = (double) val.nelem;
            INA_TEST_ASSERT_EQUAL_FLOATING(value, ((double *) val.pointer)[0]);
        } else {
            float value = (float) val.nelem;
            INA_TEST_ASSERT_EQUAL_FLOATING(value, ((float *) val.pointer)[0]);
        }
    }

    iarray_iter_read_free(I2);

    iarray_container_free(ctx, &c_x);
    return INA_SUCCESS;
}

INA_TEST_DATA(iterator) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(iterator) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.eval_flags = IARRAY_EXPR_EVAL_CHUNK;

    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(iterator) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(iterator, double_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 2;
    int64_t shape[] = {4, 6};
    int64_t pshape[] = {2, 3};

    INA_TEST_ASSERT_SUCCEED(test_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}


INA_TEST_FIXTURE(iterator, float_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 2;
    int64_t shape[] = {445, 321};
    int64_t pshape[] = {21, 17};

    INA_TEST_ASSERT_SUCCEED(test_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}


INA_TEST_FIXTURE(iterator, double_3) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 3;
    int64_t shape[] = {20, 53, 17};
    int64_t pshape[] = {12, 12, 2};

    INA_TEST_ASSERT_SUCCEED(test_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}


INA_TEST_FIXTURE(iterator, float_4) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 4;
    int64_t shape[] = {15, 18, 14, 13};
    int64_t pshape[] = {12, 12, 2, 5};

    INA_TEST_ASSERT_SUCCEED(test_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}

INA_TEST_FIXTURE(iterator, double_5) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 5;
    int64_t shape[] = {15, 18, 17, 13, 13};
    int64_t pshape[] = {7, 12, 2, 3, 6};

    INA_TEST_ASSERT_SUCCEED(test_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}

INA_TEST_FIXTURE(iterator, float_6) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 6;
    int64_t shape[] = {5, 7, 8, 9, 6, 5};
    int64_t pshape[] = {2, 5, 3, 4, 3, 2};

    INA_TEST_ASSERT_SUCCEED(test_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}

INA_TEST_FIXTURE(iterator, double_7) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 7;
    int64_t shape[] = {5, 7, 8, 9, 6, 5, 4};
    int64_t pshape[] = {2, 5, 3, 4, 3, 2, 2};

    INA_TEST_ASSERT_SUCCEED(test_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}

INA_TEST_FIXTURE(iterator, float_8) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 8;
    int64_t shape[] = {5, 7, 8, 9, 6, 5, 3, 5};
    int64_t pshape[] = {2, 5, 3, 4, 3, 2, 2, 2};

    INA_TEST_ASSERT_SUCCEED(test_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}

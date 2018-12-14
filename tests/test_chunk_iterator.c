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

static ina_rc_t test_chunk_iterator(iarray_context_t *ctx, iarray_data_type_t dtype, size_t type_size, uint8_t ndim,
                                    const uint64_t *shape, const uint64_t *pshape) {

    // Create dtshape
    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
        xdtshape.partshape[i] = pshape[i];
    }

    iarray_container_t *c_x;

    iarray_container_new(ctx, &xdtshape, NULL, 0, &c_x);

    // Start Iterator
    iarray_itr_chunk_t *I;
    iarray_itr_chunk_new(ctx, c_x, &I);

    for (iarray_itr_chunk_init(I); !iarray_itr_chunk_finished(I); iarray_itr_chunk_next(I)) {

        iarray_itr_chunk_value_t val;
        iarray_itr_chunk_value(I, &val);

        printf("Nchunk: %llu\n", val.nelem);
    }

    iarray_itr_chunk_free(ctx, I);

    // Free
    iarray_container_free(ctx, &c_x);
    return INA_SUCCESS;
}

INA_TEST_DATA(chunk_iterator) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(chunk_iterator) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.flags = IARRAY_EXPR_EVAL_CHUNK;

    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(chunk_iterator) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(chunk_iterator, double_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    uint8_t ndim = 2;
    uint64_t shape[] = {100, 100};
    uint64_t pshape[] = {20, 20};

    INA_TEST_ASSERT_SUCCEED(test_chunk_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}

INA_TEST_FIXTURE_SKIP(chunk_iterator, float_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    uint8_t ndim = 2;
    uint64_t shape[] = {445, 321};
    uint64_t pshape[] = {21, 17};

    INA_TEST_ASSERT_SUCCEED(test_chunk_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}

INA_TEST_FIXTURE_SKIP(chunk_iterator, double_5) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    uint8_t ndim = 5;
    uint64_t shape[] = {20, 25, 27, 41, 46};
    uint64_t pshape[] = {12, 24, 19, 31, 13};

    INA_TEST_ASSERT_SUCCEED(test_chunk_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}

INA_TEST_FIXTURE_SKIP(chunk_iterator, float_7) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    uint8_t ndim = 7;
    uint64_t shape[] = {10, 12, 8, 9, 13, 7, 7};
    uint64_t pshape[] = {2, 5, 3, 4, 3, 3, 3};

    INA_TEST_ASSERT_SUCCEED(test_chunk_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}
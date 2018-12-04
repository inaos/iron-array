/*
 * Copyright (C) 2018  Francesc Alted
 * Copyright (C) 2018  Aleix Alcacer
 */

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
#include <iarray_private.h>

#include <tests/iarray_test.h>

static ina_rc_t test_iterator(iarray_context_t *ctx, iarray_data_type_t dtype, size_t type_size, uint8_t ndim,
                                     const uint64_t *shape, const uint64_t *pshape) {

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

    iarray_itr_t *I;
    iarray_itr_new(c_x, &I);

    for (I->init(I); !I->finished(I); I->next(I)) {

        if(dtype == IARRAY_DATA_TYPE_DOUBLE) {
            double value = (double) I->nelem;
            memcpy(I->pointer, &value, type_size);
        } else {
            float value = (float) I->nelem;
            memcpy(I->pointer, &value, type_size);
        }
    }

    iarray_itr_free(I);

    // Assert iterator values

    uint64_t bufsize = 1;
    for (int j = 0; j < ndim; ++j) {
        bufsize *= xdtshape.shape[j];
    }

    uint8_t *bufdest = ina_mem_alloc(bufsize * type_size);
    iarray_to_buffer(ctx, c_x, bufdest, bufsize);


    if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
        for (uint64_t k = 1; k < bufsize; ++k) {
            INA_TEST_ASSERT_EQUAL_FLOATING(((double *)bufdest)[k-1] + 1, ((double *)bufdest)[k]);
        }
    } else {
        for (uint64_t k = 1; k < bufsize; ++k) {
            INA_TEST_ASSERT_EQUAL_FLOATING(((float *)bufdest)[k-1] + 1, ((float *)bufdest)[k]);
        }
    }

    // Free

    ina_mem_free(bufdest);
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
    cfg.flags = IARRAY_EXPR_EVAL_CHUNK;

    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(iterator) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(iterator, double_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    uint8_t ndim = 2;
    uint64_t shape[] = {125, 157};
    uint64_t pshape[] = {12, 13};

    INA_TEST_ASSERT_SUCCEED(test_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}

INA_TEST_FIXTURE(iterator, float_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    uint8_t ndim = 2;
    uint64_t shape[] = {445, 321};
    uint64_t pshape[] = {21, 17};

    INA_TEST_ASSERT_SUCCEED(test_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}

INA_TEST_FIXTURE(iterator, double_5) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    uint8_t ndim = 5;
    uint64_t shape[] = {20, 25, 27, 41, 46};
    uint64_t pshape[] = {12, 24, 19, 31, 13};

    INA_TEST_ASSERT_SUCCEED(test_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}

INA_TEST_FIXTURE(iterator, float_7) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    uint8_t ndim = 7;
    uint64_t shape[] = {10, 12, 8, 9, 13, 7, 7};
    uint64_t pshape[] = {2, 5, 3, 4, 3, 3, 3};

    INA_TEST_ASSERT_SUCCEED(test_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}
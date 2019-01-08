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

static ina_rc_t test_read_iterator(iarray_context_t *ctx, iarray_data_type_t dtype, size_t type_size, uint8_t ndim,
                                    const uint64_t *shape, const uint64_t *pshape, uint64_t *blockshape) {

    // Create dtshape
    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    uint64_t contsize = 1;
    for (int i = 0; i < ndim; ++i) {
        contsize *= shape[i];
        xdtshape.shape[i] = shape[i];
        xdtshape.pshape[i] = pshape[i];
    }

    iarray_container_t *c_x;

    iarray_arange(ctx, &xdtshape, 0, contsize, 1, NULL, 0, &c_x);

    // Start Iterator
    iarray_itr_read_t *I;
    iarray_itr_read_new(ctx, c_x, &I, blockshape);

    for (iarray_itr_read_init(I); !iarray_itr_read_finished(I); iarray_itr_read_next(I)) {
        iarray_itr_read_value_t val;
        iarray_itr_read_value(I, &val);
        uint64_t partsize = 1;
        for (int i = 0; i < ndim; ++i) {
            partsize *= val.shape[i];
        }

        for (uint64_t i = 0; i < partsize; ++i) {
            printf("%llu: %f\n", i, ((double *)val.pointer)[i]);
        }
    }

    iarray_itr_read_free(ctx, I);

    // Free
    iarray_container_free(ctx, &c_x);

    return INA_SUCCESS;
}

INA_TEST_DATA(read_iterator) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(read_iterator) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.flags = IARRAY_EXPR_EVAL_CHUNK;

    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(read_iterator) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(read_iterator, double_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    uint8_t ndim = 2;
    uint64_t shape[] = {10, 10};
    uint64_t pshape[] = {4, 3};
    uint64_t blockshape[] = {3, 3};

    INA_TEST_ASSERT_SUCCEED(test_read_iterator(data->ctx, dtype, type_size, ndim, shape, pshape, blockshape));
}

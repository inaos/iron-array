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

    // Iterator

    iarray_itr_t *I;
    iarray_itr_new(c_x, &I);


    for (I->init(I); !I->finished(I); I->next(I)) {
        double cont = 0;
        uint64_t inc = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            cont += I->index[i] * inc;
            inc *= shape[i];
        }
        //printf("%f\n", cont);
        memcpy(I->pointer, &cont, sizeof(double));
    }

    iarray_itr_free(I);

    uint64_t bufsize = 1;
    for (int j = 0; j < ndim; ++j) {
        bufsize *= xdtshape.shape[j];
    }
    double *bufdest = (double *) malloc(bufsize * type_size);
    iarray_to_buffer(ctx, c_x, bufdest, bufsize);

    for (uint64_t k = 0; k < bufsize; ++k) {
        printf("%f\n", bufdest[k]);
    }

    free(bufdest);
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

INA_TEST_FIXTURE(iterator, double_data) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    uint8_t ndim = 2;
    uint64_t shape[] = {4, 18};
    uint64_t pshape[] = {2, 3};

    INA_TEST_ASSERT_SUCCEED(test_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}
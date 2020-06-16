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

static ina_rc_t test_rewrite_cont(iarray_context_t *ctx, iarray_data_type_t dtype,
                                  int32_t type_size, int8_t ndim, const int64_t *shape,
                                  const int64_t *pshape, const int64_t *bshape, const int64_t *blockshape) {
    INA_UNUSED(type_size);
    // Create dtshape
    iarray_dtshape_t xdtshape;
    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    int64_t size = 1;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
        size *= shape[i];
    }

    iarray_storage_t xstore;
    xstore.backend = pshape ? IARRAY_STORAGE_BLOSC : IARRAY_STORAGE_PLAINBUFFER;
    xstore.enforce_frame = false;
    xstore.filename = NULL;
    if (pshape != NULL) {
        for (int i = 0; i < ndim; ++i) {
            xstore.chunkshape[i] = pshape[i];
            xstore.blockshape[i] = bshape[i];
        }
    }
    iarray_container_t *c_x;

    INA_TEST_ASSERT_SUCCEED(iarray_container_new(ctx, &xdtshape, &xstore, 0, &c_x));

    // Start Iterator
    iarray_iter_write_block_t *I;
    iarray_iter_write_block_value_t val;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_write_block_new(ctx, &I, c_x, blockshape, &val, false));

    while (INA_SUCCEED(iarray_iter_write_block_has_next(I))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_write_block_next(I, NULL, 0));

        int64_t nelem = 0;
        int64_t inc = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            nelem += val.elem_index[i] * inc;
            inc *= c_x->dtshape->shape[i];
        }
        if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
            for (int64_t i = 0; i < val.block_size; ++i) {
                ((double *) val.block_pointer)[i] = (double) nelem + i;
            }
        } else {
            for (int64_t i = 0; i < val.block_size; ++i) {
                ((float *) val.block_pointer)[i] = (float) nelem + i;
            }
        }
    }

    iarray_iter_write_block_free(&I);

    iarray_storage_t ystore;
    ystore.backend = IARRAY_STORAGE_PLAINBUFFER;
    ystore.enforce_frame = false;
    ystore.filename = NULL;


    // Start Iterator
    ina_rc_t err = iarray_iter_write_block_new(ctx, &I, c_x, blockshape, &val, false);
    if (c_x->catarr->storage == CATERVA_STORAGE_BLOSC) {
        if (err != 0) {
            return INA_SUCCESS;
        }
    }
    while (INA_SUCCEED(iarray_iter_write_block_has_next(I))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_write_block_next(I, NULL, 0));

        int64_t nelem = 0;
        int64_t inc = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            nelem += val.elem_index[i] * inc;
            inc *= c_x->dtshape->shape[i];
        }
        if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
            for (int64_t i = 0; i < val.block_size; ++i) {
                ((double *) val.block_pointer)[i] = 0;
            }
        } else {
            for (int64_t i = 0; i < val.block_size; ++i) {
                ((float *) val.block_pointer)[i] = 0;
            }
        }
    }

    // Start Read Iterator
    iarray_iter_read_t *itr_read;
    iarray_iter_read_value_t val_read;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_new(ctx, &itr_read, c_x, &val_read));

    while (INA_SUCCEED(iarray_iter_read_has_next(itr_read))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_read_next(itr_read));
        if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
            INA_TEST_ASSERT_EQUAL_FLOATING(((double *) val.block_pointer)[0], 0);
        } else {
            INA_TEST_ASSERT_EQUAL_FLOATING(((float *) val.block_pointer)[0], 0);
        }
    }
    return INA_SUCCESS;
}


INA_TEST_DATA(rewrite_cont) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(rewrite_cont) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(rewrite_cont) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}


INA_TEST_FIXTURE(rewrite_cont, 2_d_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 2;
    int64_t shape[] = {5, 5};
    int64_t *pshape = NULL;
    int64_t *bshape = NULL;
    int64_t blockshape[] = {3, 2};

    INA_TEST_ASSERT_SUCCEED(test_rewrite_cont(data->ctx, dtype, type_size, ndim, shape, pshape, bshape,
                                              blockshape));
}

INA_TEST_FIXTURE(rewrite_cont, 3_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 3;
    int64_t shape[] = {120, 131, 155};
    int64_t pshape[] = {23, 32, 35};
    int64_t bshape[] = {7, 7, 5};
    int64_t *blockshape = pshape;

    INA_TEST_ASSERT_SUCCEED(test_rewrite_cont(data->ctx, dtype, type_size, ndim, shape, pshape, bshape,
                                              blockshape));
}

INA_TEST_FIXTURE(rewrite_cont, 4_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 4;
    int64_t shape[] = {30, 64, 50, 43};
    int64_t pshape[] = {11, 8, 12, 21};
    int64_t bshape[] = {4, 3, 3, 4};
    int64_t *blockshape = pshape;

    INA_TEST_ASSERT_SUCCEED(test_rewrite_cont(data->ctx, dtype, type_size, ndim, shape, pshape, bshape,
                                              blockshape));
}

INA_TEST_FIXTURE(rewrite_cont, 5_f_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 5;
    int64_t shape[] = {40, 26, 35, 23, 21};
    int64_t *pshape = NULL;
    int64_t *bshape = NULL;
    int64_t blockshape[] = {12, 12, 12, 12, 12};

    INA_TEST_ASSERT_SUCCEED(test_rewrite_cont(data->ctx, dtype, type_size, ndim, shape, pshape, bshape,
                                              blockshape));
}

INA_TEST_FIXTURE(rewrite_cont, 6_d_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 6;
    int64_t shape[] = {12, 13, 21, 19, 13, 15};
    int64_t *pshape = NULL;
    int64_t *bshape = NULL;
    int64_t blockshape[] = {2, 3, 5, 4, 3, 2};

    INA_TEST_ASSERT_SUCCEED(test_rewrite_cont(data->ctx, dtype, type_size, ndim, shape, pshape, bshape,
                                              blockshape));
}

INA_TEST_FIXTURE(rewrite_cont, 7_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 7;
    int64_t shape[] = {10, 8, 6, 7, 13, 9, 10};
    int64_t pshape[] = {2, 3, 1, 3, 2, 4, 5};
    int64_t bshape[] = {2, 3, 1, 3, 2, 4, 5};
    int64_t *blockshape = pshape;

    INA_TEST_ASSERT_SUCCEED(test_rewrite_cont(data->ctx, dtype, type_size, ndim, shape, pshape, bshape,
                                              blockshape));
}


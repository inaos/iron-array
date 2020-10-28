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


static ina_rc_t test_iterator(iarray_context_t *ctx, iarray_data_type_t dtype, int32_t type_size, int8_t ndim,
                              const int64_t *shape, const int64_t *cshape, const int64_t *bshape) {

    // Create dtshape
    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
    }

    iarray_storage_t store;
    store.backend = cshape ? IARRAY_STORAGE_BLOSC : IARRAY_STORAGE_PLAINBUFFER;
    store.enforce_frame = false;
    store.filename = NULL;
    if (cshape != NULL) {
        for (int i = 0; i < ndim; ++i) {
            store.chunkshape[i] = cshape[i];
            store.blockshape[i] = bshape[i];
        }
    }

    iarray_container_t *c_trans;
    INA_TEST_ASSERT_SUCCEED(iarray_container_new(ctx, &xdtshape, &store, 0, &c_trans));

    // Start Iterator
    // Test write iterator
    iarray_iter_write_block_t *I;
    iarray_iter_write_block_value_t val;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_write_block_new(ctx, &I, c_trans,
                                                        c_trans->storage->chunkshape,
                                                        &val,
                                                        false));

    while (INA_SUCCEED(iarray_iter_write_block_has_next(I))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_write_block_next(I, NULL, 0));

        int64_t nelem = 0;
        int64_t inc = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            nelem += val.elem_index[i] * inc;
            inc *= c_trans->dtshape->shape[i];
        }
        if(dtype == IARRAY_DATA_TYPE_DOUBLE) {
            for (int64_t i = 0; i < val.block_size; ++i) {
                ((double *) val.block_pointer)[i] = (double) nelem + i;
            }
        } else {
            for (int64_t i = 0; i < val.block_size; ++i) {
                ((float *) val.block_pointer)[i] = (float) nelem  + i;
            }
        }
    }

    iarray_iter_write_block_free(&I);

    INA_TEST_ASSERT(ina_err_get_rc() == INA_RC_PACK(IARRAY_ERR_END_ITER, 0));


    iarray_container_t *c_x;
    INA_TEST_ASSERT_SUCCEED(iarray_linalg_transpose(ctx, c_trans, true, NULL, &c_x));


    iarray_storage_t ystorage;
    ystorage.backend = cshape ? IARRAY_STORAGE_BLOSC : IARRAY_STORAGE_PLAINBUFFER;
    ystorage.enforce_frame = false;
    ystorage.filename = NULL;
    for (int i = 0; i < ndim; ++i) {
        ystorage.chunkshape[i] = cshape ? cshape[ndim - 1 - i] : 0;
        ystorage.blockshape[i] = bshape ? bshape[ndim - 1 - i] : 0;
    }

    iarray_container_t *c_y;
    INA_TEST_ASSERT_SUCCEED(iarray_copy(ctx, c_x, false, &ystorage, 0, &c_y));


    // Assert iterator reading it
    iarray_iter_read_t *I2;
    iarray_iter_read_value_t val2;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_new(ctx, &I2, c_x, &val2));

    iarray_iter_read_t *I3;
    iarray_iter_read_value_t val3;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_new(ctx, &I3, c_y, &val3));

    while (INA_SUCCEED(iarray_iter_read_has_next(I2)) &&
           INA_SUCCEED(iarray_iter_read_has_next(I3))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_read_next(I2));
        INA_TEST_ASSERT_SUCCEED(iarray_iter_read_next(I3));

        if(dtype == IARRAY_DATA_TYPE_DOUBLE) {
            double value2 = ((double *) val2.elem_pointer)[0];
            double value3 = ((double *) val3.elem_pointer)[0];
            INA_TEST_ASSERT_EQUAL_FLOATING(value2, value3);
        } else {
            float value2 = ((float *) val2.elem_pointer)[0];
            float value3 = ((float *) val3.elem_pointer)[0];
            INA_TEST_ASSERT_EQUAL_FLOATING(value2, value3);
        }
    }

    iarray_iter_read_free(&I2);
    iarray_iter_read_free(&I3);
    INA_TEST_ASSERT(ina_err_get_rc() == INA_RC_PACK(IARRAY_ERR_END_ITER, 0));

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);
    iarray_container_free(ctx, &c_trans);
    return INA_SUCCESS;
}

INA_TEST_DATA(iterator_transpose) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(iterator_transpose) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(iterator_transpose) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}


INA_TEST_FIXTURE(iterator_transpose, 2_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 2;
    int64_t shape[] = {445, 321};
    int64_t cshape[] = {201, 17};
    int64_t bshape[] = {12, 8};

    INA_TEST_ASSERT_SUCCEED(test_iterator(data->ctx, dtype, type_size, ndim, shape, cshape, bshape));
}


INA_TEST_FIXTURE(iterator_transpose, 2_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 2;
    int64_t shape[] = {2000, 5033};
    int64_t cshape[] = {12, 2000};
    int64_t bshape[] = {12, 200};

    INA_TEST_ASSERT_SUCCEED(test_iterator(data->ctx, dtype, type_size, ndim, shape, cshape, bshape));
}


INA_TEST_FIXTURE(iterator_transpose, 2_f_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 2;
    int64_t shape[] = {523, 4816};
    int64_t *cshape = NULL;
    int64_t *bshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_iterator(data->ctx, dtype, type_size, ndim, shape, cshape, bshape));
}

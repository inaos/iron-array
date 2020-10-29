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

    int64_t blockshape[IARRAY_DIMENSION_MAX];
    for (int i = 0; i < ndim; ++i) {
        blockshape[i] = cshape ? cshape[i] : shape[i];
    }
    iarray_dtshape_t xdtshape;
    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    int64_t size = 1;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
        size *= shape[i];
    }

    iarray_storage_t xstorage;
    xstorage.backend = cshape ? IARRAY_STORAGE_BLOSC : IARRAY_STORAGE_PLAINBUFFER;
    xstorage.enforce_frame = false;
    xstorage.filename = NULL;
    for (int i = 0; i < ndim; ++i) {
        xstorage.chunkshape[i] = cshape ? cshape[i] : 0;
        xstorage.blockshape[i] = bshape ? bshape[i] : 0;
    }

    iarray_storage_t ystorage;
    ystorage.backend = cshape ? IARRAY_STORAGE_BLOSC : IARRAY_STORAGE_PLAINBUFFER;
    ystorage.enforce_frame = false;
    ystorage.filename = NULL;
    for (int i = 0; i < ndim; ++i) {
        ystorage.chunkshape[i] = cshape ? cshape[ndim - 1 - i] : 0;
        ystorage.blockshape[i] = bshape ? bshape[ndim - 1 - i] : 0;
    }

    iarray_container_t *c_x;

    INA_TEST_ASSERT_SUCCEED(iarray_container_new(ctx, &xdtshape, &xstorage, 0, &c_x));

    // Test write iterator
    iarray_iter_write_t *I;
    iarray_iter_write_value_t val;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_write_new(ctx, &I, c_x, &val));

    while (INA_SUCCEED(iarray_iter_write_has_next(I))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_write_next(I));

        int64_t nelem = 0;
        int64_t inc = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            nelem += val.elem_index[i] * inc;
            inc *= c_x->dtshape->shape[i];
        }
        if(dtype == IARRAY_DATA_TYPE_DOUBLE) {
            ((double *) val.elem_pointer)[0] = (double) nelem ;
        } else {
            ((float *) val.elem_pointer)[0] = (float) nelem;
        }
    }
    iarray_iter_write_free(&I);
    INA_TEST_ASSERT(ina_err_get_rc() == INA_RC_PACK(IARRAY_ERR_END_ITER, 0));

    iarray_container_t *c_trans;
    INA_TEST_ASSERT_SUCCEED(iarray_linalg_transpose(ctx, c_x, true, NULL, &c_trans));

    iarray_container_t *c_y;
    INA_TEST_ASSERT_SUCCEED(iarray_copy(ctx, c_trans, false, &ystorage, 0, &c_y));

    // Test read iterator
    iarray_iter_read_t *I2;
    iarray_iter_read_value_t val2;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_new(ctx, &I2, c_trans, &val2));

    iarray_iter_read_t *I3;
    iarray_iter_read_value_t val3;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_new(ctx, &I3, c_y, &val3));

    while (INA_SUCCEED(iarray_iter_read_has_next(I2)) && INA_SUCCEED(iarray_iter_read_has_next(I3))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_read_next(I2));
        INA_TEST_ASSERT_SUCCEED(iarray_iter_read_next(I3));

        switch (dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                INA_TEST_ASSERT_EQUAL_FLOATING(((double *) val2.elem_pointer)[0],
                                               ((double *) val3.elem_pointer)[0]);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                INA_TEST_ASSERT_EQUAL_FLOATING(((float *) val2.elem_pointer)[0],
                                               ((float *) val3.elem_pointer)[0]);
                break;
            default:
                return INA_ERR_EXCEEDED;
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

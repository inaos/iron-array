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

#include <src/iarray_private.h>
#include <libiarray/iarray.h>


static ina_rc_t test_iterator(iarray_context_t *ctx, iarray_data_type_t dtype, int32_t type_size, int8_t ndim,
                              const int64_t *shape, const int64_t *cshape, const int64_t *bshape, bool contiguous, char *urlpath) {

    // Create dtshape
    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
    }

    iarray_storage_t store;
    store.backend = cshape ? IARRAY_STORAGE_BLOSC : IARRAY_STORAGE_PLAINBUFFER;
    store.contiguous = contiguous;
    store.urlpath = urlpath;
    if (cshape != NULL) {
        for (int i = 0; i < ndim; ++i) {
            store.chunkshape[i] = cshape[i];
            store.blockshape[i] = bshape[i];
        }
    }
    blosc2_remove_urlpath(store.urlpath);

    iarray_container_t *c_x;
    INA_TEST_ASSERT_SUCCEED(iarray_empty(ctx, &xdtshape, &store, 0, &c_x));

    // Start Iterator
    iarray_iter_write_t *I;
    iarray_iter_write_value_t val;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_write_new(ctx, &I, c_x, &val));

    while (INA_SUCCEED(iarray_iter_write_has_next(I))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_write_next(I));

        if(dtype == IARRAY_DATA_TYPE_DOUBLE) {
            double value = (double) val.elem_flat_index;
            memcpy(val.elem_pointer, &value, type_size);
        } else {
            float value = (float) val.elem_flat_index;
            memcpy(val.elem_pointer, &value, type_size);
        }
    }

    iarray_iter_write_free(&I);
    INA_TEST_ASSERT(ina_err_get_rc() == INA_RC_PACK(IARRAY_ERR_END_ITER, 0));


    // Assert iterator reading it
    iarray_iter_read_t *I2;
    iarray_iter_read_value_t val2;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_new(ctx, &I2, c_x, &val2));

    while (INA_SUCCEED(iarray_iter_read_has_next(I2))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_read_next(I2));

        if(dtype == IARRAY_DATA_TYPE_DOUBLE) {
            double value = (double) val2.elem_flat_index;
            INA_TEST_ASSERT_EQUAL_FLOATING(value, ((double *) val2.elem_pointer)[0]);
        } else {
            float value = (float) val2.elem_flat_index;
            INA_TEST_ASSERT_EQUAL_FLOATING(value, ((float *) val2.elem_pointer)[0]);
        }
    }

    iarray_iter_read_free(&I2);
    INA_TEST_ASSERT(ina_err_get_rc() == INA_RC_PACK(IARRAY_ERR_END_ITER, 0));

    iarray_container_free(ctx, &c_x);
    blosc2_remove_urlpath(store.urlpath);
    return INA_SUCCESS;
}

INA_TEST_DATA(iterator) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(iterator) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(iterator) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(iterator, 2_d_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 2;
    int64_t shape[] = {4, 6};
    int64_t *cshape = NULL;
    int64_t *bshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_iterator(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, false, NULL));
}


INA_TEST_FIXTURE(iterator, 2_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 2;
    int64_t shape[] = {445, 321};
    int64_t cshape[] = {201, 17};
    int64_t bshape[] = {12, 8};

    INA_TEST_ASSERT_SUCCEED(test_iterator(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, false, "arr.iarr"));
}


INA_TEST_FIXTURE(iterator, 3_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 3;
    int64_t shape[] = {20, 53, 17};
    int64_t cshape[] = {12, 12, 2};
    int64_t bshape[] = {5, 5, 1};

    INA_TEST_ASSERT_SUCCEED(test_iterator(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, true, NULL));
}


INA_TEST_FIXTURE(iterator, 4_f_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 4;
    int64_t shape[] = {15, 18, 14, 13};
    int64_t *cshape = NULL;
    int64_t *bshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_iterator(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, true, "arr.iarr"));
}

INA_TEST_FIXTURE(iterator, 5_d_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 5;
    int64_t shape[] = {15, 18, 17, 13, 13};
    int64_t *cshape = NULL;
    int64_t *bshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_iterator(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, false, NULL));
}

INA_TEST_FIXTURE(iterator, 6_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 6;
    int64_t shape[] = {5, 7, 8, 9, 6, 5};
    int64_t cshape[] = {3, 3, 5, 5, 3, 5};
    int64_t bshape[] = {2, 2, 2, 2, 2, 2};

    INA_TEST_ASSERT_SUCCEED(test_iterator(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, false, "arr.iarr"));
}

INA_TEST_FIXTURE(iterator, 7_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 7;
    int64_t shape[] = {5, 7, 8, 9, 6, 5, 4};
    int64_t cshape[] = {2, 5, 3, 4, 3, 2, 2};
    int64_t bshape[] = {2, 2, 1, 2, 1, 2, 2};

    INA_TEST_ASSERT_SUCCEED(test_iterator(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, true, "arr.iarr"));
}

INA_TEST_FIXTURE(iterator, 8_f_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 8;
    int64_t shape[] = {5, 7, 8, 9, 6, 5, 3, 5};
    int64_t *cshape = NULL;
    int64_t *bshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_iterator(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, true, NULL));
}

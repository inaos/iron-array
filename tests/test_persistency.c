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

#include "src/iarray_private.h"
#include <libiarray/iarray.h>


static ina_rc_t test_persistency(iarray_context_t *ctx, iarray_data_type_t dtype, size_t type_size, int8_t ndim,
                                 const int64_t *shape, const int64_t *cshape, const int64_t *bshape,
                                 iarray_storage_t *store)
{
//    // For some reason, this test does not pass in Azure CI, so disable it temporarily (see #189)
//    char* envvar;
//    envvar = getenv("AGENT_OS");
//    if (envvar != NULL && strncmp(envvar, "Darwin", sizeof("Darwin")) == 0) {
//        printf("Skipping test on Azure CI (Darwin)...");
//        return INA_SUCCESS;
//    }

    iarray_dtshape_t xdtshape;
    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
        store->chunkshape[i] = cshape[i];
        store->blockshape[i] = bshape[i];
    }


    iarray_container_t *c_x;
    blosc2_remove_urlpath(store->urlpath);
    INA_TEST_ASSERT_SUCCEED(iarray_empty(ctx, &xdtshape, store, IARRAY_CONTAINER_PERSIST, &c_x));

    // Fill data via write iterator
    iarray_iter_write_t *I;
    iarray_iter_write_value_t val;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_write_new(ctx, &I, c_x, &val));
    while (INA_SUCCEED(iarray_iter_write_has_next(I))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_write_next(I));
        switch (dtype) {
            case IARRAY_DATA_TYPE_DOUBLE: {
                double value = (double) val.elem_flat_index;
                memcpy(val.elem_pointer, &value, type_size);
                break;
            }
            case IARRAY_DATA_TYPE_FLOAT: {
                float value = (float) val.elem_flat_index;
                memcpy(val.elem_pointer, &value, type_size);
                break;
            }
            case IARRAY_DATA_TYPE_INT64: {
                int64_t value = (int64_t) val.elem_flat_index;
                memcpy(val.elem_pointer, &value, type_size);
                break;
            }
            case IARRAY_DATA_TYPE_INT32: {
                int32_t value = (int32_t) val.elem_flat_index;
                memcpy(val.elem_pointer, &value, type_size);
                break;
            }
            case IARRAY_DATA_TYPE_INT16: {
                int16_t value = (int16_t) val.elem_flat_index;
                memcpy(val.elem_pointer, &value, type_size);
                break;
            }
            case IARRAY_DATA_TYPE_INT8: {
                int8_t value = (int8_t) val.elem_flat_index;
                memcpy(val.elem_pointer, &value, type_size);
                break;
            }
            case IARRAY_DATA_TYPE_UINT64: {
                uint64_t value = (uint64_t) val.elem_flat_index;
                memcpy(val.elem_pointer, &value, type_size);
                break;
            }
            case IARRAY_DATA_TYPE_UINT32: {
                uint32_t value = (uint32_t) val.elem_flat_index;
                memcpy(val.elem_pointer, &value, type_size);
                break;
            }
            case IARRAY_DATA_TYPE_UINT16: {
                uint16_t value = (uint16_t) val.elem_flat_index;
                memcpy(val.elem_pointer, &value, type_size);
                break;
            }
            case IARRAY_DATA_TYPE_UINT8: {
                uint8_t value = (uint8_t) val.elem_flat_index;
                memcpy(val.elem_pointer, &value, type_size);
                break;
            }
            case IARRAY_DATA_TYPE_BOOL: {
                bool value = (bool) val.elem_flat_index;
                memcpy(val.elem_pointer, &value, type_size);
                break;
            }
        }
    }
    iarray_iter_write_free(&I);
    INA_TEST_ASSERT(ina_err_get_rc() == INA_RC_PACK(IARRAY_ERR_END_ITER, 0));

    // Close the container and re-open it from disk
    iarray_container_free(ctx, &c_x);
    INA_TEST_ASSERT(_iarray_path_exists(store->urlpath));
    INA_TEST_ASSERT_SUCCEED(iarray_container_open(ctx, store->urlpath, &c_x));

    // Check values
    iarray_iter_read_t *I2;
    iarray_iter_read_value_t val2;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_new(ctx, &I2, c_x, &val2));
    while (INA_SUCCEED(iarray_iter_read_has_next(I2))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_read_next(I2));

        switch (dtype) {
            case IARRAY_DATA_TYPE_DOUBLE: {
                double value = (double) val2.elem_flat_index;
                INA_TEST_ASSERT_EQUAL_FLOATING(value, ((double *) val2.elem_pointer)[0]);
                break;
            }
            case IARRAY_DATA_TYPE_FLOAT: {
                float value = (float) val2.elem_flat_index;
                INA_TEST_ASSERT_EQUAL_FLOATING(value, ((float *) val2.elem_pointer)[0]);
                break;
            }
            case IARRAY_DATA_TYPE_INT64: {
                int64_t value = (int64_t) val2.elem_flat_index;
                INA_TEST_ASSERT_EQUAL_INT64(value, ((int64_t *) val2.elem_pointer)[0]);
                break;
            }
            case IARRAY_DATA_TYPE_INT32: {
                int32_t value = (int32_t) val2.elem_flat_index;
                INA_TEST_ASSERT_EQUAL_INT(value, ((int32_t *) val2.elem_pointer)[0]);
                break;
            }
            case IARRAY_DATA_TYPE_INT16: {
                int16_t value = (int16_t) val2.elem_flat_index;
                INA_TEST_ASSERT_EQUAL_INT(value, ((int16_t *) val2.elem_pointer)[0]);
                break;
            }
            case IARRAY_DATA_TYPE_INT8: {
                int8_t value = (int8_t) val2.elem_flat_index;
                INA_TEST_ASSERT_EQUAL_INT(value, ((int8_t *) val2.elem_pointer)[0]);
                break;
            }
            case IARRAY_DATA_TYPE_UINT64: {
                uint64_t value = (uint64_t) val2.elem_flat_index;
                INA_TEST_ASSERT_EQUAL_UINT64(value, ((uint64_t *) val2.elem_pointer)[0]);
                break;
            }
            case IARRAY_DATA_TYPE_UINT32: {
                uint32_t value = (uint32_t) val2.elem_flat_index;
                INA_TEST_ASSERT_EQUAL_UINT(value, ((uint32_t *) val2.elem_pointer)[0]);
                break;
            }
            case IARRAY_DATA_TYPE_UINT16: {
                uint16_t value = (uint16_t) val2.elem_flat_index;
                INA_TEST_ASSERT_EQUAL_UINT(value, ((uint16_t *) val2.elem_pointer)[0]);
                break;
            }
            case IARRAY_DATA_TYPE_UINT8: {
                uint8_t value = (uint8_t) val2.elem_flat_index;
                INA_TEST_ASSERT_EQUAL_UINT(value, ((uint8_t *) val2.elem_pointer)[0]);
                break;
            }
            case IARRAY_DATA_TYPE_BOOL: {
                bool value = (bool) val2.elem_flat_index;
                INA_TEST_ASSERT(value == ((bool *) val2.elem_pointer)[0]);
                break;
            }
        }
    }
    iarray_iter_read_free(&I2);
    INA_TEST_ASSERT(ina_err_get_rc() == INA_RC_PACK(IARRAY_ERR_END_ITER, 0));

    blosc2_remove_urlpath(store->urlpath);
    iarray_container_free(ctx, &c_x);

    return INA_SUCCESS;
}

INA_TEST_DATA(persistency) {
    iarray_context_t *ctx;
    iarray_storage_t store;
};

INA_TEST_SETUP(persistency) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));

    data->store.contiguous = true;
    data->store.urlpath = "test_persistency.b2frame";
    if (_iarray_path_exists(data->store.urlpath)) {
        blosc2_remove_urlpath(data->store.urlpath);
    }
}

INA_TEST_TEARDOWN(persistency) {
    blosc2_remove_urlpath(data->store.urlpath);
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(persistency, double_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    int8_t ndim = 2;
    int64_t shape[] = {125, 157};
    int64_t cshape[] = {12, 13};
    int64_t bshape[] = {7, 7};
    data->store.contiguous = false;

    INA_TEST_ASSERT_SUCCEED(test_persistency(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, &data->store));
}


INA_TEST_FIXTURE(persistency, float_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    int8_t ndim = 2;
    int64_t shape[] = {445, 321};
    int64_t cshape[] = {21, 17};
    int64_t bshape[] = {8, 9};
    data->store.contiguous = true;

    INA_TEST_ASSERT_SUCCEED(test_persistency(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, &data->store));
}

INA_TEST_FIXTURE(persistency, long_5) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT64;
    size_t type_size = sizeof(int64_t);

    int8_t ndim = 5;
    int64_t shape[] = {20, 25, 27, 4, 46};
    int64_t cshape[] = {12, 24, 19, 3, 13};
    int64_t bshape[] = {2, 5, 4, 3, 3};
    data->store.contiguous = false;

    INA_TEST_ASSERT_SUCCEED(test_persistency(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, &data->store));
}

INA_TEST_FIXTURE(persistency, uint_7) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT32;
    size_t type_size = sizeof(uint32_t);

    int8_t ndim = 7;
    int64_t shape[] = {10, 12, 8, 9, 1, 7, 7};
    int64_t cshape[] = {2, 5, 3, 4, 1, 3, 3};
    int64_t bshape[] = {2, 2, 2, 4, 1, 2, 1};
    data->store.contiguous = true;

    INA_TEST_ASSERT_SUCCEED(test_persistency(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, &data->store));
}

INA_TEST_FIXTURE(persistency, short_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT32;
    size_t type_size = sizeof(int32_t);

    int8_t ndim = 2;
    int64_t shape[] = {125, 157};
    int64_t cshape[] = {12, 13};
    int64_t bshape[] = {7, 7};
    data->store.contiguous = false;

    INA_TEST_ASSERT_SUCCEED(test_persistency(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, &data->store));
}


INA_TEST_FIXTURE(persistency, uchar_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT8;
    size_t type_size = sizeof(uint8_t);

    int8_t ndim = 2;
    int64_t shape[] = {445, 321};
    int64_t cshape[] = {21, 17};
    int64_t bshape[] = {8, 9};
    data->store.contiguous = true;

    INA_TEST_ASSERT_SUCCEED(test_persistency(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, &data->store));
}

INA_TEST_FIXTURE(persistency, bool_5) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_BOOL;
    size_t type_size = sizeof(bool);

    int8_t ndim = 5;
    int64_t shape[] = {20, 25, 27, 4, 46};
    int64_t cshape[] = {12, 24, 19, 3, 13};
    int64_t bshape[] = {2, 5, 4, 3, 3};
    data->store.contiguous = false;

    INA_TEST_ASSERT_SUCCEED(test_persistency(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, &data->store));
}

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


static ina_rc_t test_persistency(iarray_context_t *ctx, iarray_data_type_t dtype, size_t type_size, int8_t ndim,
                                 const int64_t *shape, const int64_t *pshape, iarray_store_properties_t *store)
{

    // Create dtshape
    iarray_dtshape_t xdtshape;
    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
        xdtshape.pshape[i] = pshape[i];
    }

    iarray_container_t *c_x;
    iarray_container_new(ctx, &xdtshape, store, IARRAY_CONTAINER_PERSIST, &c_x);

    // Start iterator
    iarray_iter_write_t *I;
    iarray_iter_write_value_t val;
    iarray_iter_write_new(ctx, &I, c_x, &val);

    while (iarray_iter_write_has_next(I)) {
        iarray_iter_write_next(I);

        if(dtype == IARRAY_DATA_TYPE_DOUBLE) {
            double value = (double) val.elem_flat_index;
            memcpy(val.elem_pointer, &value, type_size);
        } else {
            float value = (float) val.elem_flat_index;
            memcpy(val.elem_pointer, &value, type_size);
        }
    }

    iarray_iter_write_free(I);

    // Close the container and re-open it from disk
    iarray_container_free(ctx, &c_x);
    INA_TEST_ASSERT(_iarray_file_exists(store->id));
    INA_MUST_SUCCEED(iarray_from_file(ctx, store, &c_x));

    // Check values
    iarray_iter_read_t *I2;
    iarray_iter_read_value_t val2;
    iarray_iter_read_new(ctx, &I2, c_x, &val2);
    while (iarray_iter_read_has_next(I2)) {
        iarray_iter_read_next(I2);

        if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
            double value = (double) val2.elem_flat_index;
            INA_TEST_ASSERT_EQUAL_FLOATING(value, ((double *) val2.elem_pointer)[0]);
        } else {
            float value = (float) val2.elem_flat_index;
            INA_TEST_ASSERT_EQUAL_FLOATING(value, ((float *) val2.elem_pointer)[0]);
        }
    }
    iarray_iter_read_free(I2);

    iarray_container_free(ctx, &c_x);

    return INA_SUCCESS;
}

INA_TEST_DATA(persistency) {
    iarray_context_t *ctx;
    iarray_store_properties_t store;
};

INA_TEST_SETUP(persistency) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));

    data->store.id = "test_persistency.b2frame";
    if (_iarray_file_exists(data->store.id)) {
        remove(data->store.id);
    }
}

INA_TEST_TEARDOWN(persistency) {
    if (_iarray_file_exists(data->store.id)) {
        remove(data->store.id);
    }
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(persistency, double_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    int8_t ndim = 2;
    int64_t shape[] = {125, 157};
    int64_t pshape[] = {12, 13};

    INA_TEST_ASSERT_SUCCEED(test_persistency(data->ctx, dtype, type_size, ndim, shape, pshape, &data->store));
}

INA_TEST_FIXTURE(persistency, float_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    int8_t ndim = 2;
    int64_t shape[] = {445, 321};
    int64_t pshape[] = {21, 17};

    INA_TEST_ASSERT_SUCCEED(test_persistency(data->ctx, dtype, type_size, ndim, shape, pshape, &data->store));
}

INA_TEST_FIXTURE(persistency, double_5) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    int8_t ndim = 5;
    int64_t shape[] = {20, 25, 27, 4, 46};
    int64_t pshape[] = {12, 24, 19, 3, 13};

    INA_TEST_ASSERT_SUCCEED(test_persistency(data->ctx, dtype, type_size, ndim, shape, pshape, &data->store));
}

INA_TEST_FIXTURE(persistency, float_7) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    int8_t ndim = 7;
    int64_t shape[] = {10, 12, 8, 9, 1, 7, 7};
    int64_t pshape[] = {2, 5, 3, 4, 1, 3, 3};

    INA_TEST_ASSERT_SUCCEED(test_persistency(data->ctx, dtype, type_size, ndim, shape, pshape, &data->store));
}

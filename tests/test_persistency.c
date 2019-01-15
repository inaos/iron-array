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

/*
 * Check if a file exist using fopen() function
 * return 1 if the file exist otherwise return 0
 */
bool cfileexists(const char * filename)
{
    /* try to open file to read */
    FILE *file;
    if ((file = fopen(filename, "r"))) {
        fclose(file);
        return true;
    }
    return false;
}

static ina_rc_t test_persistency(iarray_context_t *ctx, iarray_data_type_t dtype, size_t type_size, uint8_t ndim,
                                 const uint64_t *shape, const uint64_t *pshape, iarray_store_properties_t *store)
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
    iarray_itr_t *I;
    iarray_itr_new(ctx, c_x, &I);

    for (iarray_itr_init(I); !iarray_itr_finished(I); iarray_itr_next(I)) {

        iarray_itr_value_t val;
        iarray_itr_value(I, &val);

        if(dtype == IARRAY_DATA_TYPE_DOUBLE) {
            double value = (double) val.nelem;
            memcpy(val.pointer, &value, type_size);
        } else {
            float value = (float) val.nelem;
            memcpy(val.pointer, &value, type_size);
        }
    }

    iarray_itr_free(ctx, I);

    // Close the container and re-open it from disk
    iarray_container_free(ctx, &c_x);
    INA_TEST_ASSERT(cfileexists(store->id));
    INA_MUST_SUCCEED(iarray_from_file(ctx, store, &c_x));

    // TODO: use the read iterators for testing this
    // Check values
    uint64_t bufsize = 1;
    for (int j = 0; j < ndim; ++j) {
        bufsize *= xdtshape.shape[j];
    }
    uint8_t *bufdest = ina_mem_alloc(bufsize * type_size);
    INA_MUST_SUCCEED(iarray_to_buffer(ctx, c_x, bufdest, bufsize));

    if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
        for (uint64_t k = 1; k < bufsize; ++k) {
            INA_TEST_ASSERT_EQUAL_FLOATING(((double *)bufdest)[k-1] + 1, ((double *)bufdest)[k]);
        }
    } else {
        for (uint64_t k = 1; k < bufsize; ++k) {
            INA_TEST_ASSERT_EQUAL_FLOATING(((float *)bufdest)[k-1] + 1, ((float *)bufdest)[k]);
        }
    }

    ina_mem_free(bufdest);
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
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.flags = IARRAY_EXPR_EVAL_CHUNK;

    iarray_context_new(&cfg, &data->ctx);

    data->store.id = "test_persistency.b2frame";
    if (cfileexists(data->store.id)) {
        remove(data->store.id);
    }
}

INA_TEST_TEARDOWN(persistency) {
    if (cfileexists(data->store.id)) {
        remove(data->store.id);
    }
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(persistency, double_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    uint8_t ndim = 2;
    uint64_t shape[] = {125, 157};
    uint64_t pshape[] = {12, 13};

    INA_TEST_ASSERT_SUCCEED(test_persistency(data->ctx, dtype, type_size, ndim, shape, pshape, &data->store));
}

INA_TEST_FIXTURE(persistency, float_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    uint8_t ndim = 2;
    uint64_t shape[] = {445, 321};
    uint64_t pshape[] = {21, 17};

    INA_TEST_ASSERT_SUCCEED(test_persistency(data->ctx, dtype, type_size, ndim, shape, pshape, &data->store));
}

INA_TEST_FIXTURE(persistency, double_5) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    uint8_t ndim = 5;
    uint64_t shape[] = {20, 25, 27, 4, 46};
    uint64_t pshape[] = {12, 24, 19, 3, 13};

    INA_TEST_ASSERT_SUCCEED(test_persistency(data->ctx, dtype, type_size, ndim, shape, pshape, &data->store));
}

INA_TEST_FIXTURE(persistency, float_7) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    uint8_t ndim = 7;
    uint64_t shape[] = {10, 12, 8, 9, 1, 7, 7};
    uint64_t pshape[] = {2, 5, 3, 4, 1, 3, 3};

    INA_TEST_ASSERT_SUCCEED(test_persistency(data->ctx, dtype, type_size, ndim, shape, pshape, &data->store));
}

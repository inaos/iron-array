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


static ina_rc_t test_load_save(iarray_context_t *ctx, iarray_data_type_t dtype, int8_t ndim,
                               const int64_t *shape, const int64_t *pshape, const int64_t *bshape,
                               double start, double stop, bool frame, bool fname)
{

    char *filename = "test_load_save.iarray";

    // For some reason, this test does not pass in Azure CI, so disable it temporarily (see #189)
//    char* envvar;
//    envvar = getenv("AGENT_OS");
//    if (envvar != NULL && strncmp(envvar, "Darwin", sizeof("Darwin")) == 0) {
//        printf("Skipping test on Azure CI (Darwin)...");
//        return INA_SUCCESS;
//    }

    // Create dtshape
    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    int64_t size = 1;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
        size *= shape[i];
    }

    double step = (stop - start) / size;
    iarray_container_t *c_x;

    int flags = 0;
    iarray_storage_t store;
    store.backend = IARRAY_STORAGE_BLOSC;
    store.filename = NULL;
    store.enforce_frame = false;
    if (frame) {
        store.enforce_frame = true;
    }
    if (fname) {
        store.filename = filename;
        flags = IARRAY_CONTAINER_PERSIST;
    }
    for (int i = 0; i < ndim; ++i) {
        store.chunkshape[i] = pshape[i];
        store.blockshape[i] = bshape[i];
        size *= shape[i];
    }
    INA_TEST_ASSERT_SUCCEED(iarray_arange(ctx, &xdtshape, start, stop, step, &store, flags, &c_x));


    if (!frame || !fname) {
        INA_TEST_ASSERT_SUCCEED(iarray_container_save(ctx, c_x, filename));
    }


    iarray_container_t *c_y;
    INA_TEST_ASSERT_SUCCEED(iarray_container_load(ctx, filename, true, &c_y));

    INA_TEST_ASSERT_SUCCEED(iarray_container_almost_equal(c_x, c_y, 1e-12));

    
    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);

    return INA_SUCCESS;
}


INA_TEST_DATA(container_load_save) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(container_load_save) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(container_load_save) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(container_load_save, 2_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 2;
    int64_t shape[] = {35, 44};
    int64_t pshape[] = {12, 10};
    int64_t bshape[] = {5, 5};
    double start = - 0.1;
    double stop = - 0.25;

    INA_TEST_ASSERT_SUCCEED(test_load_save(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop, false, false));
}

INA_TEST_FIXTURE(container_load_save, 3_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 2;
    int64_t shape[] = {43, 33};
    int64_t pshape[] = {14, 12};
    int64_t bshape[] = {7, 7};
    double start = 3123;
    double stop = 45654;

    INA_TEST_ASSERT_SUCCEED(test_load_save(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop, true, false));
}

INA_TEST_FIXTURE(container_load_save, 5_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 3;
    int64_t shape[] = {20, 55, 125};
    int64_t pshape[] = {12, 14, 15};
    int64_t bshape[] = {4, 3, 4};
    double start = 0.1;
    double stop = 0.2;

    INA_TEST_ASSERT_SUCCEED(test_load_save(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop, true, true));
}

INA_TEST_FIXTURE(container_load_save, 2_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 2;
    int64_t shape[] = {120, 100};
    int64_t pshape[] = {50, 51};
    int64_t bshape[] = {5, 22};
    double start = - 0.1;
    double stop = - 0.25;

    INA_TEST_ASSERT_SUCCEED(test_load_save(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop, false, false));
}

INA_TEST_FIXTURE(container_load_save, 3_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 3;
    int64_t shape[] = {50, 10, 80};
    int64_t pshape[] = {21, 7, 7};
    int64_t bshape[] = {21, 2, 2};
    double start = 3123;
    double stop = 45654;

    INA_TEST_ASSERT_SUCCEED(test_load_save(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop, true, false));
}

INA_TEST_FIXTURE(container_load_save, 5_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 5;
    int64_t shape[] = {4, 5, 10, 5, 4};
    int64_t pshape[] = {3, 4, 3, 3, 2};
    int64_t bshape[] = {3, 3, 2, 3, 2};
    double start = 0.1;
    double stop = 0.2;

    INA_TEST_ASSERT_SUCCEED(test_load_save(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop, true, true));
}

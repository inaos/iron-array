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

static ina_rc_t test_copy(iarray_context_t *ctx, iarray_data_type_t dtype, int8_t ndim,
                           const int64_t *shape, const int64_t *pshape, const int64_t *bshape, double start,
                           double stop, int64_t *stop_view, bool src_view, bool dest_view)
{
//    For some reason, this test does not pass in Azure CI, so disable it temporarily (see #189)
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

    iarray_storage_t store;
    store.backend = (pshape == NULL) ? IARRAY_STORAGE_PLAINBUFFER : IARRAY_STORAGE_BLOSC;
    store.filename = NULL;
    store.enforce_frame = (ndim % 2 == 0) ? false : true;
    for (int i = 0; i < ndim; ++i) {
        if (pshape != NULL) {
            store.pshape[i] = pshape[i];
            store.bshape[i] = bshape[i];
        }
    }
    double step = (stop - start) / size;

    iarray_container_t *c_x;
    iarray_container_t *c_aux;

    if (src_view) {
        INA_TEST_ASSERT_SUCCEED(iarray_arange(ctx, &xdtshape, start, stop, step, &store, 0, &c_aux));
        int64_t start_view[IARRAY_DIMENSION_MAX];
        for (int i = 0; i < ndim; ++i) {
            start_view[i] = 0;
        }
        INA_TEST_ASSERT_SUCCEED(iarray_get_slice(ctx, c_aux, start_view, stop_view, true, &store, 0, &c_x));
        INA_TEST_ASSERT_SUCCEED(iarray_squeeze(ctx, c_x));
    } else {
        INA_TEST_ASSERT_SUCCEED(iarray_arange(ctx, &xdtshape, start, stop, step, &store, 0, &c_x));
    }

    iarray_container_t *c_y;

    INA_TEST_ASSERT_SUCCEED(iarray_copy(ctx, c_x, dest_view, &store, 0, &c_y));

    // Assert iterator reading it
    double tol;
    if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
        tol = 1e-14;
    } else {
        tol = 1e-6;
    }
    iarray_container_almost_equal(c_x, c_y, tol);

    iarray_container_free(ctx, &c_y);
    iarray_container_free(ctx, &c_x);
    if (src_view) {
        iarray_container_free(ctx, &c_aux);
    }

    return INA_SUCCESS;
}

INA_TEST_DATA(constructor_copy) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(constructor_copy) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_BLOSCLZ;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(constructor_copy) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(constructor_copy, 1_f_p_n_n) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 1;
    int64_t shape[] = {1000};
    int64_t *pshape = NULL;
    int64_t *bshape = NULL;
    int64_t stop_view[] = {431};
    double start = 0;
    double stop = 1;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop, stop_view, false, false));
}


INA_TEST_FIXTURE(constructor_copy, 2_f_p_v_n) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 2;
    int64_t shape[] = {10, 200};
    int64_t *pshape = NULL;
    int64_t *bshape = NULL;
    int64_t stop_view[] = {1, 121};
    double start = - 0.1;
    double stop = - 0.2;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop, stop_view, true, false));
}

INA_TEST_FIXTURE(constructor_copy, 3_f_p_n_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 3;
    int64_t shape[] = {10, 20, 10};
    int64_t *pshape = NULL;
    int64_t *bshape = NULL;
    int64_t stop_view[] = {2, 5, 6};
    double start = 1;
    double stop = 25;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop, stop_view, false, true));
}

INA_TEST_FIXTURE(constructor_copy, 4_f_p_v_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 4;
    int64_t shape[] = {10, 1, 1, 33};
    int64_t *pshape = NULL;
    int64_t *bshape = NULL;
    int64_t stop_view[] = {5, 1, 1, 12};
    double start = - 5;
    double stop = 101010;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop, stop_view, true, true));
}


INA_TEST_FIXTURE(constructor_copy, 5_d_p_n_n) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 5;
    int64_t shape[] = {2, 3, 4, 5, 6};
    int64_t *pshape = NULL;
    int64_t *bshape = NULL;
    int64_t stop_view[] = {2, 2, 2, 2, 2};
    double start = - 0.1;
    double stop = - 0.25;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop, stop_view, false, false));
}


INA_TEST_FIXTURE(constructor_copy, 6_d_p_v_n) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 6;
    int64_t shape[] = {6, 3, 6, 3, 6, 3};
    int64_t *pshape = NULL;
    int64_t *bshape = NULL;
    int64_t stop_view[] = {4, 3, 2, 3, 4, 3};

    double start = 1000;
    double stop = 2000;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop, stop_view, true, false));
}

INA_TEST_FIXTURE(constructor_copy, 7_d_p_n_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 7;
    int64_t shape[] = {2, 4, 6, 8, 6, 4, 2};
    int64_t *pshape = NULL;
    int64_t *bshape = NULL;
    int64_t stop_view[] = {2, 3, 5, 2, 2, 2};

    double start = 0;
    double stop = 0.000001;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop, stop_view, false, true));
}

INA_TEST_FIXTURE(constructor_copy, 8_d_p_v_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 8;
    int64_t shape[] = {2, 9, 3, 8, 4, 7, 5, 6};
    int64_t *pshape = NULL;
    int64_t *bshape = NULL;
    int64_t stop_view[] = {2, 2, 2, 2, 2, 2, 2, 2};
    double start = -1;
    double stop = 1;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop, stop_view, true, true));
}



INA_TEST_FIXTURE(constructor_copy, 8_f_n_n) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 8;
    int64_t shape[] = {5, 4, 7, 5, 4, 6, 2, 3};
    int64_t pshape[] = {2, 1, 2, 2, 2, 1, 1, 2};
    int64_t bshape[] = {2, 1, 2, 2, 2, 1, 1, 2};
    int64_t stop_view[] = {2, 2, 2, 2, 2, 2, 2, 2};
    double start = 0;
    double stop = 1;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop, stop_view, false, false));
}


INA_TEST_FIXTURE(constructor_copy, 7_f_v_n) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 7;
    int64_t shape[] = {7, 4, 8, 4, 5, 8, 4};
    int64_t pshape[] = {2, 2, 2, 3, 3, 2, 2};
    int64_t bshape[] = {2, 2, 1, 2, 2, 1, 2};
    int64_t stop_view[] = {3, 3, 3, 3, 3, 3, 3};

    double start = 0;
    double stop = 5;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop, stop_view, true, false));
}


INA_TEST_FIXTURE(constructor_copy, 6_f_n_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 6;
    int64_t shape[] = {5, 7, 10, 12, 13, 6};
    int64_t pshape[] = {2, 1, 4, 5, 6, 4};
    int64_t bshape[] = {2, 1, 2, 3, 2, 3};
    int64_t stop_view[] = {4, 4, 5, 11, 12, 4};
    double start = -0.112;
    double stop = 10102;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop, stop_view, false, true));
}

INA_TEST_FIXTURE(constructor_copy, 5_f_v_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 5;
    int64_t shape[] = {31, 21, 11, 5, 11};
    int64_t pshape[] = {10, 11, 3, 2, 4};
    int64_t bshape[] = {4, 5, 1, 2, 2};
    int64_t stop_view[] = {21, 10, 3, 3, 8};

    double start = 1;
    double stop = -1;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop, stop_view, true, true));
}

INA_TEST_FIXTURE(constructor_copy, 4_d_n_n) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 4;
    int64_t shape[] = {12, 31, 54, 12};
    int64_t pshape[] = {2, 3, 23, 5};
    int64_t bshape[] = {1, 2, 10, 2};
    int64_t stop_view[] = {8, 8, 8, 3};

    double start = 0.1;
    double stop = 0.9;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop, stop_view, false, false));
}

INA_TEST_FIXTURE(constructor_copy, 3_d_v_n) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 3;
    int64_t shape[] = {31, 45, 23};
    int64_t pshape[] = {10, 12, 13};
    int64_t bshape[] = {7, 8, 10};
    int64_t stop_view[] = {21, 17, 15};

    double start = 0.00001;
    double stop = 0.00002;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop, stop_view, true, false));
}

INA_TEST_FIXTURE(constructor_copy, 2_d_n_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 2;
    int64_t shape[] = {54, 66};
    int64_t pshape[] = {21, 17};
    int64_t bshape[] = {9, 5};
    int64_t stop_view[] = {22, 31};

    double start = 3123;
    double stop = 45654;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop, stop_view, false, true));
}

INA_TEST_FIXTURE(constructor_copy, 1_d_v_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 1;
    int64_t shape[] = {445};
    int64_t pshape[] = {132};
    int64_t bshape[] = {21};
    int64_t stop_view[] = {121};
    double start = -0.1;
    double stop = 0.1;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop, stop_view, true, true));
}

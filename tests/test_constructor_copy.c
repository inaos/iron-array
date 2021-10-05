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
                           const int64_t *shape, const int64_t *cshape, const int64_t *bshape, double start,
                           double stop, int64_t *stop_view, bool src_view, bool dest_view, char *urlpath)
{
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
    store.urlpath = urlpath;
    store.contiguous = (ndim % 2 == 0) ? false : true;
    for (int i = 0; i < ndim; ++i) {
        store.chunkshape[i] = cshape[i];
        store.blockshape[i] = bshape[i];
    }
    double step = (stop - start) / size;

    iarray_container_t *c_x;
    iarray_container_t *c_aux;
    blosc2_remove_urlpath(store.urlpath);

    if (src_view) {
        INA_TEST_ASSERT_SUCCEED(iarray_arange(ctx, &xdtshape, start, step, &store, &c_aux));
        int64_t start_view[IARRAY_DIMENSION_MAX];
        for (int i = 0; i < ndim; ++i) {
            start_view[i] = 0;
        }
        INA_TEST_ASSERT_SUCCEED(iarray_get_slice(ctx, c_aux, start_view, stop_view, true, &store,
                                                 &c_x));
        INA_TEST_ASSERT_SUCCEED(iarray_squeeze(ctx, c_x));
    } else {
        INA_TEST_ASSERT_SUCCEED(iarray_arange(ctx, &xdtshape, start, step, &store, &c_x));
    }

    iarray_container_t *c_y;
    if (store.urlpath != NULL) {
        store.urlpath = "arr2.iarr";
        blosc2_remove_urlpath(store.urlpath);
    }

    INA_TEST_ASSERT_SUCCEED(iarray_copy(ctx, c_x, dest_view, &store, &c_y));

    // Assert iterator reading it
    switch (dtype) {
        case IARRAY_DATA_TYPE_DOUBLE: {
            double tol = 1e-14;
            INA_TEST_ASSERT_SUCCEED(iarray_container_almost_equal(c_x, c_y, tol));
            break;
        }
        case IARRAY_DATA_TYPE_FLOAT: {
            double tol = 1e-6;
            INA_TEST_ASSERT_SUCCEED(iarray_container_almost_equal(c_x, c_y, tol));
            break;
        }
        case IARRAY_DATA_TYPE_INT64:
            INA_TEST_ASSERT_SUCCEED(iarray_container_equal(c_x, c_y));
            break;
        case IARRAY_DATA_TYPE_INT32:
            INA_TEST_ASSERT_SUCCEED(iarray_container_equal(c_x, c_y));
            break;
        case IARRAY_DATA_TYPE_INT16:
            INA_TEST_ASSERT_SUCCEED(iarray_container_equal(c_x, c_y));
            break;
        case IARRAY_DATA_TYPE_INT8:
            INA_TEST_ASSERT_SUCCEED(iarray_container_equal(c_x, c_y));
            break;
        case IARRAY_DATA_TYPE_UINT64:
            INA_TEST_ASSERT_SUCCEED(iarray_container_equal(c_x, c_y));
            break;
        case IARRAY_DATA_TYPE_UINT32:
            INA_TEST_ASSERT_SUCCEED(iarray_container_equal(c_x, c_y));
            break;
        case IARRAY_DATA_TYPE_UINT16:
            INA_TEST_ASSERT_SUCCEED(iarray_container_equal(c_x, c_y));
            break;
        case IARRAY_DATA_TYPE_UINT8:
            INA_TEST_ASSERT_SUCCEED(iarray_container_equal(c_x, c_y));
            break;
        case IARRAY_DATA_TYPE_BOOL:
            INA_TEST_ASSERT_SUCCEED(iarray_container_equal(c_x, c_y));
            break;
    }


    iarray_container_free(ctx, &c_y);
    iarray_container_free(ctx, &c_x);
    if (src_view) {
        iarray_container_free(ctx, &c_aux);
    }
    blosc2_remove_urlpath(urlpath);
    blosc2_remove_urlpath(store.urlpath);

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


INA_TEST_FIXTURE(constructor_copy, 8_f_n_n) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 8;
    int64_t shape[] = {5, 4, 7, 5, 4, 6, 2, 3};
    int64_t cshape[] = {2, 1, 2, 2, 2, 1, 1, 2};
    int64_t bshape[] = {2, 1, 2, 2, 2, 1, 1, 2};
    int64_t stop_view[] = {2, 2, 2, 2, 2, 2, 2, 2};
    double start = 0;
    double stop = 1;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, stop_view, false, false, NULL));
}


INA_TEST_FIXTURE(constructor_copy, 3_d_v_n) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 3;
    int64_t shape[] = {31, 45, 23};
    int64_t cshape[] = {10, 12, 13};
    int64_t bshape[] = {7, 8, 10};
    int64_t stop_view[] = {21, 17, 15};

    double start = 0.00001;
    double stop = 0.00002;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, stop_view, true, false, "arr.iarr"));
}

INA_TEST_FIXTURE(constructor_copy, 3_ll_n_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT64;

    int8_t ndim = 3;
    int64_t shape[] = {5, 7, 10};
    int64_t cshape[] = {2, 1, 4};
    int64_t bshape[] = {2, 1, 2};
    int64_t stop_view[] = {4, 4, 5};
    double start = -112;
    double stop = (5 * 7 * 10 * 12 * 13 * 6 - 112 + 1) * 3;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, stop_view, false, true, "arr.iarr"));
}

/* Avoid heavy tests
INA_TEST_FIXTURE(constructor_copy, 6_ll_n_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT64;

    int8_t ndim = 6;
    int64_t shape[] = {5, 7, 10, 12, 13, 6};
    int64_t cshape[] = {2, 1, 4, 5, 6, 4};
    int64_t bshape[] = {2, 1, 2, 3, 2, 3};
    int64_t stop_view[] = {4, 4, 5, 11, 12, 4};
    double start = -112;
    double stop = (5 * 7 * 10 * 12 * 13 * 6 - 112 + 1) * 3;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, stop_view, false, true, "arr.iarr"));
}
*/

INA_TEST_FIXTURE(constructor_copy, 4_i_n_n) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT32;

    int8_t ndim = 4;
    int64_t shape[] = {12, 31, 54, 12};
    int64_t cshape[] = {2, 3, 23, 5};
    int64_t bshape[] = {1, 2, 10, 2};
    int64_t stop_view[] = {8, 8, 8, 3};

    double start = 1;
    double stop = 12 * 31 * 54 * 12 + 1;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, stop_view, false, false, NULL));
}

INA_TEST_FIXTURE(constructor_copy, 2_s_v_n) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT16;

    int8_t ndim = 2;
    int64_t shape[] = {54, 66};
    int64_t cshape[] = {21, 17};
    int64_t bshape[] = {9, 5};
    int64_t stop_view[] = {22, 31};

    double start = 3123;
    double stop = 3123 + 54 * 66 + 1;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, stop_view, true, false, "arr.iarr"));
}

INA_TEST_FIXTURE(constructor_copy, 3_sc_n_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT8;

    int8_t ndim = 3;
    int64_t shape[] = {7, 4, 8};
    int64_t cshape[] = {2, 2, 2};
    int64_t bshape[] = {2, 2, 1};
    int64_t stop_view[] = {3, 3, 3};

    double start = 0;
    double stop = 7 * 4 * 8 * 2;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, stop_view, false, true, NULL));
}

INA_TEST_FIXTURE(constructor_copy, 8_ull_n_n) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT64;

    int8_t ndim = 8;
    int64_t shape[] = {5, 4, 7, 5, 4, 6, 2, 3};
    int64_t cshape[] = {2, 1, 2, 2, 2, 1, 1, 2};
    int64_t bshape[] = {2, 1, 2, 2, 2, 1, 1, 2};
    int64_t stop_view[] = {2, 2, 2, 2, 2, 2, 2, 2};
    double start = 0;
    double stop = 5 * 4 * 7 * 5 * 4 * 6 * 2 * 3 * 23;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, stop_view, false, false, NULL));
}


INA_TEST_FIXTURE(constructor_copy, 7_ui_v_n) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT32;

    int8_t ndim = 7;
    int64_t shape[] = {7, 4, 8, 4, 5, 8, 4};
    int64_t cshape[] = {2, 2, 2, 3, 3, 2, 2};
    int64_t bshape[] = {2, 2, 1, 2, 2, 1, 2};
    int64_t stop_view[] = {3, 3, 3, 3, 3, 3, 3};

    double start = 0;
    double stop = 7 * 4 * 8 * 4 * 5 * 8 * 4 * 2;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, stop_view, true, false, "arr.iarr"));
}

INA_TEST_FIXTURE(constructor_copy, 2_us_n_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT16;

    int8_t ndim = 2;
    int64_t shape[] = {12, 13};
    int64_t cshape[] = {5, 6};
    int64_t bshape[] = {3, 2};
    int64_t stop_view[] = {11, 12};
    double start = 11;
    double stop = 5 * 7 * 10 * 12 * 13 * 6 + 11;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, stop_view, false, true, "arr.iarr"));
}

/* Avoid heavy tests
INA_TEST_FIXTURE(constructor_copy, 6_us_n_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT16;

    int8_t ndim = 6;
    int64_t shape[] = {5, 7, 10, 12, 13, 6};
    int64_t cshape[] = {2, 1, 4, 5, 6, 4};
    int64_t bshape[] = {2, 1, 2, 3, 2, 3};
    int64_t stop_view[] = {4, 4, 5, 11, 12, 4};
    double start = 11;
    double stop = 5 * 7 * 10 * 12 * 13 * 6 + 11;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, stop_view, false, true, "arr.iarr"));
}
*/

INA_TEST_FIXTURE(constructor_copy, 1_uc_n_n) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT8;

    int8_t ndim = 1;
    int64_t shape[] = {12};
    int64_t cshape[] = {2};
    int64_t bshape[] = {1};
    int64_t stop_view[] = {8};

    double start = 1;
    double stop = 12 + 1;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, stop_view, false, false, NULL));
}

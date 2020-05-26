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
#include <src/iarray_private.h>

static ina_rc_t test_slice(iarray_context_t *ctx, iarray_container_t *c_x, int64_t *start,
                           int64_t *stop, const int64_t *pshape, iarray_storage_t *stores,
                           int flags, iarray_container_t **c_out) {
    INA_TEST_ASSERT_SUCCEED(iarray_get_slice(ctx, c_x, start, stop, true, flags, c_out, NULL));
    INA_TEST_ASSERT_SUCCEED(iarray_squeeze(ctx, *c_out));

    return INA_SUCCESS;
}

static ina_rc_t _execute_iarray_slice(iarray_context_t *ctx, iarray_data_type_t dtype, int32_t type_size, int8_t ndim,
                                      const int64_t *shape, const int64_t *pshape, const int64_t *pshape_dest,
                                      int64_t *start, int64_t *stop, bool transposed) {
    void *buffer_x;
    size_t buffer_x_len;

    buffer_x_len = 1;
    for (int i = 0; i < ndim; ++i) {
        buffer_x_len *= shape[i];
    }
    buffer_x = ina_mem_alloc(buffer_x_len * type_size);

    if (type_size == sizeof(float)) {
        ffill_buf((float *) buffer_x, buffer_x_len);

    } else {
        dfill_buf((double *) buffer_x, buffer_x_len);
    }

    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    for (int j = 0; j < xdtshape.ndim; ++j) {
        xdtshape.shape[j] = shape[j];
        if (pshape)
            xdtshape.pshape[j] = pshape[j];
    }

    iarray_container_t *c_x;
    iarray_container_t *c_out;
    iarray_container_t *c_sview;

    iarray_storage_t xstore;
    xstore.backend = pshape_dest ? IARRAY_STORAGE_BLOSC : IARRAY_STORAGE_PLAINBUFFER;
    xstore.enforce_frame = false;
    xstore.filename = NULL;

    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &xdtshape, buffer_x, buffer_x_len * type_size, &xstore, 0, &c_x));

    if (transposed) {
        iarray_linalg_transpose(ctx, c_x);
    }

    iarray_storage_t outstore;
    outstore.backend = pshape_dest ? IARRAY_STORAGE_BLOSC : IARRAY_STORAGE_PLAINBUFFER;
    outstore.enforce_frame = false;
    outstore.filename = NULL;
    INA_TEST_ASSERT_SUCCEED(test_slice(ctx, c_x, start, stop, pshape_dest, &outstore, 0, &c_out));

    uint8_t *sview;
    int64_t sview_len;

    INA_TEST_ASSERT_SUCCEED(iarray_to_sview(ctx, c_out, &sview, &sview_len));

    INA_TEST_ASSERT_SUCCEED(iarray_from_sview(ctx, sview, sview_len, &c_sview));

    INA_TEST_ASSERT(c_out->dtshape->dtype == c_sview->dtshape->dtype);
    INA_TEST_ASSERT(c_out->dtshape->ndim == c_sview->dtshape->ndim);
    for (int i = 0; i < IARRAY_DIMENSION_MAX; ++i) {
        INA_TEST_ASSERT(c_out->dtshape->shape[i] == c_sview->dtshape->shape[i]);
        INA_TEST_ASSERT(c_out->dtshape->pshape[i] == c_sview->dtshape->pshape[i]);
        INA_TEST_ASSERT(c_out->auxshape->offset[i] == c_sview->auxshape->offset[i]);
        INA_TEST_ASSERT(c_out->auxshape->shape_wos[i] == c_sview->auxshape->shape_wos[i]);
        INA_TEST_ASSERT(c_out->auxshape->pshape_wos[i] == c_sview->auxshape->pshape_wos[i]);
        INA_TEST_ASSERT(c_out->auxshape->index[i] == c_sview->auxshape->index[i]);
    }
    INA_TEST_ASSERT(c_out->catarr == c_sview->catarr);
    INA_TEST_ASSERT(c_out->view == c_sview->view);
    INA_TEST_ASSERT(c_out->transposed == c_sview->transposed);

    iarray_container_free(ctx, &c_sview);
    iarray_container_free(ctx, &c_out);

    iarray_container_free(ctx, &c_x);

    ina_mem_free(buffer_x);

    return INA_SUCCESS;
}

INA_TEST_DATA(view_serialization) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(view_serialization) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;

    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(view_serialization) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

/*
INA_TEST_FIXTURE(view_serialization, 2_d_p_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    const int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t *pshape = NULL;
    int64_t start[] = {-5, -7};
    int64_t stop[] = {-1, 10};
    int64_t *pshape_dest = NULL;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, pshape, pshape_dest,
        start, stop, false));
}

INA_TEST_FIXTURE(view_serialization, 3_f_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    const int8_t ndim = 3;
    int64_t shape[] = {10, 10, 10};
    int64_t pshape[] = {3, 5, 2};
    int64_t start[] = {3, 0, 3};
    int64_t stop[] = {-4, -3, 10};
    int64_t pshape_dest[] = {2, 4, 3};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, pshape, pshape_dest,
                                                  start, stop, false));
}

INA_TEST_FIXTURE(view_serialization, 4_d_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    const int8_t ndim = 4;
    int64_t shape[] = {10, 10, 10, 10};
    int64_t pshape[] = {3, 5, 2, 7};
    int64_t start[] = {5, -7, 9, 2};
    int64_t stop[] = {-1, 6, 10, -3};
    int64_t pshape_dest[] = {2, 2, 1, 3};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, pshape, pshape_dest,
                                                  start, stop, false));
}

INA_TEST_FIXTURE(view_serialization, 5_f_p_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    const int8_t ndim = 5;
    int64_t shape[] = {10, 10, 10, 10, 10};
    int64_t *pshape = NULL;
    int64_t start[] = {-4, 0, -5, 5, 7};
    int64_t stop[] = {8, 9, -4, -4, 10};
    int64_t *pshape_dest = NULL;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, pshape, pshape_dest,
                                                  start, stop, false));
}

INA_TEST_FIXTURE(view_serialization, 6_d_p_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    const int8_t ndim = 6;
    int64_t shape[] = {10, 10, 10, 10, 10, 10};
    int64_t *pshape = NULL;
    int64_t start[] = {0, 4, -8, 4, 5, 1};
    int64_t stop[] = {1, 7, 4, -4, 8, 3};
    int64_t *pshape_dest = NULL;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, pshape, pshape_dest,
                                                  start, stop, false));
}

INA_TEST_FIXTURE(view_serialization, 7_f_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    const int8_t ndim = 7;
    int64_t shape[] = {10, 10, 10, 10, 10, 10, 10};
    int64_t pshape[] = {4, 5, 1, 8, 5, 3, 10};
    int64_t start[] = {5, 4, 3, -2, 4, 5, -9};
    int64_t stop[] = {8, 6, 5, 9, 7, 7, -7};
    int64_t pshape_dest[] = {2, 2, 1, 1, 2, 2, 2};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, pshape, pshape_dest,
                                                  start, stop, false));
}

INA_TEST_DATA(view_serialization_trans) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(view_serialization_trans) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(view_serialization_trans) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(view_serialization_trans, 2_d_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    const int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t pshape[] = {3, 4};
    int64_t start[] = {2, 1};
    int64_t stop[] = {7, 3};
    int64_t pshape_dest[] = {2, 2};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, pshape, pshape_dest,
                                                  start, stop, true));
}

INA_TEST_FIXTURE(view_serialization_trans, 2_f_p_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    const int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t pshape[] = {3, 2};
    int64_t start[] = {2, 1};
    int64_t stop[] = {7, 3};
    int64_t *pshape_dest = NULL;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, pshape, pshape_dest,
                                                  start, stop, true));
}


INA_TEST_FIXTURE(view_serialization_trans, 2_f_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    const int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t pshape[] = {1, 1};
    int64_t start[] = {3, 1};
    int64_t stop[] = {5, 8};
    int64_t pshape_dest[] = {1, 2};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, pshape, pshape_dest,
                                                  start, stop, true));
}
*/

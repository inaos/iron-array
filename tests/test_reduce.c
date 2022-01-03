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

#include "iarray_test.h"
#include <libiarray/iarray.h>
#include <src/iarray_private.h>


static ina_rc_t test_reduce(iarray_context_t *ctx, iarray_data_type_t dtype, int8_t ndim,
                               const int64_t *shape, const int64_t *cshape, const int64_t *bshape,
                               int8_t axis,
                               int64_t *dest_cshape, int64_t *dest_bshape, bool dest_frame,
                               char *dest_urlpath) {
    blosc2_remove_urlpath(dest_urlpath);
    // Create dtshape
    iarray_dtshape_t dtshape;

    dtshape.dtype = dtype;
    dtshape.ndim = ndim;
    int64_t size = 1;
    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
        size *= shape[i];
    }

    iarray_storage_t storage = {0};
    for (int i = 0; i < ndim; ++i) {
        storage.chunkshape[i] = i == axis ? shape[i] : 1;
        storage.blockshape[i] = i == axis ? shape[i] : 1;
    }

    iarray_container_t *c_x;
    IARRAY_RETURN_IF_FAILED(iarray_empty(ctx, &dtshape, &storage, 0, &c_x));


    iarray_iter_write_block_t *iter;
    iarray_iter_write_block_value_t iter_value;
    IARRAY_RETURN_IF_FAILED(iarray_iter_write_block_new(ctx, &iter, c_x, storage.chunkshape,
                                                        &iter_value, false));
    while (INA_SUCCEED(iarray_iter_write_block_has_next(iter))) {
        IARRAY_RETURN_IF_FAILED(iarray_iter_write_block_next(iter, NULL, 0));
        fill_block_iter(iter_value, 0, c_x->dtshape->dtype);
    }
    iarray_iter_write_block_free(&iter);
    IARRAY_ITER_FINISH();


    for (int i = 0; i < ndim; ++i) {
        storage.chunkshape[i] = cshape[i];
        storage.blockshape[i] = bshape[i];
    }

    iarray_container_t *c_y;
    IARRAY_RETURN_IF_FAILED(iarray_copy(ctx, c_x, false, &storage, 0, &c_y));

    iarray_storage_t dest_storage = {0};
    dest_storage.contiguous = dest_frame;
    dest_storage.urlpath = dest_urlpath;
    for (int i = 0; i < ndim - 1; ++i) {
        dest_storage.blockshape[i] = dest_bshape[i];
        dest_storage.chunkshape[i] = dest_cshape[i];
    }

    iarray_container_t *c_z;
    IARRAY_RETURN_IF_FAILED(iarray_reduce(ctx, c_y, IARRAY_REDUCE_SUM, axis, &dest_storage, &c_z));

    int64_t buffer_nitems = c_z->catarr->nitems;
    int64_t buffer_size = buffer_nitems * c_z->catarr->itemsize;
    uint8_t *buffer = malloc(buffer_size);

    IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, c_z, buffer, buffer_size));

    double val = shape[axis] * (shape[axis] - 1.) / 2;
    for (int i = 0; i < buffer_nitems; ++i) {
        // printf("%d: %f - %f\n", i, ((double *) buffer)[i], val);
        switch (c_z->dtshape->dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                INA_TEST_ASSERT_EQUAL_FLOATING(((double *) buffer)[i], val);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                INA_TEST_ASSERT_EQUAL_FLOATING(((float *) buffer)[i], val);
                break;
            case IARRAY_DATA_TYPE_INT64:
                INA_TEST_ASSERT_EQUAL_INT64(((int64_t *) buffer)[i], val);
                break;
            case IARRAY_DATA_TYPE_UINT64:
                INA_TEST_ASSERT_EQUAL_UINT64(((uint64_t *) buffer)[i], val);
                break;
            default:
                IARRAY_TRACE1(iarray.error, "Invalid dtype");
                return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
        }
    }

    iarray_container_free(ctx, &c_z);
    iarray_container_free(ctx, &c_y);
    iarray_container_free(ctx, &c_x);

    return INA_SUCCESS;
}

INA_TEST_DATA(reduce) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(reduce) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(reduce) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}


INA_TEST_FIXTURE(reduce, 2_i_1) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT32;

    int8_t ndim = 2;
    int64_t shape[] = {8, 8};
    int64_t cshape[] = {4, 4};
    int64_t bshape[] = {2, 2};
    int8_t axis = 1;

    int64_t dest_cshape[] = {4};
    int64_t dest_bshape[] = {2};
    bool dest_frame = false;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce(data->ctx, dtype, ndim, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}


INA_TEST_FIXTURE(reduce, 3_s_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT16;

    int8_t ndim = 3;
    int64_t shape[] = {12, 12, 12};
    int64_t cshape[] = {6, 9, 6};
    int64_t bshape[] = {3, 3, 3};
    int8_t axis = 2;

    int64_t dest_cshape[] = {6, 6};
    int64_t dest_bshape[] = {3, 3};
    bool dest_frame = false;
    char *dest_urlpath = NULL;

    INA_TEST_ASSERT_SUCCEED(test_reduce(data->ctx, dtype, ndim, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}


INA_TEST_FIXTURE(reduce, 4_d_0) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 4;
    int64_t shape[] = {10, 10, 10, 10};
    int64_t cshape[] = {5, 5, 5, 5};
    int64_t bshape[] = {2, 2, 2, 2};
    int8_t axis = 0;

    int64_t dest_cshape[] = {5, 1, 5};
    int64_t dest_bshape[] = {2, 1, 2};
    bool dest_frame = true;
    char *dest_urlpath = "arr.iarr";

    INA_TEST_ASSERT_SUCCEED(test_reduce(data->ctx, dtype, ndim, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}



INA_TEST_FIXTURE(reduce, 8_ull_6) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT64;

    int8_t ndim = 8;
    int64_t shape[] = {8, 8, 7, 7, 6, 7, 5, 7};
    int64_t cshape[] = {4, 5, 2, 5, 3, 4, 5, 2};
    int64_t bshape[] = {2, 2, 2, 3, 2, 1, 2, 1};
    int8_t axis = 6;

    int64_t dest_cshape[] = {4, 5, 2, 5, 3, 4, 2};
    int64_t dest_bshape[] = {2, 2, 2, 3, 2, 1, 1};
    bool dest_frame = false;
    char *dest_urlpath = NULL;

    INA_TEST_ASSERT_SUCCEED(test_reduce(data->ctx, dtype, ndim, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}



INA_TEST_FIXTURE(reduce, 2_ui_1) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT32;

    int8_t ndim = 2;
    int64_t shape[] = {120, 1000};
    int64_t cshape[] = {69, 210};
    int64_t bshape[] = {31, 2};
    int8_t axis = 1;

    int64_t dest_cshape[] = {69};
    int64_t dest_bshape[] = {31};
    bool dest_frame = true;
    char *dest_urlpath = NULL;

    INA_TEST_ASSERT_SUCCEED(test_reduce(data->ctx, dtype, ndim, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}


INA_TEST_FIXTURE(reduce, 3_f_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 3;
    int64_t shape[] = {12, 12, 12};
    int64_t cshape[] = {6, 9, 6};
    int64_t bshape[] = {3, 3, 3};
    int8_t axis = 2;

    int64_t dest_cshape[] = {6, 9};
    int64_t dest_bshape[] = {3, 3};
    bool dest_frame = false;
    char *dest_urlpath = "arr.iarr";

    INA_TEST_ASSERT_SUCCEED(test_reduce(data->ctx, dtype, ndim, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}

INA_TEST_FIXTURE(reduce, 4_us_0) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT16;

    int8_t ndim = 4;
    int64_t shape[] = {30, 10, 15, 10};
    int64_t cshape[] = {16, 3, 1, 10};
    int64_t bshape[] = {3, 3, 1, 5};
    int8_t axis = 0;

    int64_t dest_cshape[] = {3, 1, 10};
    int64_t dest_bshape[] = {3, 1, 5};
    bool dest_frame = false;
    char *dest_urlpath = NULL;

    INA_TEST_ASSERT_SUCCEED(test_reduce(data->ctx, dtype, ndim, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}


INA_TEST_FIXTURE(reduce, 8_ll_6) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT64;

    int8_t ndim = 8;
    int64_t shape[] = {8, 8, 7, 7, 6, 7, 5, 7};
    int64_t cshape[] = {4, 5, 2, 5, 3, 4, 5, 2};
    int64_t bshape[] = {2, 2, 2, 3, 2, 1, 2, 1};
    int8_t axis = 6;

    int64_t dest_cshape[] = {4, 5, 2, 5, 3, 4, 2};
    int64_t dest_bshape[] = {2, 2, 2, 3, 2, 1, 1};
    bool dest_frame = true;
    char *dest_urlpath = "arr.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce(data->ctx, dtype, ndim, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}


INA_TEST_FIXTURE(reduce, 2_sc_0) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT8;

    int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {5, 5};
    int64_t bshape[] = {2, 2};
    int8_t axis = 0;

    int64_t dest_cshape[] = {5, 1};
    int64_t dest_bshape[] = {2, 1};
    bool dest_frame = true;
    char *dest_urlpath = "arr.iarr";

    INA_TEST_ASSERT_SUCCEED(test_reduce(data->ctx, dtype, ndim, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}


INA_TEST_FIXTURE(reduce, 4_uc_3) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT8;

    int8_t ndim = 4;
    int64_t shape[] = {8, 8, 7, 7};
    int64_t cshape[] = {4, 5, 2, 5};
    int64_t bshape[] = {2, 2, 2, 3};
    int8_t axis = 3;

    int64_t dest_cshape[] = {4, 5, 2};
    int64_t dest_bshape[] = {2, 2, 2};
    bool dest_frame = false;
    char *dest_urlpath = NULL;

    INA_TEST_ASSERT_SUCCEED(test_reduce(data->ctx, dtype, ndim, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}

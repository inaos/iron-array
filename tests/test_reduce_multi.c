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


static ina_rc_t test_reduce_multi(iarray_context_t *ctx, iarray_data_type_t dtype, int8_t ndim,
                               const int64_t *shape, const int64_t *cshape, const int64_t *bshape,
                               int8_t naxis, int8_t *axis,
                               int64_t *dest_cshape, int64_t *dest_bshape, bool dest_frame,
                               char* dest_filename)
{
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
    storage.backend = IARRAY_STORAGE_BLOSC;
    for (int i = 0; i < ndim; ++i) {
        if (cshape != NULL) {
            storage.chunkshape[i] = cshape[i];
            storage.blockshape[i] = bshape[i];
        }
    }

    iarray_container_t *c_x;
    IARRAY_RETURN_IF_FAILED(iarray_ones(ctx, &dtshape, &storage, 0, &c_x));


    iarray_storage_t dest_storage = {0};
    dest_storage.backend = IARRAY_STORAGE_BLOSC;
    dest_storage.enforce_frame = dest_frame;
    dest_storage.filename = dest_filename;
    for (int i = 0; i < ndim - naxis; ++i) {
        dest_storage.blockshape[i] = dest_bshape[i];
        dest_storage.chunkshape[i] = dest_cshape[i];
    }

    iarray_container_t *c_z;

    IARRAY_RETURN_IF_FAILED(iarray_reduce_multi(ctx, c_x, IARRAY_REDUCE_SUM, naxis, axis,
                                                &dest_storage, &c_z));

    int64_t buffer_nitems = c_z->catarr->nitems;
    int64_t buffer_size = buffer_nitems * c_z->catarr->itemsize;
    uint8_t *buffer = malloc(buffer_size);

    IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, c_z, buffer, buffer_size));

    double val = 1;
    for (int i = 0; i < naxis; ++i) {
        val *= shape[axis[i]];
    }

    for (int i = 0; i < buffer_nitems; ++i) {
        // printf("%d: %f - %f\n", i, ((double *) buffer)[i], val);
        switch (c_z->dtshape->dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                INA_TEST_ASSERT_EQUAL_FLOATING(((double *) buffer)[i], val);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                INA_TEST_ASSERT_EQUAL_FLOATING(((float *) buffer)[i], val);
                break;
            default:
                IARRAY_TRACE1(iarray.error, "Invalid dtype");
                return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
        }
    }

    iarray_container_free(ctx, &c_z);
    iarray_container_free(ctx, &c_x);

    if (dest_filename) {
        remove(dest_filename);
    }
    return INA_SUCCESS;
}

INA_TEST_DATA(reduce_multi) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(reduce_multi) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(reduce_multi) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}


INA_TEST_FIXTURE(reduce_multi, 2_d_1) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 2;
    int64_t shape[] = {120, 1000};
    int64_t cshape[] = {69, 210};
    int64_t bshape[] = {31, 2};
    int8_t naxis = 1;
    int8_t axis[] = {1};

    int64_t dest_cshape[] = {50};
    int64_t dest_bshape[] = {31};
    bool dest_frame = false;
    char *dest_filename = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, dest_frame,
                                              dest_filename));
}


INA_TEST_FIXTURE(reduce_multi, 3_d_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 3;
    int64_t shape[] = {12, 12, 12};
    int64_t cshape[] = {6, 9, 6};
    int64_t bshape[] = {3, 3, 3};
    int8_t naxis = 3;
    int8_t axis[] = {0, 2, 1};

    int64_t dest_cshape[] = {};
    int64_t dest_bshape[] = {};
    bool dest_frame = false;
    char *dest_filename = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, dest_frame,
                                              dest_filename));
}

INA_TEST_FIXTURE(reduce_multi, 4_d_0) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 4;
    int64_t shape[] = {52, 21, 27, 109};
    int64_t cshape[] = {16, 3, 1, 109};
    int64_t bshape[] = {3, 3, 1, 25};
    int8_t naxis = 2;
    int8_t axis[] = {0, 3};

    int64_t dest_cshape[] = {3, 1};
    int64_t dest_bshape[] = {3, 1};
    bool dest_frame = true;
    char *dest_filename = "iarray_reduce.iarray";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, dest_frame,
                                              dest_filename));
}


INA_TEST_FIXTURE(reduce_multi, 8_d_6) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 8;
    int64_t shape[] = {8, 8, 7, 7, 6, 7, 5, 7};
    int64_t cshape[] = {4, 5, 2, 5, 3, 4, 5, 2};
    int64_t bshape[] = {2, 2, 2, 3, 2, 1, 2, 1};
    int8_t naxis = 1;
    int8_t axis[] = {7};

    int64_t dest_cshape[] = {4, 5, 2, 5, 3, 4, 5};
    int64_t dest_bshape[] = {2, 2, 2, 3, 2, 1, 2};
    bool dest_frame = false;
    char *dest_filename = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, dest_frame,
                                              dest_filename));
}



INA_TEST_FIXTURE(reduce_multi, 2_f_1) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 2;
    int64_t shape[] = {120, 1000};
    int64_t cshape[] = {69, 210};
    int64_t bshape[] = {31, 2};
    int8_t naxis = 1;
    int8_t axis[] = {0};

    int64_t dest_cshape[] = {210};
    int64_t dest_bshape[] = {2};
    bool dest_frame = false;
    char *dest_filename = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, dest_frame,
                                              dest_filename));
}


INA_TEST_FIXTURE(reduce_multi, 3_f_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 3;
    int64_t shape[] = {12, 12, 12};
    int64_t cshape[] = {6, 9, 6};
    int64_t bshape[] = {3, 3, 3};
    int8_t naxis = 2;
    int8_t axis[] = {0, 1};

    int64_t dest_cshape[] = {6};
    int64_t dest_bshape[] = {3};
    bool dest_frame = false;
    char *dest_filename = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, dest_frame,
                                              dest_filename));
}

INA_TEST_FIXTURE(reduce_multi, 4_f_0) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 4;
    int64_t shape[] = {52, 21, 27, 109};
    int64_t cshape[] = {16, 3, 1, 109};
    int64_t bshape[] = {3, 3, 1, 25};
    int8_t naxis = 1;
    int8_t axis[] = {3};

    int64_t dest_cshape[] = {16, 3, 1};
    int64_t dest_bshape[] = {3, 3, 1};
    bool dest_frame = false;
    char *dest_filename = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, dest_frame,
                                              dest_filename));
}


INA_TEST_FIXTURE(reduce_multi, 8_f_6) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 8;
    int64_t shape[] = {8, 8, 7, 7, 6, 7, 5, 7};
    int64_t cshape[] = {4, 5, 2, 5, 3, 4, 5, 2};
    int64_t bshape[] = {2, 2, 2, 3, 2, 1, 2, 1};
    int8_t naxis = 7;
    int8_t axis[] = {1, 2, 7, 5, 3, 4, 0};

    int64_t dest_cshape[] = {5};
    int64_t dest_bshape[] = {2};
    bool dest_frame = false;
    char *dest_filename = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, dest_frame,
                                              dest_filename));
}

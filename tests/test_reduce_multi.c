/*
 * Copyright ironArray SL 2021.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of ironArray SL
 * ("Confidential Information"). You shall not disclose such Confidential
 * Information and shall use it only in accordance with the terms of the license agreement.
 *
 */

#include <libiarray/iarray.h>
#include <src/iarray_private.h>


static ina_rc_t test_reduce_multi(iarray_context_t *ctx, iarray_data_type_t dtype, int8_t ndim, iarray_reduce_func_t func,
                               const int64_t *shape, const int64_t *cshape, const int64_t *bshape,
                               int8_t naxis, int8_t *axis,
                               const int64_t *dest_cshape, const int64_t *dest_bshape, bool src_contiguous, char *src_urlpath,
                               bool dest_contiguous, char* dest_urlpath, bool oneshot)
{
    blosc2_remove_urlpath(src_urlpath);
    blosc2_remove_urlpath(dest_urlpath);

    // Create dtshape
    iarray_dtshape_t dtshape;

    dtshape.dtype = dtype;
    dtshape.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
    }

    iarray_storage_t storage = {0};
    storage.contiguous = src_contiguous;
    storage.urlpath = src_urlpath;
    for (int i = 0; i < ndim; ++i) {
        storage.chunkshape[i] = cshape[i];
        storage.blockshape[i] = bshape[i];
    }

    iarray_container_t *c_x;
    IARRAY_RETURN_IF_FAILED(iarray_ones(ctx, &dtshape, &storage, &c_x));


    iarray_storage_t dest_storage = {0};
    dest_storage.contiguous = dest_contiguous;
    dest_storage.urlpath = dest_urlpath;
    for (int i = 0; i < ndim - naxis; ++i) {
        dest_storage.blockshape[i] = dest_bshape[i];
        dest_storage.chunkshape[i] = dest_cshape[i];
    }

    iarray_container_t *c_z;

    IARRAY_RETURN_IF_FAILED(iarray_reduce_multi(ctx, c_x, func, naxis, axis,
                                                &dest_storage, &c_z, oneshot, 0.0));

    int64_t buffer_nitems = c_z->catarr->nitems;
    int64_t buffer_size = buffer_nitems * c_z->catarr->itemsize;
    uint8_t *buffer = malloc(buffer_size);

    IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, c_z, buffer, buffer_size));

    double val = 1;

    switch (func) {
        case IARRAY_REDUCE_MAX:
        case IARRAY_REDUCE_MIN:
        case IARRAY_REDUCE_PROD:
        case IARRAY_REDUCE_MEAN:
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
                    case IARRAY_DATA_TYPE_INT32:
                        INA_TEST_ASSERT_EQUAL_INT(((int32_t *) buffer)[i], val);
                        break;
                    case IARRAY_DATA_TYPE_INT16:
                        INA_TEST_ASSERT_EQUAL_INT(((int16_t *) buffer)[i], val);
                        break;
                    case IARRAY_DATA_TYPE_INT8:
                        INA_TEST_ASSERT_EQUAL_INT(((int8_t *) buffer)[i], val);
                        break;
                    case IARRAY_DATA_TYPE_UINT64:
                        INA_TEST_ASSERT_EQUAL_UINT64(((uint64_t *) buffer)[i], val);
                        break;
                    case IARRAY_DATA_TYPE_UINT32:
                        INA_TEST_ASSERT_EQUAL_UINT(((uint32_t *) buffer)[i], val);
                        break;
                    case IARRAY_DATA_TYPE_UINT16:
                        INA_TEST_ASSERT_EQUAL_UINT(((uint16_t *) buffer)[i], val);
                        break;
                    case IARRAY_DATA_TYPE_UINT8:
                        INA_TEST_ASSERT_EQUAL_UINT(((uint8_t *) buffer)[i], val);
                        break;
                    case IARRAY_DATA_TYPE_BOOL:
                        INA_TEST_ASSERT(((bool *) buffer)[i] == val);
                        break;
                    default:
                        IARRAY_TRACE1(iarray.error, "Invalid dtype");
                        return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
                }
            }
            break;
        case IARRAY_REDUCE_SUM: {
            for (int i = 0; i < naxis; ++i) {
                val *= (double)shape[axis[i]];
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
            break;
        }
        default:
            return INA_ERR_EXCEEDED;
    }

    iarray_container_free(ctx, &c_z);
    iarray_container_free(ctx, &c_x);

    blosc2_remove_urlpath(dest_urlpath);
    blosc2_remove_urlpath(src_urlpath);

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


INA_TEST_FIXTURE(reduce_multi, sum_2_d_1) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 2;
    int64_t shape[] = {120, 1000};
    int64_t cshape[] = {69, 210};
    int64_t bshape[] = {31, 2};
    int8_t naxis = 1;
    int8_t axis[] = {1};

    int64_t dest_cshape[] = {50};
    int64_t dest_bshape[] = {31};
    bool src_contiguous = true;
    char *src_urlpath = NULL;
    bool dest_contiguous = false;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_SUM, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi, prod_3_f_3) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 3;
    int64_t shape[] = {12, 12, 12};
    int64_t cshape[] = {6, 9, 6};
    int64_t bshape[] = {3, 3, 3};
    int8_t naxis = 3;
    int8_t axis[] = {0, 2, 1};

    int64_t dest_cshape[] = {0};  // {} not compile on Windows
    int64_t dest_bshape[] = {0};  // {} not compile on Windows
    bool src_contiguous = true;
    char *src_urlpath = "srcarr.iarr";
    bool dest_contiguous = false;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_PROD, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi, sum_4_ll_2_oneshot) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT64;

    int8_t ndim = 4;
    int64_t shape[] = {52, 21, 27, 109};
    int64_t cshape[] = {16, 3, 1, 109};
    int64_t bshape[] = {3, 3, 1, 25};
    int8_t naxis = 2;
    int8_t axis[] = {0, 3};

    int64_t dest_cshape[] = {3, 1};
    int64_t dest_bshape[] = {3, 1};
    bool src_contiguous = false;
    char *src_urlpath = NULL;
    bool dest_contiguous = true;
    char *dest_urlpath = "iarray_reduce.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_SUM, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, true));
}


INA_TEST_FIXTURE(reduce_multi, prod_5_i_1_oneshot) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT32;

    int8_t ndim = 5;
    int64_t shape[] = {8, 8, 7, 7, 6};
    int64_t cshape[] = {4, 5, 2, 5, 3};
    int64_t bshape[] = {2, 2, 2, 3, 2};
    int8_t naxis = 1;
    int8_t axis[] = {4};

    int64_t dest_cshape[] = {4, 5, 2, 5};
    int64_t dest_bshape[] = {2, 2, 2, 3};
    bool src_contiguous = false;
    char *src_urlpath = "srcarr.iarr";
    bool dest_contiguous = false;
    char *dest_urlpath = "destarr.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_PROD, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, true));
}


/* Avoid heavy tests
INA_TEST_FIXTURE(reduce_multi, prod_8_i_1) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT32;

    int8_t ndim = 8;
    int64_t shape[] = {8, 8, 7, 7, 6, 7, 5, 7};
    int64_t cshape[] = {4, 5, 2, 5, 3, 4, 5, 2};
    int64_t bshape[] = {2, 2, 2, 3, 2, 1, 2, 1};
    int8_t naxis = 1;
    int8_t axis[] = {7};

    int64_t dest_cshape[] = {4, 5, 2, 5, 3, 4, 5};
    int64_t dest_bshape[] = {2, 2, 2, 3, 2, 1, 2};
    bool src_contiguous = false;
    char *src_urlpath = "srcarr.iarr";
    bool dest_contiguous = false;
    char *dest_urlpath = "destarr.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_PROD, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath));
}
*/


INA_TEST_FIXTURE(reduce_multi, sum_2_ui_1) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT32;

    int8_t ndim = 2;
    int64_t shape[] = {120, 1000};
    int64_t cshape[] = {69, 210};
    int64_t bshape[] = {31, 2};
    int8_t naxis = 1;
    int8_t axis[] = {0};

    int64_t dest_cshape[] = {210};
    int64_t dest_bshape[] = {2};
    bool src_contiguous = true;
    char *src_urlpath = NULL;
    bool dest_contiguous = false;
    char *dest_urlpath = "destarr.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_SUM, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi, prod_3_s_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT16;

    int8_t ndim = 3;
    int64_t shape[] = {12, 12, 12};
    int64_t cshape[] = {6, 9, 6};
    int64_t bshape[] = {3, 3, 3};
    int8_t naxis = 2;
    int8_t axis[] = {0, 1};

    int64_t dest_cshape[] = {6};
    int64_t dest_bshape[] = {3};
    bool src_contiguous = false;
    char *src_urlpath = NULL;
    bool dest_contiguous = true;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_PROD, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi, sum_4_ui_1_oneshot) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT32;

    int8_t ndim = 4;
    int64_t shape[] = {52, 21, 27, 109};
    int64_t cshape[] = {16, 3, 1, 109};
    int64_t bshape[] = {3, 3, 1, 25};
    int8_t naxis = 1;
    int8_t axis[] = {3};

    int64_t dest_cshape[] = {16, 3, 1};
    int64_t dest_bshape[] = {3, 3, 1};
    bool src_contiguous = true;
    char *src_urlpath = NULL;
    bool dest_contiguous = true;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_SUM, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, true));
}


INA_TEST_FIXTURE(reduce_multi, prod_4_ull_3_oneshot) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT64;

    int8_t ndim = 4;
    int64_t shape[] = {8, 8, 7, 7};
    int64_t cshape[] = {4, 5, 2, 5};
    int64_t bshape[] = {2, 2, 2, 3};
    int8_t naxis = 3;
    int8_t axis[] = {1, 2, 0};

    int64_t dest_cshape[] = {5};
    int64_t dest_bshape[] = {2};
    bool src_contiguous = true;
    char *src_urlpath = NULL;
    bool dest_contiguous = false;
    char *dest_urlpath = "iarray_reduce.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_PROD, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, true));
}


/* Avoid heavy tests
INA_TEST_FIXTURE(reduce_multi, prod_8_ull_7) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT64;

    int8_t ndim = 8;
    int64_t shape[] = {8, 8, 7, 7, 6, 7, 5, 7};
    int64_t cshape[] = {4, 5, 2, 5, 3, 4, 5, 2};
    int64_t bshape[] = {2, 2, 2, 3, 2, 1, 2, 1};
    int8_t naxis = 7;
    int8_t axis[] = {1, 2, 7, 5, 3, 4, 0};

    int64_t dest_cshape[] = {5};
    int64_t dest_bshape[] = {2};
    bool src_contiguous = true;
    char *src_urlpath = NULL;
    bool dest_contiguous = false;
    char *dest_urlpath = "iarray_reduce.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_PROD, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath));
}
*/


INA_TEST_FIXTURE(reduce_multi, sum_2_sc_1) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT8;

    int8_t ndim = 2;
    int64_t shape[] = {12, 10};
    int64_t cshape[] = {69, 210};
    int64_t bshape[] = {31, 2};
    int8_t naxis = 1;
    int8_t axis[] = {0};

    int64_t dest_cshape[] = {210};
    int64_t dest_bshape[] = {2};
    bool src_contiguous = true;
    char *src_urlpath = NULL;
    bool dest_contiguous = false;
    char *dest_urlpath = "destarr.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_SUM, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi, prod_2_uc_1) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT8;

    int8_t ndim = 2;
    int64_t shape[] = {12, 12};
    int64_t cshape[] = {6, 9};
    int64_t bshape[] = {3, 3};
    int8_t naxis = 1;
    int8_t axis[] = {0};

    int64_t dest_cshape[] = {6};
    int64_t dest_bshape[] = {3};
    bool src_contiguous = false;
    char *src_urlpath = NULL;
    bool dest_contiguous = true;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_PROD, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi, sum_2_b_1_oneshot) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_BOOL;

    int8_t ndim = 2;
    int64_t shape[] = {12, 12};
    int64_t cshape[] = {6, 9};
    int64_t bshape[] = {3, 3};
    int8_t naxis = 1;
    int8_t axis[] = {0};

    int64_t dest_cshape[] = {6};
    int64_t dest_bshape[] = {3};
    bool src_contiguous = false;
    char *src_urlpath = NULL;
    bool dest_contiguous = true;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_SUM, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, true));
}


INA_TEST_FIXTURE(reduce_multi, min_2_d_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 2;
    int64_t shape[] = {12, 10};
    int64_t cshape[] = {6, 2};
    int64_t bshape[] = {3, 2};
    int8_t naxis = 2;
    int8_t axis[] = {1, 0};

    int64_t dest_cshape[] = {1};
    int64_t dest_bshape[] = {1};
    bool src_contiguous = true;
    char *src_urlpath = NULL;
    bool dest_contiguous = false;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_MIN, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi, max_3_f_3) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 3;
    int64_t shape[] = {12, 12, 12};
    int64_t cshape[] = {6, 9, 6};
    int64_t bshape[] = {3, 3, 3};
    int8_t naxis = 3;
    int8_t axis[] = {0, 2, 1};

    int64_t dest_cshape[] = {1};  // {} not compile on Windows
    int64_t dest_bshape[] = {1};  // {} not compile on Windows
    bool src_contiguous = true;
    char *src_urlpath = "srcarr.iarr";
    bool dest_contiguous = false;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_MAX, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi, min_4_ll_2_oneshot) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT64;

    int8_t ndim = 4;
    int64_t shape[] = {5, 21, 27, 10};
    int64_t cshape[] = {4, 3, 1, 10};
    int64_t bshape[] = {3, 3, 1, 5};
    int8_t naxis = 2;
    int8_t axis[] = {0, 3};

    int64_t dest_cshape[] = {3, 1};
    int64_t dest_bshape[] = {3, 1};
    bool src_contiguous = false;
    char *src_urlpath = NULL;
    bool dest_contiguous = true;
    char *dest_urlpath = "iarray_reduce.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_MIN, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, true));
}


INA_TEST_FIXTURE(reduce_multi, max_3_i_1_oneshot) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT32;

    int8_t ndim = 3;
    int64_t shape[] = {8, 8, 7};
    int64_t cshape[] = {4, 5, 2};
    int64_t bshape[] = {2, 2, 2};
    int8_t naxis = 1;
    int8_t axis[] = {2};

    int64_t dest_cshape[] = {4, 5};
    int64_t dest_bshape[] = {2, 2};
    bool src_contiguous = false;
    char *src_urlpath = "srcarr.iarr";
    bool dest_contiguous = false;
    char *dest_urlpath = "destarr.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_MAX, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, true));
}


/* Avoid heavy tests
INA_TEST_FIXTURE(reduce_multi, max_8_i_1) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT32;

    int8_t ndim = 8;
    int64_t shape[] = {8, 8, 7, 7, 6, 7, 5, 7};
    int64_t cshape[] = {4, 5, 2, 5, 3, 4, 5, 2};
    int64_t bshape[] = {2, 2, 2, 3, 2, 1, 2, 1};
    int8_t naxis = 1;
    int8_t axis[] = {7};

    int64_t dest_cshape[] = {4, 5, 2, 5, 3, 4, 5};
    int64_t dest_bshape[] = {2, 2, 2, 3, 2, 1, 2};
    bool src_contiguous = false;
    char *src_urlpath = "srcarr.iarr";
    bool dest_contiguous = false;
    char *dest_urlpath = "destarr.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_MAX, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath));
}
*/


INA_TEST_FIXTURE(reduce_multi, min_2_ui_1) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT32;

    int8_t ndim = 2;
    int64_t shape[] = {12, 10};
    int64_t cshape[] = {6, 2};
    int64_t bshape[] = {3, 2};
    int8_t naxis = 1;
    int8_t axis[] = {0};

    int64_t dest_cshape[] = {6};
    int64_t dest_bshape[] = {2};
    bool src_contiguous = true;
    char *src_urlpath = NULL;
    bool dest_contiguous = false;
    char *dest_urlpath = "destarr.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_MIN, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi, max_3_s_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT16;

    int8_t ndim = 3;
    int64_t shape[] = {12, 12, 12};
    int64_t cshape[] = {6, 9, 6};
    int64_t bshape[] = {3, 3, 3};
    int8_t naxis = 2;
    int8_t axis[] = {0, 1};

    int64_t dest_cshape[] = {6};
    int64_t dest_bshape[] = {3};
    bool src_contiguous = false;
    char *src_urlpath = NULL;
    bool dest_contiguous = true;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_MAX, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi, min_4_ui_4_oneshot) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT32;

    int8_t ndim = 4;
    int64_t shape[] = {5, 21, 27, 10};
    int64_t cshape[] = {4, 3, 1, 10};
    int64_t bshape[] = {3, 3, 1, 5};
    int8_t naxis = 4;
    int8_t axis[] = {3, 1, 2, 0};

    int64_t dest_cshape[] = {1};
    int64_t dest_bshape[] = {1};
    bool src_contiguous = true;
    char *src_urlpath = NULL;
    bool dest_contiguous = true;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_MIN, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, true));
}


INA_TEST_FIXTURE(reduce_multi, max_5_ull_4_oneshot) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT64;

    int8_t ndim = 5;
    int64_t shape[] = {8, 8, 7, 7, 6};
    int64_t cshape[] = {4, 5, 2, 5, 3};
    int64_t bshape[] = {2, 2, 2, 3, 2};
    int8_t naxis = 4;
    int8_t axis[] = {1, 2, 4, 0};

    int64_t dest_cshape[] = {5};
    int64_t dest_bshape[] = {2};
    bool src_contiguous = true;
    char *src_urlpath = NULL;
    bool dest_contiguous = false;
    char *dest_urlpath = "iarray_reduce.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_MAX, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, true));
}


/* Avoid heavy tests
INA_TEST_FIXTURE(reduce_multi, max_8_ull_7) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT64;

    int8_t ndim = 8;
    int64_t shape[] = {8, 8, 7, 7, 6, 7, 5, 7};
    int64_t cshape[] = {4, 5, 2, 5, 3, 4, 5, 2};
    int64_t bshape[] = {2, 2, 2, 3, 2, 1, 2, 1};
    int8_t naxis = 7;
    int8_t axis[] = {1, 2, 7, 5, 3, 4, 0};

    int64_t dest_cshape[] = {5};
    int64_t dest_bshape[] = {2};
    bool src_contiguous = true;
    char *src_urlpath = NULL;
    bool dest_contiguous = false;
    char *dest_urlpath = "iarray_reduce.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_MAX, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath));
}
*/


INA_TEST_FIXTURE(reduce_multi, min_2_sc_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT8;

    int8_t ndim = 2;
    int64_t shape[] = {12, 10};
    int64_t cshape[] = {69, 210};
    int64_t bshape[] = {31, 2};
    int8_t naxis = 2;
    int8_t axis[] = {1, 0};

    int64_t dest_cshape[] = {1};
    int64_t dest_bshape[] = {1};
    bool src_contiguous = true;
    char *src_urlpath = NULL;
    bool dest_contiguous = false;
    char *dest_urlpath = "destarr.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_MIN, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi, max_2_uc_1) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT8;

    int8_t ndim = 2;
    int64_t shape[] = {12, 12};
    int64_t cshape[] = {6, 9};
    int64_t bshape[] = {3, 3};
    int8_t naxis = 1;
    int8_t axis[] = {0};

    int64_t dest_cshape[] = {6};
    int64_t dest_bshape[] = {3};
    bool src_contiguous = false;
    char *src_urlpath = NULL;
    bool dest_contiguous = true;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_MAX, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi, min_2_b_2_oneshot) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_BOOL;

    int8_t ndim = 2;
    int64_t shape[] = {12, 12};
    int64_t cshape[] = {6, 9};
    int64_t bshape[] = {3, 3};
    int8_t naxis = 2;
    int8_t axis[] = {0, 1};

    int64_t dest_cshape[] = {1};
    int64_t dest_bshape[] = {1};
    bool src_contiguous = false;
    char *src_urlpath = NULL;
    bool dest_contiguous = true;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_MIN, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, true));
}


INA_TEST_FIXTURE(reduce_multi, mean_2_d_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 2;
    int64_t shape[] = {12, 100};
    int64_t cshape[] = {6, 21};
    int64_t bshape[] = {3, 2};
    int8_t naxis = 2;
    int8_t axis[] = {1, 0};

    int64_t dest_cshape[] = {1};
    int64_t dest_bshape[] = {1};
    bool src_contiguous = true;
    char *src_urlpath = NULL;
    bool dest_contiguous = false;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_MEAN, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi, mean_3_f_3_oneshot) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 3;
    int64_t shape[] = {12, 12, 12};
    int64_t cshape[] = {6, 9, 6};
    int64_t bshape[] = {3, 3, 3};
    int8_t naxis = 3;
    int8_t axis[] = {0, 2, 1};

    int64_t dest_cshape[] = {1};  // {} not compile on Windows
    int64_t dest_bshape[] = {1};  // {} not compile on Windows
    bool src_contiguous = true;
    char *src_urlpath = "srcarr.iarr";
    bool dest_contiguous = false;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_MEAN, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, true));
}


INA_TEST_FIXTURE(reduce_multi, mean_4_ll_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT64;

    int8_t ndim = 4;
    int64_t shape[] = {5, 21, 27, 10};
    int64_t cshape[] = {4, 3, 1, 10};
    int64_t bshape[] = {3, 3, 1, 5};
    int8_t naxis = 2;
    int8_t axis[] = {0, 3};

    int64_t dest_cshape[] = {3, 1};
    int64_t dest_bshape[] = {3, 1};
    bool src_contiguous = false;
    char *src_urlpath = NULL;
    bool dest_contiguous = true;
    char *dest_urlpath = "iarray_reduce.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_MEAN, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi, mean_2_ui_1_oneshot) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT32;

    int8_t ndim = 2;
    int64_t shape[] = {12, 10};
    int64_t cshape[] = {6, 2};
    int64_t bshape[] = {3, 2};
    int8_t naxis = 1;
    int8_t axis[] = {0};

    int64_t dest_cshape[] = {6};
    int64_t dest_bshape[] = {2};
    bool src_contiguous = true;
    char *src_urlpath = NULL;
    bool dest_contiguous = false;
    char *dest_urlpath = "destarr.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_MEAN, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, true));
}


INA_TEST_FIXTURE(reduce_multi, mean_3_s_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT16;

    int8_t ndim = 3;
    int64_t shape[] = {12, 12, 12};
    int64_t cshape[] = {6, 9, 6};
    int64_t bshape[] = {3, 3, 3};
    int8_t naxis = 2;
    int8_t axis[] = {0, 1};

    int64_t dest_cshape[] = {6};
    int64_t dest_bshape[] = {3};
    bool src_contiguous = false;
    char *src_urlpath = NULL;
    bool dest_contiguous = true;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_MEAN, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi, mean_2_sc_2_oneshot) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT8;

    int8_t ndim = 2;
    int64_t shape[] = {12, 10};
    int64_t cshape[] = {69, 210};
    int64_t bshape[] = {31, 2};
    int8_t naxis = 2;
    int8_t axis[] = {1, 0};

    int64_t dest_cshape[] = {1};
    int64_t dest_bshape[] = {1};
    bool src_contiguous = true;
    char *src_urlpath = NULL;
    bool dest_contiguous = false;
    char *dest_urlpath = "destarr.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_MEAN, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, true));
}


INA_TEST_FIXTURE(reduce_multi, mean_2_b_1) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_BOOL;

    int8_t ndim = 2;
    int64_t shape[] = {12, 12};
    int64_t cshape[] = {6, 9};
    int64_t bshape[] = {3, 3};
    int8_t naxis = 1;
    int8_t axis[] = {0};

    int64_t dest_cshape[] = {6};
    int64_t dest_bshape[] = {3};
    bool src_contiguous = false;
    char *src_urlpath = NULL;
    bool dest_contiguous = true;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi(data->ctx, dtype, ndim, IARRAY_REDUCE_MEAN, shape, cshape, bshape,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}

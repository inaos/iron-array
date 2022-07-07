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


static ina_rc_t test_reduce_multi_type(iarray_context_t *ctx, iarray_data_type_t dtype, iarray_data_type_t view_dtype, int8_t ndim, iarray_reduce_func_t func,
                               const int64_t *shape, const int64_t *cshape, const int64_t *bshape,
                               const int64_t *view_start, const int64_t *view_stop,
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

    iarray_container_t *c_view;
    IARRAY_RETURN_IF_FAILED(iarray_get_type_view(ctx, c_x, view_dtype, &c_view));

    iarray_container_t *c_slice;
    IARRAY_RETURN_IF_FAILED(iarray_get_slice(ctx, c_view, view_start, view_stop, true, NULL, &c_slice));

    iarray_storage_t dest_storage = {0};
    dest_storage.contiguous = dest_contiguous;
    dest_storage.urlpath = dest_urlpath;
    for (int i = 0; i < ndim - naxis; ++i) {
        dest_storage.blockshape[i] = dest_bshape[i];
        dest_storage.chunkshape[i] = dest_cshape[i];
    }

    iarray_container_t *c_z;
    IARRAY_RETURN_IF_FAILED(iarray_reduce_multi(ctx, c_slice, func, naxis, axis,
                                                &dest_storage, &c_z, oneshot));

    int64_t buffer_nitems = c_z->catarr->nitems;
    int64_t buffer_size = buffer_nitems * c_z->catarr->itemsize;
    uint8_t *buffer = ina_mem_alloc(buffer_size);

    IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, c_z, buffer, buffer_size));

    double val = 1;

    switch (func) {
        case IARRAY_REDUCE_MAX:
        case IARRAY_REDUCE_MIN:
        case IARRAY_REDUCE_PROD:
        case IARRAY_REDUCE_MEAN:
            for (int i = 0; i < buffer_nitems; ++i) {
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
                val *= (double)c_slice->dtshape->shape[axis[i]];
            }
            for (int i = 0; i < buffer_nitems; ++i) {
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
    iarray_container_free(ctx, &c_slice);
    iarray_container_free(ctx, &c_view);

    ina_mem_free(buffer);

    blosc2_remove_urlpath(dest_urlpath);
    blosc2_remove_urlpath(src_urlpath);

    return INA_SUCCESS;
}

INA_TEST_DATA(reduce_multi_type) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(reduce_multi_type) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.max_num_threads = 1;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(reduce_multi_type) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

/*
INA_TEST_FIXTURE(reduce_multi_type, prod_3_f_ll_3) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_INT64;

    int8_t ndim = 3;
    int64_t shape[] = {12, 12, 12};
    int64_t cshape[] = {6, 9, 6};
    int64_t bshape[] = {3, 3, 3};
    int8_t naxis = 3;
    int8_t axis[] = {0, 2, 1};
    
   int64_t view_start[IARRAY_DIMENSION_MAX] = {0};
    int64_t view_stop[] = {8, 12, 7};
    
    int64_t dest_cshape[] = {0};  // {} not compile on Windows
    int64_t dest_bshape[] = {0};  // {} not compile on Windows
    bool src_contiguous = true;
    char *src_urlpath = "srcarr.iarr";
    bool dest_contiguous = false;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_PROD, shape, cshape, bshape, view_start, view_stop,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath));
}
*/

INA_TEST_FIXTURE(reduce_multi_type, sum_4_ll_d_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT64;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 4;
    int64_t shape[] = {52, 21, 27, 109};
    int64_t cshape[] = {16, 3, 1, 109};
    int64_t bshape[] = {3, 3, 1, 25};
    int8_t naxis = 2;
    int8_t axis[] = {0, 3};

    int64_t view_start[] = {12, 12, 12, 0};
    int64_t view_stop[] = {34, 15, 13, 109};
    
    int64_t dest_cshape[] = {3, 1};
    int64_t dest_bshape[] = {3, 1};
    bool src_contiguous = false;
    char *src_urlpath = NULL;
    bool dest_contiguous = true;
    char *dest_urlpath = "iarray_reduce.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_SUM, shape, cshape, bshape, view_start, view_stop,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi_type, prod_3_i_f_1) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT32;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 3;
    int64_t shape[] = {8, 8, 7};
    int64_t cshape[] = {4, 5, 2};
    int64_t bshape[] = {2, 2, 2};
    int8_t naxis = 1;
    int8_t axis[] = {2};

    int64_t view_start[IARRAY_DIMENSION_MAX] = {0};
    int64_t view_stop[] = {4, 6, 7};

    int64_t dest_cshape[] = {4, 5};
    int64_t dest_bshape[] = {2, 2};
    bool src_contiguous = false;
    char *src_urlpath = "srcarr.iarr";
    bool dest_contiguous = false;
    char *dest_urlpath = "destarr.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_PROD, shape, cshape, bshape, view_start, view_stop,
                                                   naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                                   dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi_type, sum_2_ui_ll_1_oneshot) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT32;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_INT64;

    int8_t ndim = 2;
    int64_t shape[] = {120, 1000};
    int64_t cshape[] = {69, 210};
    int64_t bshape[] = {31, 2};
    int8_t naxis = 1;
    int8_t axis[] = {0};

    int64_t view_start[] = {10, 10};
    int64_t view_stop[] = {110, 300};
    
    int64_t dest_cshape[] = {210};
    int64_t dest_bshape[] = {2};
    bool src_contiguous = true;
    char *src_urlpath = NULL;
    bool dest_contiguous = false;
    char *dest_urlpath = "destarr.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_SUM, shape, cshape, bshape, view_start, view_stop,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, true));
}


INA_TEST_FIXTURE(reduce_multi_type, prod_3_s_f_2_oneshot) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT16;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 3;
    int64_t shape[] = {12, 12, 12};
    int64_t cshape[] = {6, 9, 6};
    int64_t bshape[] = {3, 3, 3};
    int8_t naxis = 2;
    int8_t axis[] = {0, 1};

   int64_t view_start[IARRAY_DIMENSION_MAX] = {0};
    int64_t view_stop[] = {12, 12, 12};
    
    int64_t dest_cshape[] = {6};
    int64_t dest_bshape[] = {3};
    bool src_contiguous = false;
    char *src_urlpath = NULL;
    bool dest_contiguous = true;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_PROD, shape, cshape, bshape, view_start, view_stop,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, true));
}

INA_TEST_FIXTURE(reduce_multi_type, sum_4_ui_ull_1) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT32;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_UINT64;

    int8_t ndim = 4;
    int64_t shape[] = {52, 21, 27, 109};
    int64_t cshape[] = {16, 3, 1, 109};
    int64_t bshape[] = {3, 3, 1, 25};
    int8_t naxis = 1;
    int8_t axis[] = {3};

   int64_t view_start[IARRAY_DIMENSION_MAX] = {0};
    int64_t view_stop[] = {16, 3, 1, 109};
    
    int64_t dest_cshape[] = {16, 3, 1};
    int64_t dest_bshape[] = {3, 3, 1};
    bool src_contiguous = true;
    char *src_urlpath = NULL;
    bool dest_contiguous = true;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_SUM, shape, cshape, bshape, view_start, view_stop,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi_type, prod_3_ull_ll_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT64;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_INT64;

    int8_t ndim = 3;
    int64_t shape[] = {8, 8, 7};
    int64_t cshape[] = {4, 5, 2};
    int64_t bshape[] = {2, 2, 2};
    int8_t naxis = 2;
    int8_t axis[] = {1, 2};

    int64_t view_start[IARRAY_DIMENSION_MAX] = {0};
    int64_t view_stop[] = {4, 8, 7};

    int64_t dest_cshape[] = {5};
    int64_t dest_bshape[] = {2};
    bool src_contiguous = true;
    char *src_urlpath = NULL;
    bool dest_contiguous = false;
    char *dest_urlpath = "iarray_reduce.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_PROD, shape, cshape, bshape, view_start, view_stop,
                                                   naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                                   dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi_type, sum_2_sc_b_1_oneshot) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT8;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_BOOL;

    int8_t ndim = 2;
    int64_t shape[] = {12, 10};
    int64_t cshape[] = {69, 210};
    int64_t bshape[] = {31, 2};
    int8_t naxis = 1;
    int8_t axis[] = {0};

    int64_t view_start[] = {5, 5};
    int64_t view_stop[] = {10, 10};
    
    int64_t dest_cshape[] = {210};
    int64_t dest_bshape[] = {2};
    bool src_contiguous = true;
    char *src_urlpath = NULL;
    bool dest_contiguous = false;
    char *dest_urlpath = "destarr.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_SUM, shape, cshape, bshape, view_start, view_stop,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, true));
}


INA_TEST_FIXTURE(reduce_multi_type, prod_2_uc_d_1_oneshot) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT8;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 2;
    int64_t shape[] = {12, 12};
    int64_t cshape[] = {6, 9};
    int64_t bshape[] = {3, 3};
    int8_t naxis = 1;
    int8_t axis[] = {0};

   int64_t view_start[IARRAY_DIMENSION_MAX] = {0};
    int64_t view_stop[] = {6, 6};
    
    int64_t dest_cshape[] = {6};
    int64_t dest_bshape[] = {3};
    bool src_contiguous = false;
    char *src_urlpath = NULL;
    bool dest_contiguous = true;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_PROD, shape, cshape, bshape, view_start, view_stop,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, true));
}


INA_TEST_FIXTURE(reduce_multi_type, sum_2_b_i_1) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_BOOL;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_INT32;

    int8_t ndim = 2;
    int64_t shape[] = {12, 12};
    int64_t cshape[] = {6, 9};
    int64_t bshape[] = {3, 3};
    int8_t naxis = 1;
    int8_t axis[] = {0};

    int64_t view_start[] = {2, 1};
    int64_t view_stop[] = {12, 11};
    
    int64_t dest_cshape[] = {6};
    int64_t dest_bshape[] = {3};
    bool src_contiguous = false;
    char *src_urlpath = NULL;
    bool dest_contiguous = true;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_SUM, shape, cshape, bshape, view_start, view_stop,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi_type, min_2_d_ull_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_UINT64;

    int8_t ndim = 2;
    int64_t shape[] = {12, 10};
    int64_t cshape[] = {6, 2};
    int64_t bshape[] = {3, 2};
    int8_t naxis = 2;
    int8_t axis[] = {1, 0};

   int64_t view_start[IARRAY_DIMENSION_MAX] = {0};
    int64_t view_stop[] = {10, 10};
    
    int64_t dest_cshape[] = {1};
    int64_t dest_bshape[] = {1};
    bool src_contiguous = true;
    char *src_urlpath = NULL;
    bool dest_contiguous = false;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_MIN, shape, cshape, bshape, view_start, view_stop,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi_type, max_3_f_d_3) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 3;
    int64_t shape[] = {12, 12, 12};
    int64_t cshape[] = {6, 9, 6};
    int64_t bshape[] = {3, 3, 3};
    int8_t naxis = 3;
    int8_t axis[] = {0, 2, 1};

   int64_t view_start[IARRAY_DIMENSION_MAX] = {0};
    int64_t view_stop[] = {8, 8, 8};
    
    int64_t dest_cshape[] = {1};  // {} not compile on Windows
    int64_t dest_bshape[] = {1};  // {} not compile on Windows
    bool src_contiguous = true;
    char *src_urlpath = "srcarr.iarr";
    bool dest_contiguous = false;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_MAX, shape, cshape, bshape, view_start, view_stop,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi_type, min_4_ll_d_2_oneshot) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT64;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 4;
    int64_t shape[] = {5, 21, 27, 10};
    int64_t cshape[] = {4, 3, 1, 10};
    int64_t bshape[] = {3, 3, 1, 5};
    int8_t naxis = 2;
    int8_t axis[] = {0, 3};

    int64_t view_start[] = {1, 3, 0, 1};
    int64_t view_stop[] = {5, 8, 10, 10};
    
    int64_t dest_cshape[] = {3, 1};
    int64_t dest_bshape[] = {3, 1};
    bool src_contiguous = false;
    char *src_urlpath = NULL;
    bool dest_contiguous = true;
    char *dest_urlpath = "iarray_reduce.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_MIN, shape, cshape, bshape, view_start, view_stop,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, true));
}


INA_TEST_FIXTURE(reduce_multi_type, max_5_i_ui_1_oneshot) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT32;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_UINT32;

    int8_t ndim = 5;
    int64_t shape[] = {8, 8, 7, 7, 6};
    int64_t cshape[] = {4, 5, 2, 5, 3};
    int64_t bshape[] = {2, 2, 2, 3, 2};
    int8_t naxis = 1;
    int8_t axis[] = {4};

    int64_t view_start[IARRAY_DIMENSION_MAX] = {0};
    int64_t view_stop[] = {1, 8, 7, 7, 6};

    int64_t dest_cshape[] = {4, 5, 2, 5};
    int64_t dest_bshape[] = {2, 2, 2, 3};
    bool src_contiguous = false;
    char *src_urlpath = "srcarr.iarr";
    bool dest_contiguous = false;
    char *dest_urlpath = "destarr.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_MAX, shape, cshape, bshape, view_start, view_stop,
                                                   naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                                   dest_contiguous, dest_urlpath, true));
}


INA_TEST_FIXTURE(reduce_multi_type, min_2_ui_f_1) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT32;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 2;
    int64_t shape[] = {12, 10};
    int64_t cshape[] = {6, 2};
    int64_t bshape[] = {3, 2};
    int8_t naxis = 1;
    int8_t axis[] = {0};

    int64_t view_start[] = {1, 1};
    int64_t view_stop[] = {5, 3};
    
    int64_t dest_cshape[] = {6};
    int64_t dest_bshape[] = {2};
    bool src_contiguous = true;
    char *src_urlpath = NULL;
    bool dest_contiguous = false;
    char *dest_urlpath = "destarr.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_MIN, shape, cshape, bshape, view_start, view_stop,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi_type, max_3_s_ll_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT16;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_INT64;

    int8_t ndim = 3;
    int64_t shape[] = {12, 12, 12};
    int64_t cshape[] = {6, 9, 6};
    int64_t bshape[] = {3, 3, 3};
    int8_t naxis = 2;
    int8_t axis[] = {0, 1};

   int64_t view_start[IARRAY_DIMENSION_MAX] = {0};
    int64_t view_stop[] = {10, 10, 10};
    
    int64_t dest_cshape[] = {6};
    int64_t dest_bshape[] = {3};
    bool src_contiguous = false;
    char *src_urlpath = NULL;
    bool dest_contiguous = true;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_MAX, shape, cshape, bshape, view_start, view_stop,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi_type, min_4_ui_i_4_oneshot) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT32;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_INT32;

    int8_t ndim = 4;
    int64_t shape[] = {5, 21, 27, 10};
    int64_t cshape[] = {4, 3, 1, 10};
    int64_t bshape[] = {3, 3, 1, 5};
    int8_t naxis = 4;
    int8_t axis[] = {3, 1, 2, 0};

   int64_t view_start[IARRAY_DIMENSION_MAX] = {0};
    int64_t view_stop[] = {5, 10, 27, 10};
    
    int64_t dest_cshape[] = {1};
    int64_t dest_bshape[] = {1};
    bool src_contiguous = true;
    char *src_urlpath = NULL;
    bool dest_contiguous = true;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_MIN, shape, cshape, bshape, view_start, view_stop,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, true));
}


INA_TEST_FIXTURE(reduce_multi_type, max_4_ull_d_3_oneshot) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT64;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 4;
    int64_t shape[] = {8, 8, 7, 7};
    int64_t cshape[] = {4, 5, 2, 5};
    int64_t bshape[] = {2, 2, 2, 3};
    int8_t naxis = 3;
    int8_t axis[] = {1, 2, 0};

    int64_t view_start[IARRAY_DIMENSION_MAX] = {0};
    int64_t view_stop[] = {4, 8, 2, 6};

    int64_t dest_cshape[] = {5};
    int64_t dest_bshape[] = {2};
    bool src_contiguous = true;
    char *src_urlpath = NULL;
    bool dest_contiguous = false;
    char *dest_urlpath = "iarray_reduce.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_MAX, shape, cshape, bshape, view_start, view_stop,
                                                   naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                                   dest_contiguous, dest_urlpath, true));
}


INA_TEST_FIXTURE(reduce_multi_type, min_2_sc_ui_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT8;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_UINT32;

    int8_t ndim = 2;
    int64_t shape[] = {12, 10};
    int64_t cshape[] = {69, 210};
    int64_t bshape[] = {31, 2};
    int8_t naxis = 2;
    int8_t axis[] = {1, 0};

   int64_t view_start[IARRAY_DIMENSION_MAX] = {0};
    int64_t view_stop[] = {10, 10};
    
    int64_t dest_cshape[] = {1};
    int64_t dest_bshape[] = {1};
    bool src_contiguous = true;
    char *src_urlpath = NULL;
    bool dest_contiguous = false;
    char *dest_urlpath = "destarr.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_MIN, shape, cshape, bshape, view_start, view_stop,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi_type, max_2_uc_f_1) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT8;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 2;
    int64_t shape[] = {12, 12};
    int64_t cshape[] = {6, 9};
    int64_t bshape[] = {3, 3};
    int8_t naxis = 1;
    int8_t axis[] = {0};

   int64_t view_start[IARRAY_DIMENSION_MAX] = {1, 10};
    int64_t view_stop[] = {3, 12};
    
    int64_t dest_cshape[] = {6};
    int64_t dest_bshape[] = {3};
    bool src_contiguous = false;
    char *src_urlpath = NULL;
    bool dest_contiguous = true;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_MAX, shape, cshape, bshape, view_start, view_stop,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi_type, min_2_b_f_2_oneshot) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_BOOL;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 2;
    int64_t shape[] = {12, 12};
    int64_t cshape[] = {6, 9};
    int64_t bshape[] = {3, 3};
    int8_t naxis = 2;
    int8_t axis[] = {0, 1};

   int64_t view_start[IARRAY_DIMENSION_MAX] = {0};
    int64_t view_stop[] = {12, 12};
    
    int64_t dest_cshape[] = {1};
    int64_t dest_bshape[] = {1};
    bool src_contiguous = false;
    char *src_urlpath = NULL;
    bool dest_contiguous = true;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_MIN, shape, cshape, bshape, view_start, view_stop,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, true));
}


INA_TEST_FIXTURE(reduce_multi_type, mean_2_d_ll_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_INT64;

    int8_t ndim = 2;
    int64_t shape[] = {12, 100};
    int64_t cshape[] = {6, 21};
    int64_t bshape[] = {3, 2};
    int8_t naxis = 2;
    int8_t axis[] = {1, 0};

   int64_t view_start[IARRAY_DIMENSION_MAX] = {0};
    int64_t view_stop[] = {12, 22};
    
    int64_t dest_cshape[] = {1};
    int64_t dest_bshape[] = {1};
    bool src_contiguous = true;
    char *src_urlpath = NULL;
    bool dest_contiguous = false;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_MEAN, shape, cshape, bshape, view_start, view_stop,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi_type, mean_3_f_ll_3_oneshot) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_INT64;

    int8_t ndim = 3;
    int64_t shape[] = {12, 12, 12};
    int64_t cshape[] = {6, 9, 6};
    int64_t bshape[] = {3, 3, 3};
    int8_t naxis = 3;
    int8_t axis[] = {0, 2, 1};

   int64_t view_start[IARRAY_DIMENSION_MAX] = {11, 11, 11};
    int64_t view_stop[] = {12, 12, 12};
    
    int64_t dest_cshape[] = {1};  // {} not compile on Windows
    int64_t dest_bshape[] = {1};  // {} not compile on Windows
    bool src_contiguous = true;
    char *src_urlpath = "srcarr.iarr";
    bool dest_contiguous = false;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_MEAN, shape, cshape, bshape, view_start, view_stop,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, true));
}


INA_TEST_FIXTURE(reduce_multi_type, mean_4_ll_d_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT64;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 4;
    int64_t shape[] = {5, 21, 27, 10};
    int64_t cshape[] = {4, 3, 1, 10};
    int64_t bshape[] = {3, 3, 1, 5};
    int8_t naxis = 2;
    int8_t axis[] = {0, 3};

   int64_t view_start[IARRAY_DIMENSION_MAX] = {0};
    int64_t view_stop[] = {6, 6, 6, 6};
    
    int64_t dest_cshape[] = {3, 1};
    int64_t dest_bshape[] = {3, 1};
    bool src_contiguous = false;
    char *src_urlpath = NULL;
    bool dest_contiguous = true;
    char *dest_urlpath = "iarray_reduce.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_MEAN, shape, cshape, bshape, view_start, view_stop,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi_type, mean_2_ui_ull_1_oneshot) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT32;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_UINT64;

    int8_t ndim = 2;
    int64_t shape[] = {12, 10};
    int64_t cshape[] = {6, 2};
    int64_t bshape[] = {3, 2};
    int8_t naxis = 1;
    int8_t axis[] = {0};

   int64_t view_start[IARRAY_DIMENSION_MAX] = {0};
    int64_t view_stop[] = {5, 3};
    
    int64_t dest_cshape[] = {6};
    int64_t dest_bshape[] = {2};
    bool src_contiguous = true;
    char *src_urlpath = NULL;
    bool dest_contiguous = false;
    char *dest_urlpath = "destarr.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_MEAN, shape, cshape, bshape, view_start, view_stop,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, true));
}


INA_TEST_FIXTURE(reduce_multi_type, mean_3_s_f_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT16;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 3;
    int64_t shape[] = {12, 12, 12};
    int64_t cshape[] = {6, 9, 6};
    int64_t bshape[] = {3, 3, 3};
    int8_t naxis = 2;
    int8_t axis[] = {0, 1};

   int64_t view_start[IARRAY_DIMENSION_MAX] = {0};
    int64_t view_stop[] = {12, 1, 12};
    
    int64_t dest_cshape[] = {6};
    int64_t dest_bshape[] = {3};
    bool src_contiguous = false;
    char *src_urlpath = NULL;
    bool dest_contiguous = true;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_MEAN, shape, cshape, bshape, view_start, view_stop,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi_type, mean_2_sc_d_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT8;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 2;
    int64_t shape[] = {12, 10};
    int64_t cshape[] = {69, 210};
    int64_t bshape[] = {31, 2};
    int8_t naxis = 2;
    int8_t axis[] = {1, 0};

    int64_t view_start[IARRAY_DIMENSION_MAX] = {0};
    int64_t view_stop[] = {10, 10};
    
    int64_t dest_cshape[] = {1};
    int64_t dest_bshape[] = {1};
    bool src_contiguous = true;
    char *src_urlpath = "srcarr.iarr";
    bool dest_contiguous = false;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_MEAN, shape, cshape, bshape, view_start, view_stop,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
    // Check that the temporary file is removed properly when the dest_urlpath = NULL and the view comes from disk
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_MEAN, shape, cshape, bshape, view_start, view_stop,
                                                   naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                                   dest_contiguous, dest_urlpath, false));
}


INA_TEST_FIXTURE(reduce_multi_type, mean_2_b_uc_1) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_BOOL;
    iarray_data_type_t  view_dtype = IARRAY_DATA_TYPE_UINT8;

    int8_t ndim = 2;
    int64_t shape[] = {12, 12};
    int64_t cshape[] = {6, 9};
    int64_t bshape[] = {3, 3};
    int8_t naxis = 1;
    int8_t axis[] = {0};

    int64_t view_start[IARRAY_DIMENSION_MAX] = {0};
    int64_t view_stop[] = {6, 10};

    int64_t dest_cshape[] = {6};
    int64_t dest_bshape[] = {3};
    bool src_contiguous = false;
    char *src_urlpath = NULL;
    bool dest_contiguous = true;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_multi_type(data->ctx, dtype, view_dtype, ndim, IARRAY_REDUCE_MEAN, shape, cshape, bshape, view_start, view_stop,
                                              naxis, axis, dest_cshape, dest_bshape, src_contiguous, src_urlpath,
                                              dest_contiguous, dest_urlpath, false));
}

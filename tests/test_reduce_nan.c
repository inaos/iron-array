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
#include <math.h>
#include <stdlib.h>


static ina_rc_t test_reduce_nan(iarray_context_t *ctx, iarray_data_type_t dtype, int8_t ndim, iarray_reduce_func_t func,
                               const int64_t *shape, const int64_t *cshape, const int64_t *bshape,
                               int8_t axis,
                               const int64_t *dest_cshape, const int64_t *dest_bshape, bool dest_frame,
                               char *dest_urlpath) {
    blosc2_remove_urlpath(dest_urlpath);
    // Create dtshape
    iarray_dtshape_t dtshape;

    dtshape.dtype = dtype;
    dtshape.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
    }

    iarray_storage_t storage = {0};
    for (int i = 0; i < ndim; ++i) {
        storage.chunkshape[i] = cshape[i];
        storage.blockshape[i] = bshape[i];
    }

    iarray_container_t *c_x;
    IARRAY_RETURN_IF_FAILED(iarray_ones(ctx, &dtshape, &storage, &c_x));

    // Fill array with nans
    int64_t nnans = rand() % (shape[axis] + 1);
    int64_t start[IARRAY_DIMENSION_MAX] = {0};
    int64_t stop[IARRAY_DIMENSION_MAX] = {0};
    for (int i = 0; i < ndim; ++i) {
        stop[i] = 1;
    }
    stop[axis] = nnans;

    void *nan_buf = malloc(nnans * c_x->dtshape->dtype_size);
    float *fnan_buf = (float *) nan_buf;
    double *dnan_buf = (double *) nan_buf;
    if (dtype == IARRAY_DATA_TYPE_FLOAT) {
        for (int i = 0; i < nnans; ++i) {
            fnan_buf[i] = NAN;
        }
    }
    else {
        for (int i = 0; i < nnans; ++i) {
            dnan_buf[i] = NAN;
        }
    }

    IARRAY_RETURN_IF_FAILED(iarray_set_slice_buffer(ctx, c_x, start, stop, nan_buf, nnans * c_x->dtshape->dtype_size));
    free(nan_buf);

    for (int i = 0; i < ndim; ++i) {
        storage.chunkshape[i] = cshape[i];
        storage.blockshape[i] = bshape[i];
    }

    iarray_storage_t dest_storage = {0};
    dest_storage.contiguous = dest_frame;
    dest_storage.urlpath = dest_urlpath;
    for (int i = 0; i < ndim - 1; ++i) {
        dest_storage.blockshape[i] = dest_bshape[i];
        dest_storage.chunkshape[i] = dest_cshape[i];
    }

    iarray_container_t *c_z;
    IARRAY_RETURN_IF_FAILED(iarray_reduce(ctx, c_x, func, axis, &dest_storage, &c_z));

    int64_t buffer_nitems = c_z->catarr->nitems;
    int64_t buffer_size = buffer_nitems * c_z->catarr->itemsize;
    uint8_t *buffer = malloc(buffer_size);

    IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, c_z, buffer, buffer_size));

    double val_ = 1;
    switch (func) {
        case IARRAY_REDUCE_NAN_MAX:
        case IARRAY_REDUCE_NAN_MIN:
        case IARRAY_REDUCE_NAN_MEDIAN:
        case IARRAY_REDUCE_NAN_MEAN:
            for (int i = 0; i < buffer_nitems; ++i) {
                switch (c_z->dtshape->dtype) {
                    case IARRAY_DATA_TYPE_DOUBLE:
                        if (i == 0 && nnans == shape[axis]) {
                            INA_TEST_ASSERT(isnan(((double *) buffer)[i]));
                        } else {
                            INA_TEST_ASSERT_EQUAL_FLOATING(((double *) buffer)[i], val_);
                        }
                        break;
                    case IARRAY_DATA_TYPE_FLOAT:
                        if (i == 0 && nnans == shape[axis]) {
                            INA_TEST_ASSERT(isnan(((float *) buffer)[i]));
                        } else {
                            INA_TEST_ASSERT_EQUAL_FLOATING(((float *) buffer)[i], val_);
                        }
                        break;
                    default:
                        IARRAY_TRACE1(iarray.error, "Invalid dtype");
                        return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
                }
            }
            break;
        case IARRAY_REDUCE_NAN_SUM: {
            for (int i = 0; i < buffer_nitems; ++i) {
                if (i == 0) {
                    val_ = (double)shape[axis] - (double)nnans;
                }
                else {
                    val_ = (double) shape[axis];
                }
                switch (c_z->dtshape->dtype) {
                    case IARRAY_DATA_TYPE_DOUBLE:
                        if (i == 0 && nnans == shape[axis]) {
                            INA_TEST_ASSERT_EQUAL_FLOATING(((double *) buffer)[i], 0.);
                        } else {
                            INA_TEST_ASSERT_EQUAL_FLOATING(((double *) buffer)[i], val_);
                        }
                        break;
                    case IARRAY_DATA_TYPE_FLOAT:
                        if (i == 0 && nnans == shape[axis]) {
                            INA_TEST_ASSERT_EQUAL_FLOATING(((float *) buffer)[i], 0.);
                        } else {
                            INA_TEST_ASSERT_EQUAL_FLOATING(((float *) buffer)[i], val_);
                        }
                        break;
                    default:
                        IARRAY_TRACE1(iarray.error, "Invalid dtype");
                        return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
                }
            }
            break;
        }
        case IARRAY_REDUCE_NAN_PROD:
            for (int i = 0; i < buffer_nitems; ++i) {
                switch (c_z->dtshape->dtype) {
                    case IARRAY_DATA_TYPE_DOUBLE:
                        if (i == 0 && nnans == shape[axis]) {
                            INA_TEST_ASSERT_EQUAL_FLOATING(((double *) buffer)[i], 1.);
                        } else {
                            INA_TEST_ASSERT_EQUAL_FLOATING(((double *) buffer)[i], val_);
                        }
                        break;
                    case IARRAY_DATA_TYPE_FLOAT:
                        if (i == 0 && nnans == shape[axis]) {
                            INA_TEST_ASSERT_EQUAL_FLOATING(((float *) buffer)[i], 1.);
                        } else {
                            INA_TEST_ASSERT_EQUAL_FLOATING(((float *) buffer)[i], val_);
                        }
                        break;
                    default:
                        IARRAY_TRACE1(iarray.error, "Invalid dtype");
                        return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
                }
            }
            break;
        case IARRAY_REDUCE_MAX:
        case IARRAY_REDUCE_MIN:
        case IARRAY_REDUCE_MEDIAN:
        case IARRAY_REDUCE_PROD:
        case IARRAY_REDUCE_MEAN:
            for (int i = 0; i < buffer_nitems; ++i) {
                switch (c_z->dtshape->dtype) {
                    case IARRAY_DATA_TYPE_DOUBLE:
                        if (i == 0 && nnans != 0) {
                            INA_TEST_ASSERT(isnan(((double *) buffer)[i]));
                        } else {
                            INA_TEST_ASSERT_EQUAL_FLOATING(((double *) buffer)[i], val_);
                        }
                        break;
                    case IARRAY_DATA_TYPE_FLOAT:
                        if (i == 0 && nnans != 0) {
                            INA_TEST_ASSERT(isnan(((float *) buffer)[i]));
                        } else {
                            INA_TEST_ASSERT_EQUAL_FLOATING(((float *) buffer)[i], val_);
                        }
                        break;
                    default:
                        IARRAY_TRACE1(iarray.error, "Invalid dtype");
                        return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
                }
            }
            break;
        case IARRAY_REDUCE_SUM: {
            for (int i = 0; i < buffer_nitems; ++i) {
                val_ = (double) shape[axis];
                switch (c_z->dtshape->dtype) {
                    case IARRAY_DATA_TYPE_DOUBLE:
                        if (i == 0 && nnans != 0) {
                            INA_TEST_ASSERT(isnan(((double *) buffer)[i]));
                        } else {
                            INA_TEST_ASSERT_EQUAL_FLOATING(((double *) buffer)[i], val_);
                        }
                        break;
                    case IARRAY_DATA_TYPE_FLOAT:
                        if (i == 0 && nnans != 0) {
                            INA_TEST_ASSERT(isnan(((float *) buffer)[i]));
                        } else {
                            INA_TEST_ASSERT_EQUAL_FLOATING(((float *) buffer)[i], val_);
                        }
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

    free(buffer);
    iarray_container_free(ctx, &c_z);
    iarray_container_free(ctx, &c_x);

    return INA_SUCCESS;
}

INA_TEST_DATA(reduce_nan) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(reduce_nan) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(reduce_nan) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}


INA_TEST_FIXTURE(reduce_nan, sum_2_f_1) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 2;
    int64_t shape[] = {8, 8};
    int64_t cshape[] = {4, 4};
    int64_t bshape[] = {2, 2};
    int8_t axis = 1;

    int64_t dest_cshape[] = {4};
    int64_t dest_bshape[] = {2};
    bool dest_frame = false;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_nan(data->ctx, dtype, ndim, IARRAY_REDUCE_NAN_SUM, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}


INA_TEST_FIXTURE(reduce_nan, sum_2_d_0) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {5, 5};
    int64_t bshape[] = {2, 2};
    int8_t axis = 0;

    int64_t dest_cshape[] = {5};
    int64_t dest_bshape[] = {1};
    bool dest_frame = true;
    char *dest_urlpath = "arr.iarr";

    INA_TEST_ASSERT_SUCCEED(test_reduce_nan(data->ctx, dtype, ndim, IARRAY_REDUCE_NAN_SUM, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}


INA_TEST_FIXTURE(reduce_nan, max_2_d_1) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 2;
    int64_t shape[] = {8, 4};
    int64_t cshape[] = {4, 4};
    int64_t bshape[] = {2, 2};
    int8_t axis = 1;

    int64_t dest_cshape[] = {4};
    int64_t dest_bshape[] = {2};
    bool dest_frame = false;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_nan(data->ctx, dtype, ndim, IARRAY_REDUCE_NAN_MAX, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}


INA_TEST_FIXTURE(reduce_nan, max_2_f_0) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 2;
    int64_t shape[] = {12, 100};
    int64_t cshape[] = {6, 21};
    int64_t bshape[] = {3, 2};
    int8_t axis = 0;

    int64_t dest_cshape[] = {6};
    int64_t dest_bshape[] = {3};
    bool dest_frame = true;
    char *dest_urlpath = NULL;

    INA_TEST_ASSERT_SUCCEED(test_reduce_nan(data->ctx, dtype, ndim, IARRAY_REDUCE_NAN_MAX, shape, cshape, bshape, axis,
                                            dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}


INA_TEST_FIXTURE(reduce_nan, min_3_f_1) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 3;
    int64_t shape[] = {6, 12, 6};
    int64_t cshape[] = {6, 9, 6};
    int64_t bshape[] = {3, 3, 3};
    int8_t axis = 1;

    int64_t dest_cshape[] = {6, 6};
    int64_t dest_bshape[] = {3, 3};
    bool dest_frame = false;
    char *dest_urlpath = NULL;

    INA_TEST_ASSERT_SUCCEED(test_reduce_nan(data->ctx, dtype, ndim, IARRAY_REDUCE_NAN_MIN, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}


INA_TEST_FIXTURE(reduce_nan, min_3_d_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 3;
    int64_t shape[] = {7, 6, 7};
    int64_t cshape[] = {5, 3, 5};
    int64_t bshape[] = {3, 2, 2};
    int8_t axis = 2;

    int64_t dest_cshape[] = {5, 3};
    int64_t dest_bshape[] = {3, 2};
    bool dest_frame = true;
    char *dest_urlpath = "arr.iarr";
    INA_TEST_ASSERT_SUCCEED(test_reduce_nan(data->ctx, dtype, ndim, IARRAY_REDUCE_NAN_MIN, shape, cshape, bshape, axis,
                                            dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}


INA_TEST_FIXTURE(reduce_nan, mean_3_d_1) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 3;
    int64_t shape[] = {12, 12, 12};
    int64_t cshape[] = {6, 9, 6};
    int64_t bshape[] = {3, 3, 3};
    int8_t axis = 1;

    int64_t dest_cshape[] = {6, 6};
    int64_t dest_bshape[] = {3, 3};
    bool dest_frame = false;
    char *dest_urlpath = NULL;

    INA_TEST_ASSERT_SUCCEED(test_reduce_nan(data->ctx, dtype, ndim, IARRAY_REDUCE_NAN_MEAN, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}


INA_TEST_FIXTURE(reduce_nan, mean_2_f_0) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {5, 5};
    int64_t bshape[] = {2, 2};
    int8_t axis = 0;

    int64_t dest_cshape[] = {5};
    int64_t dest_bshape[] = {1};
    bool dest_frame = true;
    char *dest_urlpath = "arr.iarr";

    INA_TEST_ASSERT_SUCCEED(test_reduce_nan(data->ctx, dtype, ndim, IARRAY_REDUCE_NAN_MEAN, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}


INA_TEST_FIXTURE(reduce_nan, var_2_f_0) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 2;
    int64_t shape[] = {12, 100};
    int64_t cshape[] = {6, 21};
    int64_t bshape[] = {3, 2};
    int8_t axis = 0;

    int64_t dest_cshape[] = {6};
    int64_t dest_bshape[] = {3};
    bool dest_frame = true;
    char *dest_urlpath = NULL;

    INA_TEST_ASSERT_SUCCEED(test_reduce_nan(data->ctx, dtype, ndim, IARRAY_REDUCE_NAN_VAR, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}


INA_TEST_FIXTURE(reduce_nan, var_3_d_1) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 3;
    int64_t shape[] = {12, 12, 12};
    int64_t cshape[] = {6, 9, 6};
    int64_t bshape[] = {3, 3, 3};
    int8_t axis = 1;

    int64_t dest_cshape[] = {6, 6};
    int64_t dest_bshape[] = {3, 3};
    bool dest_frame = false;
    char *dest_urlpath = NULL;

    INA_TEST_ASSERT_SUCCEED(test_reduce_nan(data->ctx, dtype, ndim, IARRAY_REDUCE_NAN_VAR, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}


INA_TEST_FIXTURE(reduce_nan, std_2_d_0) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 2;
    int64_t shape[] = {8, 8};
    int64_t cshape[] = {2, 4};
    int64_t bshape[] = {2, 2};
    int8_t axis = 0;

    int64_t dest_cshape[] = {4};
    int64_t dest_bshape[] = {2};
    bool dest_frame = false;
    char *dest_urlpath = NULL;

    INA_TEST_ASSERT_SUCCEED(test_reduce_nan(data->ctx, dtype, ndim, IARRAY_REDUCE_NAN_STD, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}


INA_TEST_FIXTURE(reduce_nan, std_3_f_1) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 3;
    int64_t shape[] = {4, 5, 5};
    int64_t cshape[] = {4, 5, 5};
    int64_t bshape[] = {2, 2, 2};
    int8_t axis = 1;

    int64_t dest_cshape[] = {4, 5};
    int64_t dest_bshape[] = {2, 2};
    bool dest_frame = false;
    char *dest_urlpath = NULL;

    INA_TEST_ASSERT_SUCCEED(test_reduce_nan(data->ctx, dtype, ndim, IARRAY_REDUCE_NAN_STD, shape, cshape, bshape, axis,
                                            dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}


INA_TEST_FIXTURE(reduce_nan, median_3_f_1) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 3;
    int64_t shape[] = {12, 12, 12};
    int64_t cshape[] = {6, 9, 6};
    int64_t bshape[] = {3, 3, 3};
    int8_t axis = 1;

    int64_t dest_cshape[] = {6, 6};
    int64_t dest_bshape[] = {3, 3};
    bool dest_frame = false;
    char *dest_urlpath = NULL;

    INA_TEST_ASSERT_SUCCEED(test_reduce_nan(data->ctx, dtype, ndim, IARRAY_REDUCE_NAN_MEDIAN, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}

INA_TEST_FIXTURE(reduce_nan, median_2_d_0) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 2;
    int64_t shape[] = {4, 4};
    int64_t cshape[] = {4, 4};
    int64_t bshape[] = {2, 2};
    int8_t axis = 0;

    int64_t dest_cshape[] = {4};
    int64_t dest_bshape[] = {2};
    bool dest_frame = false;
    char *dest_urlpath = NULL;

    INA_TEST_ASSERT_SUCCEED(test_reduce_nan(data->ctx, dtype, ndim, IARRAY_REDUCE_NAN_MEDIAN, shape, cshape, bshape, axis,
                                            dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}


INA_TEST_FIXTURE(reduce_nan, prod_2_d_0) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {5, 5};
    int64_t bshape[] = {2, 2};
    int8_t axis = 0;

    int64_t dest_cshape[] = {5};
    int64_t dest_bshape[] = {1};
    bool dest_frame = true;
    char *dest_urlpath = "arr.iarr";

    INA_TEST_ASSERT_SUCCEED(test_reduce_nan(data->ctx, dtype, ndim, IARRAY_REDUCE_NAN_PROD, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}


INA_TEST_FIXTURE(reduce_nan, prod_2_f_0) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 2;
    int64_t shape[] = {4, 4};
    int64_t cshape[] = {4, 4};
    int64_t bshape[] = {2, 2};
    int8_t axis = 0;

    int64_t dest_cshape[] = {4};
    int64_t dest_bshape[] = {2};
    bool dest_frame = false;
    char *dest_urlpath = NULL;

    INA_TEST_ASSERT_SUCCEED(test_reduce_nan(data->ctx, dtype, ndim, IARRAY_REDUCE_NAN_PROD, shape, cshape, bshape, axis,
                                            dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}

INA_TEST_FIXTURE(reduce_nan, sum_3_f_2) {
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

    INA_TEST_ASSERT_SUCCEED(test_reduce_nan(data->ctx, dtype, ndim, IARRAY_REDUCE_SUM, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}


INA_TEST_FIXTURE(reduce_nan, max_2_d_0) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {5, 5};
    int64_t bshape[] = {2, 2};
    int8_t axis = 0;

    int64_t dest_cshape[] = {5};
    int64_t dest_bshape[] = {2};
    bool dest_frame = true;
    char *dest_urlpath = "arr.iarr";

    INA_TEST_ASSERT_SUCCEED(test_reduce_nan(data->ctx, dtype, ndim, IARRAY_REDUCE_MAX, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}


INA_TEST_FIXTURE(reduce_nan, min_d_f_1) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 3;
    int64_t shape[] = {6, 12, 6};
    int64_t cshape[] = {6, 9, 6};
    int64_t bshape[] = {3, 3, 3};
    int8_t axis = 1;

    int64_t dest_cshape[] = {6, 6};
    int64_t dest_bshape[] = {3, 3};
    bool dest_frame = false;
    char *dest_urlpath = "arr.iarr";

    INA_TEST_ASSERT_SUCCEED(test_reduce_nan(data->ctx, dtype, ndim, IARRAY_REDUCE_MIN, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}


INA_TEST_FIXTURE(reduce_nan, mean_2_d_0) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {5, 5};
    int64_t bshape[] = {4, 4};
    int8_t axis = 0;

    int64_t dest_cshape[] = {8};
    int64_t dest_bshape[] = {8};
    bool dest_frame = true;
    char *dest_urlpath = "arr.iarr";

    INA_TEST_ASSERT_SUCCEED(test_reduce_nan(data->ctx, dtype, ndim, IARRAY_REDUCE_MEAN, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}


INA_TEST_FIXTURE(reduce_nan, var_2_f_1) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 2;
    int64_t shape[] = {8, 8};
    int64_t cshape[] = {4, 4};
    int64_t bshape[] = {2, 2};
    int8_t axis = 1;

    int64_t dest_cshape[] = {4};
    int64_t dest_bshape[] = {2};
    bool dest_frame = false;
    char *dest_urlpath = NULL;
    INA_TEST_ASSERT_SUCCEED(test_reduce_nan(data->ctx, dtype, ndim, IARRAY_REDUCE_VAR, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}


INA_TEST_FIXTURE(reduce_nan, std_3_f_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 3;
    int64_t shape[] = {12, 12, 12};
    int64_t cshape[] = {6, 9, 6};
    int64_t bshape[] = {3, 3, 3};
    int8_t axis = 2;

    int64_t dest_cshape[] = {6, 6};
    int64_t dest_bshape[] = {3, 3};
    bool dest_frame = false;
    char *dest_urlpath = NULL;

    INA_TEST_ASSERT_SUCCEED(test_reduce_nan(data->ctx, dtype, ndim, IARRAY_REDUCE_STD, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}


INA_TEST_FIXTURE(reduce_nan, median_4_d_0) {
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

    INA_TEST_ASSERT_SUCCEED(test_reduce_nan(data->ctx, dtype, ndim, IARRAY_REDUCE_MEDIAN, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}


INA_TEST_FIXTURE(reduce_nan, prod_3_d_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 3;
    int64_t shape[] = {8, 8, 7};
    int64_t cshape[] = {4, 5, 2};
    int64_t bshape[] = {2, 2, 2};
    int8_t axis = 2;

    int64_t dest_cshape[] = {4, 5};
    int64_t dest_bshape[] = {2, 2};
    bool dest_frame = false;
    char *dest_urlpath = NULL;

    INA_TEST_ASSERT_SUCCEED(test_reduce_nan(data->ctx, dtype, ndim, IARRAY_REDUCE_PROD, shape, cshape, bshape, axis,
                                        dest_cshape, dest_bshape, dest_frame, dest_urlpath));
}

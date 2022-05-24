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

#include <src/iarray_private.h>
#include <libiarray/iarray.h>
#include <tests/iarray_test.h>

static ina_rc_t test_slice_type(iarray_context_t *ctx, iarray_container_t *c_x, int64_t *start, int64_t *stop,
                               iarray_container_t **c_slice, iarray_container_t **c_out, iarray_data_type_t view_dtype) {
    INA_TEST_ASSERT_SUCCEED(iarray_get_slice(ctx, c_x, start, stop, true, NULL, c_slice));
    INA_TEST_ASSERT_SUCCEED(iarray_get_type_view(ctx, (*c_slice), view_dtype, c_out));

    INA_TEST_ASSERT_SUCCEED(iarray_squeeze(ctx, *c_out));

    return INA_SUCCESS;
}

static ina_rc_t execute_iarray_slice_type(iarray_context_t *ctx, iarray_data_type_t src_dtype, int32_t src_type_size,
                                          iarray_data_type_t view_dtype, int8_t ndim,
                                      const int64_t *shape, const int64_t *cshape, const int64_t *bshape,
                                      int64_t *start, int64_t *stop, const void *result, bool xcontiguous, char *xurlpath) {
    void *buffer_x;
    size_t buffer_x_len;

    buffer_x_len = 1;
    for (int i = 0; i < ndim; ++i) {
        buffer_x_len *= shape[i];
    }
    buffer_x = ina_mem_alloc(buffer_x_len * src_type_size);

    fill_buf(src_dtype, buffer_x, buffer_x_len);

    iarray_dtshape_t xdtshape;

    xdtshape.dtype = src_dtype;
    xdtshape.ndim = ndim;
    for (int j = 0; j < xdtshape.ndim; ++j) {
        xdtshape.shape[j] = shape[j];
    }

    iarray_storage_t xstore;
    xstore.contiguous = xcontiguous;
    xstore.urlpath = xurlpath;
    for (int i = 0; i < ndim; ++i) {
        xstore.chunkshape[i] = cshape[i];
        xstore.blockshape[i] = bshape[i];
    }
    blosc2_remove_urlpath(xstore.urlpath);
    iarray_container_t *c_x;
    iarray_container_t *c_slice;
    iarray_container_t *c_out;


    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &xdtshape, buffer_x, buffer_x_len * src_type_size,
                                               &xstore, &c_x));

    INA_TEST_ASSERT_SUCCEED(test_slice_type(ctx, c_x, start, stop, &c_slice, &c_out, view_dtype));

    int64_t bufdes_size = 1;

    for (int k = 0; k < ndim; ++k) {
        int64_t st = (start[k] + shape[k]) % shape[k];
        int64_t sp = (stop[k] + shape[k] - 1) % shape[k] + 1;
        bufdes_size *= sp - st;
    }

    uint8_t *bufdes;

    switch (view_dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            bufdes = ina_mem_alloc(bufdes_size * sizeof(double));
            INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, bufdes_size * sizeof(double)));
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_FLOATING(((double *) bufdes)[l], ((double *) result)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            bufdes = ina_mem_alloc(bufdes_size * sizeof(float));
            INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, bufdes_size * sizeof(float)));
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_FLOATING(((float *) bufdes)[l], ((float *) result)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_INT64:
            bufdes = ina_mem_alloc(bufdes_size * sizeof(int64_t));
            INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, bufdes_size * sizeof(int64_t)));
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_INT64(((int64_t *) bufdes)[l], ((int64_t *) result)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_INT32:
            bufdes = ina_mem_alloc(bufdes_size * sizeof(int32_t));
            INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, bufdes_size * sizeof(int32_t)));
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_INT(((int32_t *) bufdes)[l], ((int32_t *) result)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_INT16:
            bufdes = ina_mem_alloc(bufdes_size * sizeof(int16_t));
            INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, bufdes_size * sizeof(int16_t)));
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_INT(((int16_t *) bufdes)[l], ((int16_t *) result)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_INT8:
            bufdes = ina_mem_alloc(bufdes_size * sizeof(int8_t));
            INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, bufdes_size * sizeof(int8_t)));
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_INT(((int8_t *) bufdes)[l], ((int8_t *) result)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_UINT64:
            bufdes = ina_mem_alloc(bufdes_size * sizeof(uint64_t));
            INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, bufdes_size * sizeof(uint64_t)));
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_UINT64(((uint64_t *) bufdes)[l], ((uint64_t *) result)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_UINT32:
            bufdes = ina_mem_alloc(bufdes_size * sizeof(uint32_t));
            INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, bufdes_size * sizeof(uint32_t)));
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_UINT(((uint32_t *) bufdes)[l], ((uint32_t *) result)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_UINT16:
            bufdes = ina_mem_alloc(bufdes_size * sizeof(uint16_t));
            INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, bufdes_size * sizeof(uint16_t)));
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_UINT(((uint16_t *) bufdes)[l], ((uint16_t *) result)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_UINT8:
            bufdes = ina_mem_alloc(bufdes_size * sizeof(uint8_t));
            INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, bufdes_size * sizeof(uint8_t)));
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_UINT(((uint8_t *) bufdes)[l], ((uint8_t *) result)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_BOOL:
            bufdes = ina_mem_alloc(bufdes_size * sizeof(bool));
            INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, bufdes_size * sizeof(bool)));
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT(((bool *) bufdes)[l] == ((bool *) result)[l]);
            }
            break;
        default:
            return INA_ERR_EXCEEDED;
    }

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_slice);
    iarray_container_free(ctx, &c_out);
    blosc2_remove_urlpath(xstore.urlpath);

    ina_mem_free(buffer_x);
    if (bufdes_size != 0) {
        ina_mem_free(bufdes);
    }

    return INA_SUCCESS;
}

INA_TEST_DATA(slice_type) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(slice_type) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.max_num_threads = 3;
    iarray_context_new(&cfg, &data->ctx);

}

INA_TEST_TEARDOWN(slice_type) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(slice_type, 3_f_ll_v) {
    iarray_data_type_t src_dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t src_type_size = sizeof(float);
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_INT64;

    const int8_t ndim = 3;
    int64_t shape[] = {10, 10, 10};
    int64_t cshape[] = {9, 5, 4};
    int64_t bshape[] = {5, 5, 2};
    int64_t start[] = {3, 0, 3};
    int64_t stop[] = {-4, -3, 10};

    int64_t result[] = {303, 304, 305, 306, 307, 308, 309,
                        313, 314, 315, 316, 317, 318, 319,
                      323, 324, 325, 326, 327, 328, 329,
                        333, 334, 335, 336, 337, 338, 339,
                      343, 344, 345, 346, 347, 348, 349,
                        353, 354, 355, 356, 357, 358, 359,
                      363, 364, 365, 366, 367, 368, 369,
                        403, 404, 405, 406, 407, 408, 409,
                      413, 414, 415, 416, 417, 418, 419,
                        423, 424, 425, 426, 427, 428, 429,
                      433, 434, 435, 436, 437, 438, 439,
                        443, 444, 445, 446, 447, 448, 449,
                      453, 454, 455, 456, 457, 458, 459,
                        463, 464, 465, 466, 467, 468, 469,
                      503, 504, 505, 506, 507, 508, 509,
                        513, 514, 515, 516, 517, 518, 519,
                      523, 524, 525, 526, 527, 528, 529,
                        533, 534, 535, 536, 537, 538, 539,
                      543, 544, 545, 546, 547, 548, 549,
                        553, 554, 555, 556, 557, 558, 559,
                      563, 564, 565, 566, 567, 568, 569};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_slice_type(data->ctx, src_dtype, src_type_size, view_dtype, ndim, shape, cshape, bshape,
                                                     start, stop, result, true, "xarr.iarr"));
}

INA_TEST_FIXTURE(slice_type, 4_ll_d_v) {
    iarray_data_type_t src_dtype = IARRAY_DATA_TYPE_INT64;
    int32_t src_type_size = sizeof(int64_t);
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_DOUBLE;

    const int8_t ndim = 4;
    int64_t shape[] = {10, 10, 10, 10};
    int64_t cshape[] = {3, 5, 2, 7};
    int64_t bshape[] = {2, 2, 2, 4};
    int64_t start[] = {5, -7, 9, 2};
    int64_t stop[] = {-1, 6, 10, -3};

    double result[] = {5392, 5393, 5394, 5395, 5396, 5492, 5493, 5494, 5495, 5496, 5592, 5593,
                       5594, 5595, 5596, 6392, 6393, 6394, 6395, 6396, 6492, 6493, 6494, 6495,
                       6496, 6592, 6593, 6594, 6595, 6596, 7392, 7393, 7394, 7395, 7396, 7492,
                       7493, 7494, 7495, 7496, 7592, 7593, 7594, 7595, 7596, 8392, 8393, 8394,
                       8395, 8396, 8492, 8493, 8494, 8495, 8496, 8592, 8593, 8594, 8595, 8596};


    INA_TEST_ASSERT_SUCCEED(execute_iarray_slice_type(data->ctx, src_dtype, src_type_size, view_dtype, ndim, shape, cshape, bshape,
                                                     start, stop, result, false, NULL));
}

INA_TEST_FIXTURE(slice_type, 2_uc_ll_v) {
    iarray_data_type_t src_dtype = IARRAY_DATA_TYPE_UINT8;
    int32_t src_type_size = sizeof(uint8_t);
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_INT64;

    const int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {4, 5};
    int64_t bshape[] = {2, 2};
    int64_t start[] = {5, 4};
    int64_t stop[] = {8, 6};

    int64_t result[] = {54, 55,
                        64, 65,
                        74, 75};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_slice_type(data->ctx, src_dtype, src_type_size, view_dtype, ndim, shape, cshape, bshape,
                                                     start, stop, result, false, "xarr.iarr"));
}

INA_TEST_FIXTURE(slice_type, 3_s_f_v) {
    iarray_data_type_t src_dtype = IARRAY_DATA_TYPE_INT16;
    int32_t src_type_size = sizeof(int16_t);
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_FLOAT;

    const int8_t ndim = 3;
    int64_t shape[] = {10, 10, 10};
    int64_t cshape[] = {4, 5, 1};
    int64_t bshape[] = {2, 2, 1};
    int64_t start[] = {5, 4, 3};
    int64_t stop[] = {8, 6, 3};

    float result[] = {0}; // Fix windows

    INA_TEST_ASSERT_SUCCEED(execute_iarray_slice_type(data->ctx, src_dtype, src_type_size, view_dtype, ndim, shape, cshape, bshape,
                                                     start, stop, result, false, NULL));
}

INA_TEST_FIXTURE(slice_type, 3_us_d_v) {
    iarray_data_type_t src_dtype = IARRAY_DATA_TYPE_UINT16;
    int32_t src_type_size = sizeof(uint16_t);
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_DOUBLE;

    const int8_t ndim = 3;
    int64_t shape[] = {10, 10, 10};
    int64_t cshape[] = {5, 4, 5};
    int64_t bshape[] = {2, 2, 1};
    int64_t start[] = {0, 5, 5};
    int64_t stop[] = {0, 10, 5};

    double result[] = {0}; // Fix windows

    INA_TEST_ASSERT_SUCCEED(execute_iarray_slice_type(data->ctx, src_dtype, src_type_size, view_dtype, ndim, shape, cshape, bshape,
                                                     start, stop, result, true, NULL));
}

INA_TEST_FIXTURE(slice_type, 2_b_ui_v) {
    iarray_data_type_t src_dtype = IARRAY_DATA_TYPE_BOOL;
    int32_t src_type_size = sizeof(bool);
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_UINT32;

    const int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {9, 5};
    int64_t bshape[] = {5, 5};
    int64_t start[] = {3, 0};
    int64_t stop[] = {-4, -3};

    uint32_t result[] = {0, 1, 0, 1, 0, 1, 0,
                         0, 1, 0, 1, 0, 1, 0,
                         0, 1, 0, 1, 0, 1, 0};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_slice_type(data->ctx, src_dtype, src_type_size, view_dtype, ndim, shape, cshape, bshape,
                                                     start, stop, result, true, "xarr.iarr"));
}

INA_TEST_FIXTURE(slice_type, 4_ull_d_v) {
    iarray_data_type_t src_dtype = IARRAY_DATA_TYPE_UINT64;
    int32_t src_type_size = sizeof(uint64_t);
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_DOUBLE;

    const int8_t ndim = 4;
    int64_t shape[] = {10, 10, 10, 10};
    int64_t cshape[] = {3, 5, 2, 7};
    int64_t bshape[] = {2, 2, 2, 4};
    int64_t start[] = {5, -7, 9, 2};
    int64_t stop[] = {-1, 6, 10, -3};

    double result[] = {5392, 5393, 5394, 5395, 5396, 5492, 5493, 5494, 5495, 5496, 5592, 5593,
                       5594, 5595, 5596, 6392, 6393, 6394, 6395, 6396, 6492, 6493, 6494, 6495,
                       6496, 6592, 6593, 6594, 6595, 6596, 7392, 7393, 7394, 7395, 7396, 7492,
                       7493, 7494, 7495, 7496, 7592, 7593, 7594, 7595, 7596, 8392, 8393, 8394,
                       8395, 8396, 8492, 8493, 8494, 8495, 8496, 8592, 8593, 8594, 8595, 8596};


    INA_TEST_ASSERT_SUCCEED(execute_iarray_slice_type(data->ctx, src_dtype, src_type_size, view_dtype, ndim, shape, cshape, bshape,
                                                     start, stop, result, false, NULL));
}

INA_TEST_FIXTURE(slice_type, 3_ui_f_v) {
    iarray_data_type_t src_dtype = IARRAY_DATA_TYPE_UINT32;
    int32_t src_type_size = sizeof(uint32_t);
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_FLOAT;

    const int8_t ndim = 3;
    int64_t shape[] = {10, 10, 10};
    int64_t cshape[] = {4, 5, 1};
    int64_t bshape[] = {2, 2, 1};
    int64_t start[] = {5, 4, 3};
    int64_t stop[] = {8, 6, 5};

    float result[] = {543, 544,
                     553, 554,
                     643, 644,
                     653, 654,
                     743, 744,
                     753, 754};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_slice_type(data->ctx, src_dtype, src_type_size, view_dtype, ndim, shape, cshape, bshape,
                                                     start, stop, result, false, "xarr.iarr"));
}

INA_TEST_FIXTURE(slice_type, 3_uc_b_v) {
    iarray_data_type_t src_dtype = IARRAY_DATA_TYPE_UINT8;
    int32_t src_type_size = sizeof(uint8_t);
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_BOOL;

    const int8_t ndim = 3;
    int64_t shape[] = {10, 10, 10};
    int64_t cshape[] = {5, 4, 5};
    int64_t bshape[] = {2, 2, 1};
    int64_t start[] = {0, 5, 5};
    int64_t stop[] = {0, 10, 5};

    bool result[] = {true}; // Fix windows

    INA_TEST_ASSERT_SUCCEED(execute_iarray_slice_type(data->ctx, src_dtype, src_type_size, view_dtype, ndim, shape, cshape, bshape,
                                                     start, stop, result, true, NULL));
}

INA_TEST_FIXTURE(slice_type, 2_c_b_v) {
    iarray_data_type_t src_dtype = IARRAY_DATA_TYPE_INT8;
    int32_t src_type_size = sizeof(int8_t);
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_BOOL;

    const int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {4, 5};
    int64_t bshape[] = {2, 2};
    int64_t start[] = {5, 4};
    int64_t stop[] = {8, 6};

    bool result[] = {true, true,
                     true, true,
                     true, true};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_slice_type(data->ctx, src_dtype, src_type_size, view_dtype, ndim, shape, cshape, bshape,
                                                     start, stop, result, false, "xarr.iarr"));
}

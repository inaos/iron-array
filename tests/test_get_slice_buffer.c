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

#include <src/iarray_private.h>
#include <libiarray/iarray.h>
#include <tests/iarray_test.h>

static ina_rc_t test_slice_buffer(iarray_context_t *ctx, iarray_container_t *c_x, int64_t * start, int64_t *stop,
    void *buffer, int64_t buflen) {

    INA_TEST_ASSERT_SUCCEED(iarray_get_slice_buffer(ctx, c_x, start, stop, buffer, buflen));

    return INA_SUCCESS;
}

static ina_rc_t _execute_iarray_slice(iarray_context_t *ctx, iarray_data_type_t dtype, int64_t type_size, int8_t ndim,
                                      const int64_t *shape, const int64_t *cshape, const int64_t *bshape,
                                      int64_t *start, int64_t *stop, const void *result, bool contiguous, char *urlpath) {
    void *buffer_x;
    size_t buffer_x_len;

    buffer_x_len = 1;
    for (int i = 0; i < ndim; ++i) {
        buffer_x_len *= shape[i];
    }
    buffer_x = ina_mem_alloc(buffer_x_len * type_size);

    fill_buf(dtype, buffer_x, buffer_x_len);

    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    for (int j = 0; j < xdtshape.ndim; ++j) {
        xdtshape.shape[j] = shape[j];
    }

    iarray_storage_t store;
    store.contiguous = contiguous;
    store.urlpath = urlpath;
    for (int j = 0; j < xdtshape.ndim; ++j) {
        store.chunkshape[j] = cshape[j];
        store.blockshape[j] = bshape[j];
    }
    int64_t bufdes_size = 1;
    blosc2_remove_urlpath(store.urlpath);

    for (int k = 0; k < ndim; ++k) {
        int64_t st = (start[k] + shape[k]) % shape[k];
        int64_t sp = (stop[k] + shape[k] - 1) % shape[k] + 1;
        bufdes_size *= (int64_t) sp - st;
    }

    uint8_t *bufdes;

    int64_t buflen = bufdes_size;

    switch (dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
        case IARRAY_DATA_TYPE_INT64:
        case IARRAY_DATA_TYPE_UINT64:
            buflen *= 8;
            break;
        case IARRAY_DATA_TYPE_FLOAT:
        case IARRAY_DATA_TYPE_INT32:
        case IARRAY_DATA_TYPE_UINT32:
            buflen *= 4;
            break;
        case IARRAY_DATA_TYPE_INT16:
        case IARRAY_DATA_TYPE_UINT16:
            buflen *= 2;
            break;
        case IARRAY_DATA_TYPE_INT8:
        case IARRAY_DATA_TYPE_UINT8:
            buflen *= 1;
            break;
        case IARRAY_DATA_TYPE_BOOL:
            buflen *= sizeof(boolean_t);
            break;
    }

    bufdes = ina_mem_alloc(bufdes_size * sizeof(double));

    iarray_container_t *c_x;

    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &xdtshape, buffer_x, buffer_x_len * type_size, &store, 0, &c_x));

    INA_TEST_ASSERT_SUCCEED(test_slice_buffer(ctx, c_x, start, stop, bufdes, buflen));


    switch (dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_FLOATING(((double *) bufdes)[l], ((double *) result)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_FLOATING(((float *) bufdes)[l], ((float *) result)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_INT64:
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_INT64(((int64_t *) bufdes)[l], ((int64_t *) result)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_INT32:
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_INT(((int32_t *) bufdes)[l], ((int32_t *) result)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_INT16:
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_INT(((int16_t *) bufdes)[l], ((int16_t *) result)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_INT8:
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_INT(((int8_t *) bufdes)[l], ((int8_t *) result)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_UINT64:
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_UINT64(((uint64_t *) bufdes)[l], ((uint64_t *) result)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_UINT32:
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_UINT(((uint32_t *) bufdes)[l], ((uint32_t *) result)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_UINT16:
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_UINT(((uint16_t *) bufdes)[l], ((uint16_t *) result)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_UINT8:
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_UINT(((uint8_t *) bufdes)[l], ((uint8_t *) result)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_BOOL:
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT(((boolean_t *) bufdes)[l] == ((boolean_t *) result)[l]);
            }
            break;
    }

    iarray_container_free(ctx, &c_x);

    ina_mem_free(buffer_x);
    ina_mem_free(bufdes);
    blosc2_remove_urlpath(store.urlpath);

    return INA_SUCCESS;
}

INA_TEST_DATA(get_slice_buffer) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(get_slice_buffer) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.eval_method = IARRAY_EVAL_METHOD_ITERCHUNK;

    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(get_slice_buffer) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}
INA_TEST_FIXTURE(get_slice_buffer, 2_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    const int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {5, 5};
    int64_t bshape[] = {2, 2};
    int64_t start[] = {-5, -7};
    int64_t stop[] = {-1, 10};

    double result[] = {53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68, 69, 73, 74, 75, 76,
                       77, 78, 79, 83, 84, 85, 86, 87, 88, 89};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                  start, stop, result, false, NULL));
}

INA_TEST_FIXTURE(get_slice_buffer, 3_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    const int8_t ndim = 3;
    int64_t shape[] = {10, 10, 10};
    int64_t cshape[] = {3, 5, 2};
    int64_t bshape[] = {3, 5, 2};
    int64_t start[] = {3, 0, 3};
    int64_t stop[] = {-4, -3, 10};


    float result[] = {303, 304, 305, 306, 307, 308, 309, 313, 314, 315, 316, 317, 318, 319,
                      323, 324, 325, 326, 327, 328, 329, 333, 334, 335, 336, 337, 338, 339,
                      343, 344, 345, 346, 347, 348, 349, 353, 354, 355, 356, 357, 358, 359,
                      363, 364, 365, 366, 367, 368, 369, 403, 404, 405, 406, 407, 408, 409,
                      413, 414, 415, 416, 417, 418, 419, 423, 424, 425, 426, 427, 428, 429,
                      433, 434, 435, 436, 437, 438, 439, 443, 444, 445, 446, 447, 448, 449,
                      453, 454, 455, 456, 457, 458, 459, 463, 464, 465, 466, 467, 468, 469,
                      503, 504, 505, 506, 507, 508, 509, 513, 514, 515, 516, 517, 518, 519,
                      523, 524, 525, 526, 527, 528, 529, 533, 534, 535, 536, 537, 538, 539,
                      543, 544, 545, 546, 547, 548, 549, 553, 554, 555, 556, 557, 558, 559,
                      563, 564, 565, 566, 567, 568, 569};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                  start, stop, result, false, "arr.iarr"));
}

INA_TEST_FIXTURE(get_slice_buffer, 4_ll) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT64;
    int32_t type_size = sizeof(int64_t);

    const int8_t ndim = 4;
    int64_t shape[] = {10, 10, 10, 10};
    int64_t cshape[] = {7, 8, 8, 4};
    int64_t bshape[] = {3, 5, 2, 4};
    int64_t start[] = {5, -7, 9, 2};
    int64_t stop[] = {-1, 6, 10, -3};

    int64_t result[] = {5392, 5393, 5394, 5395, 5396, 5492, 5493, 5494, 5495, 5496, 5592, 5593,
                       5594, 5595, 5596, 6392, 6393, 6394, 6395, 6396, 6492, 6493, 6494, 6495,
                       6496, 6592, 6593, 6594, 6595, 6596, 7392, 7393, 7394, 7395, 7396, 7492,
                       7493, 7494, 7495, 7496, 7592, 7593, 7594, 7595, 7596, 8392, 8393, 8394,
                       8395, 8396, 8492, 8493, 8494, 8495, 8496, 8592, 8593, 8594, 8595, 8596};


    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                  start, stop, result, true, NULL));
}

INA_TEST_FIXTURE(get_slice_buffer, 5_i) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT32;
    int32_t type_size = sizeof(int32_t);

    const int8_t ndim = 5;
    int64_t shape[] = {10, 10, 10, 10, 10};
    int64_t cshape[] = {5, 5, 5, 5, 5};
    int64_t bshape[] = {2, 5, 1, 5, 2};
    int64_t start[] = {-4, 0, -5, 5, 7};
    int64_t stop[] = {8, 9, -4, -4, 10};

    int32_t result[] = {60557, 60558, 60559, 61557, 61558, 61559, 62557, 62558, 62559, 63557,
                      63558, 63559, 64557, 64558, 64559, 65557, 65558, 65559, 66557, 66558,
                      66559, 67557, 67558, 67559, 68557, 68558, 68559, 70557, 70558, 70559,
                      71557, 71558, 71559, 72557, 72558, 72559, 73557, 73558, 73559, 74557,
                      74558, 74559, 75557, 75558, 75559, 76557, 76558, 76559, 77557, 77558,
                      77559, 78557, 78558, 78559};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                  start, stop, result, true, "arr.iarr"));
}

INA_TEST_FIXTURE(get_slice_buffer, 5_s) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT16;
    int32_t type_size = sizeof(int16_t);

    const int8_t ndim = 5;
    int64_t shape[] = {10, 10, 10, 10, 10};
    int64_t cshape[] = {10, 10, 10, 10, 10};
    int64_t bshape[] = {5, 5, 5, 5, 5};
    int64_t start[] = {0, 4, -8, 4, 5};
    int64_t stop[] = {1, 7, 4, -4, 8};


    int16_t result[] = {4245, 4246, 4247, 4255, 4256, 4257, 4345, 4346, 4347, 4355, 4356, 4357,
                        5245, 5246, 5247, 5255, 5256, 5257, 5345, 5346, 5347, 5355, 5356, 5357,
                        6245, 6246, 6247, 6255, 6256, 6257, 6345, 6346, 6347, 6355, 6356, 6357};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                  start, stop, result, false, "arr.iarr"));
}

INA_TEST_FIXTURE(get_slice_buffer, 2_sc) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT8;
    int32_t type_size = sizeof(int8_t);

    const int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {3, 10};
    int64_t bshape[] = {1, 7};
    int64_t start[] = {5, -9};
    int64_t stop[] = {7, -7};

    int8_t result[] = {51, 52, 61, 62};


    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                  start, stop, result, false, NULL));
}

INA_TEST_FIXTURE(get_slice_buffer, 4_ull) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT64;
    int32_t type_size = sizeof(uint64_t);

    const int8_t ndim = 4;
    int64_t shape[] = {10, 10, 10, 10};
    int64_t cshape[] = {7, 8, 8, 4};
    int64_t bshape[] = {3, 5, 2, 4};
    int64_t start[] = {5, -7, 9, 2};
    int64_t stop[] = {-1, 6, 10, -3};

    uint64_t result[] = {5392, 5393, 5394, 5395, 5396, 5492, 5493, 5494, 5495, 5496, 5592, 5593,
                        5594, 5595, 5596, 6392, 6393, 6394, 6395, 6396, 6492, 6493, 6494, 6495,
                        6496, 6592, 6593, 6594, 6595, 6596, 7392, 7393, 7394, 7395, 7396, 7492,
                        7493, 7494, 7495, 7496, 7592, 7593, 7594, 7595, 7596, 8392, 8393, 8394,
                        8395, 8396, 8492, 8493, 8494, 8495, 8496, 8592, 8593, 8594, 8595, 8596};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                  start, stop, result, false, NULL));
}

INA_TEST_FIXTURE(get_slice_buffer, 3_ui) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT32;
    int32_t type_size = sizeof(uint32_t);

    const int8_t ndim = 3;
    int64_t shape[] = {10, 10, 10};
    int64_t cshape[] = {3, 5, 2};
    int64_t bshape[] = {3, 5, 2};
    int64_t start[] = {3, 0, 3};
    int64_t stop[] = {-4, -3, 10};


    uint32_t result[] = {303, 304, 305, 306, 307, 308, 309, 313, 314, 315, 316, 317, 318, 319,
                      323, 324, 325, 326, 327, 328, 329, 333, 334, 335, 336, 337, 338, 339,
                      343, 344, 345, 346, 347, 348, 349, 353, 354, 355, 356, 357, 358, 359,
                      363, 364, 365, 366, 367, 368, 369, 403, 404, 405, 406, 407, 408, 409,
                      413, 414, 415, 416, 417, 418, 419, 423, 424, 425, 426, 427, 428, 429,
                      433, 434, 435, 436, 437, 438, 439, 443, 444, 445, 446, 447, 448, 449,
                      453, 454, 455, 456, 457, 458, 459, 463, 464, 465, 466, 467, 468, 469,
                      503, 504, 505, 506, 507, 508, 509, 513, 514, 515, 516, 517, 518, 519,
                      523, 524, 525, 526, 527, 528, 529, 533, 534, 535, 536, 537, 538, 539,
                      543, 544, 545, 546, 547, 548, 549, 553, 554, 555, 556, 557, 558, 559,
                      563, 564, 565, 566, 567, 568, 569};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                  start, stop, result, false, "arr.iarr"));
}

INA_TEST_FIXTURE(get_slice_buffer, 4_us) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT16;
    int32_t type_size = sizeof(uint16_t);

    const int8_t ndim = 4;
    int64_t shape[] = {10, 10, 10, 10};
    int64_t cshape[] = {7, 8, 8, 4};
    int64_t bshape[] = {3, 5, 2, 4};
    int64_t start[] = {5, -7, 9, 2};
    int64_t stop[] = {-1, 6, 10, -3};

    uint16_t result[] = {5392, 5393, 5394, 5395, 5396, 5492, 5493, 5494, 5495, 5496, 5592, 5593,
                        5594, 5595, 5596, 6392, 6393, 6394, 6395, 6396, 6492, 6493, 6494, 6495,
                        6496, 6592, 6593, 6594, 6595, 6596, 7392, 7393, 7394, 7395, 7396, 7492,
                        7493, 7494, 7495, 7496, 7592, 7593, 7594, 7595, 7596, 8392, 8393, 8394,
                        8395, 8396, 8492, 8493, 8494, 8495, 8496, 8592, 8593, 8594, 8595, 8596};


    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                  start, stop, result, true, NULL));
}

INA_TEST_FIXTURE(get_slice_buffer, 2_uc) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT8;
    int32_t type_size = sizeof(uint8_t);

    const int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {5, 5};
    int64_t bshape[] = {2, 5};
    int64_t start[] = {-4, 0};
    int64_t stop[] = {8, 9};

    uint8_t result[] = {60, 61, 62, 63, 64, 65, 66, 67, 68,
                        70, 71, 72, 73, 74, 75, 76, 77, 78};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                  start, stop, result, true, "arr.iarr"));
}

INA_TEST_FIXTURE(get_slice_buffer, 6_b) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_BOOL;
    int32_t type_size = sizeof(boolean_t);

    const int8_t ndim = 6;
    int64_t shape[] = {10, 10, 10, 10, 10, 10};
    int64_t cshape[] = {10, 10, 10, 10, 10, 10};
    int64_t bshape[] = {5, 5, 5, 5, 5, 5};
    int64_t start[] = {0, 4, -8, 4, 5, 1};
    int64_t stop[] = {1, 7, 4, -4, 8, 3};

    boolean_t result[] = {TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE,
                          TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE,
                          TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE,
                          TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE,
                          TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE,
                          TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE,
                          TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE,
                          TRUE, FALSE};


    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                  start, stop, result, false, "arr.iarr"));
}

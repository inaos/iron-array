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

static ina_rc_t test_slice(iarray_context_t *ctx, iarray_container_t *c_x, int64_t *start,
                           int64_t *stop, iarray_storage_t *stores, iarray_container_t **c_out) {
    INA_TEST_ASSERT_SUCCEED(iarray_get_slice(ctx, c_x, start, stop, true, stores, c_out));
    INA_TEST_ASSERT_SUCCEED(iarray_squeeze(ctx, *c_out));

    return INA_SUCCESS;
}

static ina_rc_t _execute_iarray_slice(iarray_context_t *ctx, iarray_data_type_t dtype, int32_t type_size, int8_t ndim,
                                      const int64_t *shape, const int64_t *cshape, const int64_t *bshape,
                                      int64_t *start, int64_t *stop, const void *result, bool xcontiguous, char *xurlpath) {
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

    iarray_storage_t xstore;
    xstore.urlpath = xurlpath;
    xstore.contiguous = xcontiguous;
    for (int i = 0; i < ndim; ++i) {
        xstore.chunkshape[i] = cshape[i];
        xstore.blockshape[i] = bshape[i];
    }
    blosc2_remove_urlpath(xstore.urlpath);
    iarray_container_t *c_x;
    iarray_container_t *c_out;

    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &xdtshape, buffer_x, buffer_x_len * type_size,
                                               &xstore, &c_x));

    INA_TEST_ASSERT_SUCCEED(test_slice(ctx, c_x, start, stop, NULL, &c_out));

    iarray_iter_read_t *iter;
    iarray_iter_read_value_t val;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_new(ctx, &iter, c_out, &val));
    while (INA_SUCCEED(iarray_iter_read_has_next(iter))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_read_next(iter));
        switch (dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                INA_TEST_ASSERT_EQUAL_FLOATING(((double *) val.elem_pointer)[0], ((double *) result)[val.elem_flat_index]);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                INA_TEST_ASSERT_EQUAL_FLOATING(((float *) val.elem_pointer)[0], ((float *) result)[val.elem_flat_index]);
                break;
            case IARRAY_DATA_TYPE_INT64:
                INA_TEST_ASSERT_EQUAL_INT64(((int64_t *) val.elem_pointer)[0], ((int64_t *) result)[val.elem_flat_index]);
                break;
            case IARRAY_DATA_TYPE_INT32:
                INA_TEST_ASSERT_EQUAL_INT(((int32_t *) val.elem_pointer)[0], ((int32_t *) result)[val.elem_flat_index]);
                break;
            case IARRAY_DATA_TYPE_INT16:
                INA_TEST_ASSERT_EQUAL_INT(((int16_t *) val.elem_pointer)[0], ((int16_t *) result)[val.elem_flat_index]);
                break;
            case IARRAY_DATA_TYPE_INT8:
                INA_TEST_ASSERT_EQUAL_INT(((int8_t *) val.elem_pointer)[0], ((int8_t *) result)[val.elem_flat_index]);
                break;
            case IARRAY_DATA_TYPE_UINT64:
                INA_TEST_ASSERT_EQUAL_UINT64(((uint64_t *) val.elem_pointer)[0], ((uint64_t *) result)[val.elem_flat_index]);
                break;
            case IARRAY_DATA_TYPE_UINT32:
                INA_TEST_ASSERT_EQUAL_UINT(((uint32_t *) val.elem_pointer)[0], ((uint32_t *) result)[val.elem_flat_index]);
                break;
            case IARRAY_DATA_TYPE_UINT16:
                INA_TEST_ASSERT_EQUAL_UINT(((uint16_t *) val.elem_pointer)[0], ((uint16_t *) result)[val.elem_flat_index]);
                break;
            case IARRAY_DATA_TYPE_UINT8:
                INA_TEST_ASSERT_EQUAL_UINT(((uint8_t *) val.elem_pointer)[0], ((uint8_t *) result)[val.elem_flat_index]);
                break;
            case IARRAY_DATA_TYPE_BOOL:
                INA_TEST_ASSERT(((bool *) val.elem_pointer)[0] == ((bool *) result)[val.elem_flat_index]);
                break;
            default:
                return INA_ERR_EXCEEDED;
        }
    }
    iarray_iter_read_free(&iter);
    INA_TEST_ASSERT_SUCCEED(ina_err_get_rc() != INA_RC_PACK(IARRAY_ERR_END_ITER, 0));

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_out);
    blosc2_remove_urlpath(xstore.urlpath);

    ina_mem_free(buffer_x);

    return INA_SUCCESS;
}

INA_TEST_DATA(view_iter) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(view_iter) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;

    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(view_iter) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}


INA_TEST_FIXTURE(view_iter, 3_us_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT16;
    int32_t type_size = sizeof(uint16_t);

    const int8_t ndim = 3;
    int64_t shape[] = {10, 10, 10};
    int64_t cshape[] = {9, 9, 9};
    int64_t bshape[] = {3, 5, 5};
    int64_t start[] = {3, 0, 3};
    int64_t stop[] = {-4, -3, 10};

    uint16_t result[] = {303, 304, 305, 306, 307, 308, 309, 313, 314, 315, 316, 317, 318, 319,
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
                                                      start, stop, result, false, "xarr.iarr"));
}


INA_TEST_FIXTURE(view_iter, 4_d_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    const int8_t ndim = 4;
    int64_t shape[] = {10, 10, 10, 10};
    int64_t cshape[] = {3, 5, 2, 7};
    int64_t bshape[] = {2, 2, 1, 4};
    int64_t start[] = {5, -7, 9, 2};
    int64_t stop[] = {-1, 6, 10, -3};

    double result[] = {5392, 5393, 5394, 5395, 5396, 5492, 5493, 5494, 5495, 5496, 5592, 5593,
                       5594, 5595, 5596, 6392, 6393, 6394, 6395, 6396, 6492, 6493, 6494, 6495,
                       6496, 6592, 6593, 6594, 6595, 6596, 7392, 7393, 7394, 7395, 7396, 7492,
                       7493, 7494, 7495, 7496, 7592, 7593, 7594, 7595, 7596, 8392, 8393, 8394,
                       8395, 8396, 8492, 8493, 8494, 8495, 8496, 8592, 8593, 8594, 8595, 8596};


    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                      start, stop, result, true, "xarr.iarr"));
}

INA_TEST_FIXTURE(view_iter, 2_f_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    const int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {4, 5};
    int64_t bshape[] = {2, 3};
    int64_t start[] = {5, 4};
    int64_t stop[] = {8, 6};

    float result[] = {54, 55,
                      64, 65,
                      74, 75};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                      start, stop, result, true, NULL));
}

/* Avoid heavy tests
INA_TEST_FIXTURE(view_iter, 7_f_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    const int8_t ndim = 7;
    int64_t shape[] = {10, 10, 10, 10, 10, 10, 10};
    int64_t cshape[] = {4, 5, 1, 8, 5, 3, 10};
    int64_t bshape[] = {2, 3, 1, 2, 2, 3, 3};
    int64_t start[] = {5, 4, 3, -2, 4, 5, -9};
    int64_t stop[] = {8, 6, 5, 9, 7, 7, -7};

    float result[] = {5438451, 5438452, 5438461, 5438462, 5438551, 5438552, 5438561, 5438562,
                      5438651, 5438652, 5438661, 5438662, 5448451, 5448452, 5448461, 5448462,
                      5448551, 5448552, 5448561, 5448562, 5448651, 5448652, 5448661, 5448662,
                      5538451, 5538452, 5538461, 5538462, 5538551, 5538552, 5538561, 5538562,
                      5538651, 5538652, 5538661, 5538662, 5548451, 5548452, 5548461, 5548462,
                      5548551, 5548552, 5548561, 5548562, 5548651, 5548652, 5548661, 5548662,
                      6438451, 6438452, 6438461, 6438462, 6438551, 6438552, 6438561, 6438562,
                      6438651, 6438652, 6438661, 6438662, 6448451, 6448452, 6448461, 6448462,
                      6448551, 6448552, 6448561, 6448562, 6448651, 6448652, 6448661, 6448662,
                      6538451, 6538452, 6538461, 6538462, 6538551, 6538552, 6538561, 6538562,
                      6538651, 6538652, 6538661, 6538662, 6548451, 6548452, 6548461, 6548462,
                      6548551, 6548552, 6548561, 6548562, 6548651, 6548652, 6548661, 6548662,
                      7438451, 7438452, 7438461, 7438462, 7438551, 7438552, 7438561, 7438562,
                      7438651, 7438652, 7438661, 7438662, 7448451, 7448452, 7448461, 7448462,
                      7448551, 7448552, 7448561, 7448562, 7448651, 7448652, 7448661, 7448662,
                      7538451, 7538452, 7538461, 7538462, 7538551, 7538552, 7538561, 7538562,
                      7538651, 7538652, 7538661, 7538662, 7548451, 7548452, 7548461, 7548462,
                      7548551, 7548552, 7548561, 7548562, 7548651, 7548652, 7548661, 7548662};

    INA_TEST_ASSERT_SUCCEED(test_orthogonal_selection(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                  start, stop, result, true, NULL));
}
*/

INA_TEST_FIXTURE(view_iter, 3_i_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT32;
    int32_t type_size = sizeof(int32_t);

    const int8_t ndim = 3;
    int64_t shape[] = {10, 10, 10};
    int64_t cshape[] = {9, 9, 9};
    int64_t bshape[] = {3, 5, 5};
    int64_t start[] = {3, 0, 3};
    int64_t stop[] = {-4, -3, 10};

    int32_t result[] = {303, 304, 305, 306, 307, 308, 309, 313, 314, 315, 316, 317, 318, 319,
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
                                                      start, stop, result, false, "xarr.iarr"));
}


INA_TEST_FIXTURE(view_iter, 4_ui_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT32;
    int32_t type_size = sizeof(uint32_t);

    const int8_t ndim = 4;
    int64_t shape[] = {10, 10, 10, 10};
    int64_t cshape[] = {3, 5, 2, 7};
    int64_t bshape[] = {2, 2, 1, 4};
    int64_t start[] = {5, -7, 9, 2};
    int64_t stop[] = {-1, 6, 10, -3};

    uint32_t result[] = {5392, 5393, 5394, 5395, 5396, 5492, 5493, 5494, 5495, 5496, 5592, 5593,
                       5594, 5595, 5596, 6392, 6393, 6394, 6395, 6396, 6492, 6493, 6494, 6495,
                       6496, 6592, 6593, 6594, 6595, 6596, 7392, 7393, 7394, 7395, 7396, 7492,
                       7493, 7494, 7495, 7496, 7592, 7593, 7594, 7595, 7596, 8392, 8393, 8394,
                       8395, 8396, 8492, 8493, 8494, 8495, 8496, 8592, 8593, 8594, 8595, 8596};


    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                      start, stop, result, true, "xarr.iarr"));
}

INA_TEST_FIXTURE(view_iter, 3_ull_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT64;
    int32_t type_size = sizeof(uint64_t);

    const int8_t ndim = 3;
    int64_t shape[] = {10, 10, 10};
    int64_t cshape[] = {5, 3, 10};
    int64_t bshape[] = {2, 3, 3};
    int64_t start[] = {4, 5, -9};
    int64_t stop[] = {7, 7, -7};

    uint64_t result[] = {451, 452,
                         461, 462,
                         551, 552,
                         561, 562,
                         651, 652,
                         661, 662};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                      start, stop, result, true, NULL));
}

/* Avoid heavy tests
INA_TEST_FIXTURE(view_iter, 7_ull_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT64;
    int32_t type_size = sizeof(uint64_t);

    const int8_t ndim = 7;
    int64_t shape[] = {10, 10, 10, 10, 10, 10, 10};
    int64_t cshape[] = {4, 5, 1, 8, 5, 3, 10};
    int64_t bshape[] = {2, 3, 1, 2, 2, 3, 3};
    int64_t start[] = {5, 4, 3, -2, 4, 5, -9};
    int64_t stop[] = {8, 6, 5, 9, 7, 7, -7};

    uint64_t result[] = {5438451, 5438452, 5438461, 5438462, 5438551, 5438552, 5438561, 5438562,
                      5438651, 5438652, 5438661, 5438662, 5448451, 5448452, 5448461, 5448462,
                      5448551, 5448552, 5448561, 5448562, 5448651, 5448652, 5448661, 5448662,
                      5538451, 5538452, 5538461, 5538462, 5538551, 5538552, 5538561, 5538562,
                      5538651, 5538652, 5538661, 5538662, 5548451, 5548452, 5548461, 5548462,
                      5548551, 5548552, 5548561, 5548562, 5548651, 5548652, 5548661, 5548662,
                      6438451, 6438452, 6438461, 6438462, 6438551, 6438552, 6438561, 6438562,
                      6438651, 6438652, 6438661, 6438662, 6448451, 6448452, 6448461, 6448462,
                      6448551, 6448552, 6448561, 6448562, 6448651, 6448652, 6448661, 6448662,
                      6538451, 6538452, 6538461, 6538462, 6538551, 6538552, 6538561, 6538562,
                      6538651, 6538652, 6538661, 6538662, 6548451, 6548452, 6548461, 6548462,
                      6548551, 6548552, 6548561, 6548562, 6548651, 6548652, 6548661, 6548662,
                      7438451, 7438452, 7438461, 7438462, 7438551, 7438552, 7438561, 7438562,
                      7438651, 7438652, 7438661, 7438662, 7448451, 7448452, 7448461, 7448462,
                      7448551, 7448552, 7448561, 7448562, 7448651, 7448652, 7448661, 7448662,
                      7538451, 7538452, 7538461, 7538462, 7538551, 7538552, 7538561, 7538562,
                      7538651, 7538652, 7538661, 7538662, 7548451, 7548452, 7548461, 7548462,
                      7548551, 7548552, 7548561, 7548562, 7548651, 7548652, 7548661, 7548662};

    INA_TEST_ASSERT_SUCCEED(test_orthogonal_selection(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                  start, stop, result, true, NULL));
}
*/

INA_TEST_FIXTURE(view_iter, 3_s_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT16;
    int32_t type_size = sizeof(int16_t);

    const int8_t ndim = 3;
    int64_t shape[] = {10, 10, 10};
    int64_t cshape[] = {9, 9, 9};
    int64_t bshape[] = {3, 5, 5};
    int64_t start[] = {3, 0, 3};
    int64_t stop[] = {-4, -3, 10};

    int16_t result[] = {303, 304, 305, 306, 307, 308, 309, 313, 314, 315, 316, 317, 318, 319,
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
                                                      start, stop, result, false, "xarr.iarr"));
}


INA_TEST_FIXTURE(view_iter, 4_b_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_BOOL;
    int32_t type_size = sizeof(bool);

    const int8_t ndim = 4;
    int64_t shape[] = {10, 10, 10, 10};
    int64_t cshape[] = {3, 5, 2, 7};
    int64_t bshape[] = {2, 2, 1, 4};
    int64_t start[] = {5, -7, 9, 2};
    int64_t stop[] = {-1, 6, 10, -3};

    bool result[] = {false, true, false, true, false,
                     false, true, false, true, false,
                     false, true, false, true, false,
                     false, true, false, true, false,
                     false, true, false, true, false,
                     false, true, false, true, false,
                     false, true, false, true, false,
                     false, true, false, true, false,
                     false, true, false, true, false,
                     false, true, false, true, false,
                     false, true, false, true, false,
                     false, true, false, true, false};


    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                      start, stop, result, true, "xarr.iarr"));
}

INA_TEST_FIXTURE(view_iter, 2_sc_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT8;
    int32_t type_size = sizeof(int8_t);

    const int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {4, 5};
    int64_t bshape[] = {2, 3};
    int64_t start[] = {5, 4};
    int64_t stop[] = {8, 6};

    int8_t result[] = {54, 55, 64, 65, 74, 75};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                      start, stop, result, true, NULL));
}

INA_TEST_FIXTURE(view_iter, 1_uc_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT8;
    int32_t type_size = sizeof(uint8_t);

    const int8_t ndim = 1;
    int64_t shape[] = {10};
    int64_t cshape[] = {9};
    int64_t bshape[] = {3};
    int64_t start[] = {3};
    int64_t stop[] = {10};

    uint8_t result[] = {3, 4, 5, 6, 7, 8, 9};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                      start, stop, result, false, "xarr.iarr"));
}


INA_TEST_FIXTURE(view_iter, 4_ll_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT64;
    int32_t type_size = sizeof(int64_t);

    const int8_t ndim = 4;
    int64_t shape[] = {10, 10, 10, 10};
    int64_t cshape[] = {3, 5, 2, 7};
    int64_t bshape[] = {2, 2, 1, 4};
    int64_t start[] = {5, -7, 9, 2};
    int64_t stop[] = {-1, 6, 10, -3};

    int64_t result[] = {5392, 5393, 5394, 5395, 5396, 5492, 5493, 5494, 5495, 5496, 5592, 5593,
                       5594, 5595, 5596, 6392, 6393, 6394, 6395, 6396, 6492, 6493, 6494, 6495,
                       6496, 6592, 6593, 6594, 6595, 6596, 7392, 7393, 7394, 7395, 7396, 7492,
                       7493, 7494, 7495, 7496, 7592, 7593, 7594, 7595, 7596, 8392, 8393, 8394,
                       8395, 8396, 8492, 8493, 8494, 8495, 8496, 8592, 8593, 8594, 8595, 8596};


    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                      start, stop, result, true, "xarr.iarr"));
}

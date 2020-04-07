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

static ina_rc_t test_slice_buffer(iarray_context_t *ctx, iarray_container_t *c_x, int64_t * start, int64_t *stop,
    void *buffer, int64_t buflen) {

    INA_TEST_ASSERT_SUCCEED(iarray_get_slice_buffer(ctx, c_x, start, stop, buffer, buflen));

    return INA_SUCCESS;
}

static ina_rc_t _execute_iarray_slice(iarray_context_t *ctx, iarray_data_type_t dtype, int64_t type_size, int8_t ndim,
                                      const int64_t *shape, const int64_t *pshape,
                                      int64_t *start, int64_t *stop, const void *result, int transposed) {
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

    iarray_store_properties_t store;
    store.backend = pshape ? IARRAY_STORAGE_BLOSC : IARRAY_STORAGE_PLAINBUFFER;
    store.enforce_frame = false;
    store.filename = NULL;

    int64_t bufdes_size = 1;

    for (int k = 0; k < ndim; ++k) {
        int64_t st = (start[k] + shape[k]) % shape[k];
        int64_t sp = (stop[k] + shape[k] - 1) % shape[k] + 1;
        bufdes_size *= (int64_t) sp - st;
    }

    uint8_t *bufdes;

    int64_t buflen = bufdes_size;

    if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
        buflen *= sizeof(double);
    } else {
        buflen *= sizeof(float);
    }

    bufdes = ina_mem_alloc(bufdes_size * sizeof(double));

    iarray_container_t *c_x;

    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &xdtshape, buffer_x, buffer_x_len * type_size, &store, 0, &c_x));

    if (transposed == 1) {
        iarray_linalg_transpose(ctx, c_x);
    }

    INA_TEST_ASSERT_SUCCEED(test_slice_buffer(ctx, c_x, start, stop, bufdes, buflen));


    if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
        for (int64_t l = 0; l < bufdes_size; ++l) {
            INA_TEST_ASSERT_EQUAL_FLOATING(((double *) bufdes)[l], ((double *) result)[l]);
        }
    } else {
        for (int64_t l = 0; l < bufdes_size; ++l) {
            INA_TEST_ASSERT_EQUAL_FLOATING(((float *) bufdes)[l], ((float *) result)[l]);
        }
    }

    iarray_container_free(ctx, &c_x);

    ina_mem_free(buffer_x);
    ina_mem_free(bufdes);

    return INA_SUCCESS;
}

INA_TEST_DATA(get_slice_buffer) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(get_slice_buffer) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.eval_flags = IARRAY_EVAL_METHOD_ITERCHUNK;

    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(get_slice_buffer) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}
INA_TEST_FIXTURE(get_slice_buffer, 2_d_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    const int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t *pshape = NULL;
    int64_t start[] = {-5, -7};
    int64_t stop[] = {-1, 10};
    bool transposed = false;

    double result[] = {53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68, 69, 73, 74, 75, 76,
                       77, 78, 79, 83, 84, 85, 86, 87, 88, 89};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                  start, stop, result, transposed));
}

INA_TEST_FIXTURE(get_slice_buffer, 3_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    const int8_t ndim = 3;
    int64_t shape[] = {10, 10, 10};
    int64_t pshape[] = {3, 5, 2};
    int64_t start[] = {3, 0, 3};
    int64_t stop[] = {-4, -3, 10};
    bool transposed = false;


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

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                  start, stop, result, transposed));
}

INA_TEST_FIXTURE(get_slice_buffer, 4_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    const int8_t ndim = 4;
    int64_t shape[] = {10, 10, 10, 10};
    int64_t pshape[] = {3, 5, 2, 7};
    int64_t start[] = {5, -7, 9, 2};
    int64_t stop[] = {-1, 6, 10, -3};
    bool transposed = false;

    double result[] = {5392, 5393, 5394, 5395, 5396, 5492, 5493, 5494, 5495, 5496, 5592, 5593,
                       5594, 5595, 5596, 6392, 6393, 6394, 6395, 6396, 6492, 6493, 6494, 6495,
                       6496, 6592, 6593, 6594, 6595, 6596, 7392, 7393, 7394, 7395, 7396, 7492,
                       7493, 7494, 7495, 7496, 7592, 7593, 7594, 7595, 7596, 8392, 8393, 8394,
                       8395, 8396, 8492, 8493, 8494, 8495, 8496, 8592, 8593, 8594, 8595, 8596};


    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                  start, stop, result, transposed));
}

INA_TEST_FIXTURE(get_slice_buffer, 5_f_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    const int8_t ndim = 5;
    int64_t shape[] = {10, 10, 10, 10, 10};
    int64_t *pshape = NULL;
    int64_t start[] = {-4, 0, -5, 5, 7};
    int64_t stop[] = {8, 9, -4, -4, 10};
    bool transposed = false;

    float result[] = {60557, 60558, 60559, 61557, 61558, 61559, 62557, 62558, 62559, 63557,
                      63558, 63559, 64557, 64558, 64559, 65557, 65558, 65559, 66557, 66558,
                      66559, 67557, 67558, 67559, 68557, 68558, 68559, 70557, 70558, 70559,
                      71557, 71558, 71559, 72557, 72558, 72559, 73557, 73558, 73559, 74557,
                      74558, 74559, 75557, 75558, 75559, 76557, 76558, 76559, 77557, 77558,
                      77559, 78557, 78558, 78559};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                  start, stop, result, transposed));
}

INA_TEST_FIXTURE(get_slice_buffer, 6_d_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    const int8_t ndim = 6;
    int64_t shape[] = {10, 10, 10, 10, 10, 10};
    int64_t *pshape = NULL;
    int64_t start[] = {0, 4, -8, 4, 5, 1};
    int64_t stop[] = {1, 7, 4, -4, 8, 3};
    bool transposed = false;

    double result[] = {42451, 42452, 42461, 42462, 42471, 42472, 42551, 42552, 42561, 42562,
                       42571, 42572, 43451, 43452, 43461, 43462, 43471, 43472, 43551, 43552,
                       43561, 43562, 43571, 43572, 52451, 52452, 52461, 52462, 52471, 52472,
                       52551, 52552, 52561, 52562, 52571, 52572, 53451, 53452, 53461, 53462,
                       53471, 53472, 53551, 53552, 53561, 53562, 53571, 53572, 62451, 62452,
                       62461, 62462, 62471, 62472, 62551, 62552, 62561, 62562, 62571, 62572,
                       63451, 63452, 63461, 63462, 63471, 63472, 63551, 63552, 63561, 63562,
                       63571, 63572};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                  start, stop, result, transposed));
}

INA_TEST_FIXTURE(get_slice_buffer, 7_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    const int8_t ndim = 7;
    int64_t shape[] = {10, 10, 10, 10, 10, 10, 10};
    int64_t pshape[] = {4, 5, 1, 8, 5, 3, 10};
    int64_t start[] = {5, 4, 3, -2, 4, 5, -9};
    int64_t stop[] = {8, 6, 5, 9, 7, 7, -7};
    bool transposed = false;

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

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                  start, stop, result, transposed));
}

INA_TEST_FIXTURE(get_slice_buffer, 8_d_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    const int8_t ndim = 8;
    int64_t shape[] = {10, 10, 10, 10, 10, 10, 10, 10};
    int64_t *pshape = NULL;
    int64_t start[] = {3, 5, 2, 4, 5, 1, 6, 0};
    int64_t stop[] = {6, 6, 4, 6, 7, 3, 7, 3};
    bool transposed = false;

    double result[] = {35245160, 35245161, 35245162, 35245260, 35245261, 35245262, 35246160,
                       35246161, 35246162, 35246260, 35246261, 35246262, 35255160, 35255161,
                       35255162, 35255260, 35255261, 35255262, 35256160, 35256161, 35256162,
                       35256260, 35256261, 35256262, 35345160, 35345161, 35345162, 35345260,
                       35345261, 35345262, 35346160, 35346161, 35346162, 35346260, 35346261,
                       35346262, 35355160, 35355161, 35355162, 35355260, 35355261, 35355262,
                       35356160, 35356161, 35356162, 35356260, 35356261, 35356262, 45245160,
                       45245161, 45245162, 45245260, 45245261, 45245262, 45246160, 45246161,
                       45246162, 45246260, 45246261, 45246262, 45255160, 45255161, 45255162,
                       45255260, 45255261, 45255262, 45256160, 45256161, 45256162, 45256260,
                       45256261, 45256262, 45345160, 45345161, 45345162, 45345260, 45345261,
                       45345262, 45346160, 45346161, 45346162, 45346260, 45346261, 45346262,
                       45355160, 45355161, 45355162, 45355260, 45355261, 45355262, 45356160,
                       45356161, 45356162, 45356260, 45356261, 45356262, 55245160, 55245161,
                       55245162, 55245260, 55245261, 55245262, 55246160, 55246161, 55246162,
                       55246260, 55246261, 55246262, 55255160, 55255161, 55255162, 55255260,
                       55255261, 55255262, 55256160, 55256161, 55256162, 55256260, 55256261,
                       55256262, 55345160, 55345161, 55345162, 55345260, 55345261, 55345262,
                       55346160, 55346161, 55346162, 55346260, 55346261, 55346262, 55355160,
                       55355161, 55355162, 55355260, 55355261, 55355262, 55356160, 55356161,
                       55356162, 55356260, 55356261, 55356262};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                  start, stop, result, transposed));
}


INA_TEST_DATA(get_slice_buffer_trans) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(get_slice_buffer_trans) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(get_slice_buffer_trans) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(get_slice_buffer_trans, 2_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    const int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t pshape[] = {3, 4};
    int64_t start[] = {2, 1};
    int64_t stop[] = {7, 3};
    bool transposed = true;

    double result[] = {12, 22, 13, 23, 14, 24, 15, 25, 16, 26};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                  start, stop, result, transposed));
}

INA_TEST_FIXTURE(get_slice_buffer_trans, 2_f_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    const int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t *pshape = NULL;
    int64_t start[] = {3, 1};
    int64_t stop[] = {5, 8};
    bool transposed = true;

    float result[] = {13, 23, 33, 43, 53, 63, 73, 14, 24, 34, 44, 54, 64, 74};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                  start, stop, result, transposed));
}
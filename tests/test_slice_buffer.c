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
    void *buffer, uint64_t buflen) {

    INA_TEST_ASSERT_SUCCEED(iarray_slice_buffer(ctx, c_x, start, stop, buffer, buflen));

    return INA_SUCCESS;
}

static ina_rc_t _execute_iarray_slice(iarray_context_t *ctx, iarray_data_type_t dtype, size_t type_size, uint8_t ndim,
                                      const uint64_t *shape, const uint64_t *pshape,
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
        xdtshape.pshape[j] = pshape[j];
    }

    uint64_t bufdes_size = 1;

    for (int k = 0; k < ndim; ++k) {
        int64_t st = (start[k] + shape[k]) % shape[k];
        int64_t sp = (stop[k] + shape[k] - 1) % shape[k] + 1;
        bufdes_size *= (uint64_t) sp - st;
    }

    uint8_t *bufdes;

    uint64_t buflen = bufdes_size;

    if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
        buflen *= sizeof(double);
    } else {
        buflen *= sizeof(float);
    }

    bufdes = ina_mem_alloc(bufdes_size * sizeof(double));

    iarray_container_t *c_x;

    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &xdtshape, buffer_x, buffer_x_len * type_size, NULL, 0, &c_x));

    if (transposed == 1) {
        iarray_linalg_transpose(ctx, c_x);
    }

    INA_TEST_ASSERT_SUCCEED(test_slice_buffer(ctx, c_x, start, stop, bufdes, buflen));


    if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
        for (uint64_t l = 0; l < bufdes_size; ++l) {
            INA_TEST_ASSERT_EQUAL_FLOATING(((double *) bufdes)[l], ((double *) result)[l]);
        }
    } else {
        for (uint64_t l = 0; l < bufdes_size; ++l) {
            INA_TEST_ASSERT_EQUAL_FLOATING(((float *) bufdes)[l], ((float *) result)[l]);
        }
    }

    iarray_container_free(ctx, &c_x);

    ina_mem_free(buffer_x);
    ina_mem_free(bufdes);

    return INA_SUCCESS;
}

INA_TEST_DATA(slice_buffer) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(slice_buffer) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.flags = IARRAY_EXPR_EVAL_CHUNK;

    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(slice_buffer) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(slice_buffer, double_data_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    const uint64_t ndim = 2;
    uint64_t shape[] = {10, 10};
    uint64_t pshape[] = {3, 2};
    int64_t start[] = {5, -7};
    int64_t stop[] = {-1, 10};

    int transposed = 0;

    double result[] = {53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68, 69, 73, 74, 75, 76,
                           77, 78, 79, 83, 84, 85, 86, 87, 88, 89};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, pshape,
        start, stop, result, transposed));
}

INA_TEST_FIXTURE(slice_buffer, float_data_3) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    uint64_t const ndim = 3;
    uint64_t shape[] = {10, 10, 10};
    uint64_t pshape[] = {3, 5, 2};
    int64_t start[] = {-7, 0, 3};
    int64_t stop[] = {6, -3, 10};

    int transposed = 0;

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


INA_TEST_DATA(slice_buffer_trans) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(slice_buffer_trans) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.flags = IARRAY_EXPR_EVAL_CHUNK;

    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(slice_buffer_trans) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}


INA_TEST_FIXTURE(slice_buffer_trans, double_data_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    const uint64_t ndim = 2;
    uint64_t shape[] = {10, 10};
    uint64_t pshape[] = {3, 4};
    int64_t start[] = {2, 1};
    int64_t stop[] = {7, 3};

    int transposed = 1;

    double result[] = {12, 22, 13, 23, 14, 24, 15, 25, 16, 26};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                  start, stop, result, transposed));
}


INA_TEST_FIXTURE(slice_buffer_trans, float_data_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    const uint64_t ndim = 2;
    uint64_t shape[] = {10, 10};
    uint64_t pshape[] = {2, 7};
    int64_t start[] = {3, 1};
    int64_t stop[] = {5, 8};

    int transposed = 1;

    float result[] = {13, 23, 33, 43, 53, 63, 73, 14, 24, 34, 44, 54, 64, 74};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                  start, stop, result, transposed));
}
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

#include "iarray_test.h"
#include <libiarray/iarray.h>
#include <src/iarray_private.h>

static ina_rc_t test_rewrite_cont(iarray_context_t *ctx, iarray_data_type_t dtype,
                                  int32_t type_size, int8_t ndim, const int64_t *shape,
                                  const int64_t *cshape, const int64_t *bshape, const int64_t *blockshape,
                                  bool xcontiguous, char *xurlpath) {
    INA_UNUSED(type_size);
    // Create dtshape
    iarray_dtshape_t xdtshape;
    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
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

    INA_TEST_ASSERT_SUCCEED(iarray_empty(ctx, &xdtshape, &xstore, &c_x));

    // Start Iterator
    iarray_iter_write_block_t *I;
    iarray_iter_write_block_value_t val;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_write_block_new(ctx, &I, c_x, blockshape, &val, false));

    while (INA_SUCCEED(iarray_iter_write_block_has_next(I))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_write_block_next(I, NULL, 0));

        int64_t nelem = 0;
        int64_t inc = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            nelem += val.elem_index[i] * inc;
            inc *= c_x->dtshape->shape[i];
        }
        fill_block_iter(val, nelem, dtype);
    }

    iarray_iter_write_block_free(&I);


    // Start Iterator
    ina_rc_t err = iarray_iter_write_block_new(ctx, &I, c_x, blockshape, &val, false);
    if (err != 0) {
        return INA_SUCCESS;
    }

    while (INA_SUCCEED(iarray_iter_write_block_has_next(I))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_write_block_next(I, NULL, 0));

        switch (dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                for (int64_t i = 0; i < val.block_size; ++i) {
                    ((double *) val.block_pointer)[i] = 0;
                }
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                for (int64_t i = 0; i < val.block_size; ++i) {
                    ((float *) val.block_pointer)[i] = 0;
                }
                break;
            case IARRAY_DATA_TYPE_INT64:
                for (int64_t i = 0; i < val.block_size; ++i) {
                    ((int64_t *) val.block_pointer)[i] = 0;
                }
                break;
            default:
                return INA_ERR_EXCEEDED;
        }
    }

    // Start Read Iterator
    iarray_iter_read_t *itr_read;
    iarray_iter_read_value_t val_read;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_new(ctx, &itr_read, c_x, &val_read));

    while (INA_SUCCEED(iarray_iter_read_has_next(itr_read))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_read_next(itr_read));
        switch (dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                INA_TEST_ASSERT_EQUAL_FLOATING(((double *) val.block_pointer)[0], 0);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                INA_TEST_ASSERT_EQUAL_FLOATING(((float *) val.block_pointer)[0], 0);
                break;
            case IARRAY_DATA_TYPE_INT64:
                INA_TEST_ASSERT_EQUAL_INT64(((int64_t *) val.block_pointer)[0], 0);
                break;
            default:
                return INA_ERR_EXCEEDED;
        }
    }
    blosc2_remove_urlpath(xstore.urlpath);
    return INA_SUCCESS;
}


INA_TEST_DATA(rewrite_cont) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(rewrite_cont) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(rewrite_cont) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}


INA_TEST_FIXTURE(rewrite_cont, 3_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 3;
    int64_t shape[] = {120, 131, 155};
    int64_t cshape[] = {23, 32, 35};
    int64_t bshape[] = {7, 7, 5};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_rewrite_cont(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                              blockshape, false, "xarr.iarr"));
}

INA_TEST_FIXTURE(rewrite_cont, 4_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 4;
    int64_t shape[] = {30, 64, 50, 43};
    int64_t cshape[] = {11, 8, 12, 21};
    int64_t bshape[] = {4, 3, 3, 4};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_rewrite_cont(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                              blockshape, true, NULL));
}

INA_TEST_FIXTURE(rewrite_cont, 2_ll) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT64;
    int32_t type_size = sizeof(int64_t);

    int8_t ndim = 2;
    int64_t shape[] = {10, 8};
    int64_t cshape[] = {2, 3};
    int64_t bshape[] = {2, 3};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_rewrite_cont(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                              blockshape, false, "xarr.iarr"));
}

/* Avoid heavy tests
INA_TEST_FIXTURE(rewrite_cont, 7_ll) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT64;
    int32_t type_size = sizeof(int64_t);

    int8_t ndim = 7;
    int64_t shape[] = {10, 8, 6, 7, 13, 9, 10};
    int64_t cshape[] = {2, 3, 1, 3, 2, 4, 5};
    int64_t bshape[] = {2, 3, 1, 3, 2, 4, 5};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_rewrite_cont(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                              blockshape, false, "xarr.iarr"));
}
*/

INA_TEST_FIXTURE(rewrite_cont, 3_i) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT32;
    int32_t type_size = sizeof(int32_t);

    int8_t ndim = 3;
    int64_t shape[] = {120, 131, 155};
    int64_t cshape[] = {23, 32, 35};
    int64_t bshape[] = {7, 7, 5};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_rewrite_cont(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                              blockshape, false, "xarr.iarr"));
}

INA_TEST_FIXTURE(rewrite_cont, 4_s) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT16;
    int32_t type_size = sizeof(int16_t);

    int8_t ndim = 4;
    int64_t shape[] = {30, 64, 50, 43};
    int64_t cshape[] = {11, 8, 12, 21};
    int64_t bshape[] = {4, 3, 3, 4};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_rewrite_cont(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                              blockshape, true, NULL));
}

INA_TEST_FIXTURE(rewrite_cont, 3_sc) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT8;
    int32_t type_size = sizeof(int8_t);

    int8_t ndim = 3;
    int64_t shape[] = {10, 8, 6};
    int64_t cshape[] = {2, 3, 1};
    int64_t bshape[] = {2, 3, 1};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_rewrite_cont(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                              blockshape, false, "xarr.iarr"));
}

/* Avoid heavy tests
INA_TEST_FIXTURE(rewrite_cont, 7_sc) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT8;
    int32_t type_size = sizeof(int8_t);

    int8_t ndim = 7;
    int64_t shape[] = {10, 8, 6, 7, 13, 9, 10};
    int64_t cshape[] = {2, 3, 1, 3, 2, 4, 5};
    int64_t bshape[] = {2, 3, 1, 3, 2, 4, 5};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_rewrite_cont(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                              blockshape, false, "xarr.iarr"));
}
*/

INA_TEST_FIXTURE(rewrite_cont, 3_ull) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT64;
    int32_t type_size = sizeof(uint64_t);

    int8_t ndim = 3;
    int64_t shape[] = {120, 131, 155};
    int64_t cshape[] = {23, 32, 35};
    int64_t bshape[] = {7, 7, 5};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_rewrite_cont(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                              blockshape, false, "xarr.iarr"));
}

INA_TEST_FIXTURE(rewrite_cont, 4_ui) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT32;
    int32_t type_size = sizeof(uint32_t);

    int8_t ndim = 4;
    int64_t shape[] = {30, 64, 50, 43};
    int64_t cshape[] = {11, 8, 12, 21};
    int64_t bshape[] = {4, 3, 3, 4};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_rewrite_cont(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                              blockshape, true, NULL));
}

INA_TEST_FIXTURE(rewrite_cont, 2_us) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT16;
    int32_t type_size = sizeof(uint16_t);

    int8_t ndim = 2;
    int64_t shape[] = {10, 8};
    int64_t cshape[] = {2, 3};
    int64_t bshape[] = {2, 3};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_rewrite_cont(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                              blockshape, false, "xarr.iarr"));
}

/* Avoid heavy tests
INA_TEST_FIXTURE(rewrite_cont, 7_us) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT16;
    int32_t type_size = sizeof(uint16_t);

    int8_t ndim = 7;
    int64_t shape[] = {10, 8, 6, 7, 13, 9, 10};
    int64_t cshape[] = {2, 3, 1, 3, 2, 4, 5};
    int64_t bshape[] = {2, 3, 1, 3, 2, 4, 5};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_rewrite_cont(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                              blockshape, false, "xarr.iarr"));
}
*/

INA_TEST_FIXTURE(rewrite_cont, 3_uc) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT8;
    int32_t type_size = sizeof(uint8_t);

    int8_t ndim = 3;
    int64_t shape[] = {120, 131, 155};
    int64_t cshape[] = {23, 32, 35};
    int64_t bshape[] = {7, 7, 5};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_rewrite_cont(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                              blockshape, false, "xarr.iarr"));
}

INA_TEST_FIXTURE(rewrite_cont, 4_b) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_BOOL;
    int32_t type_size = sizeof(bool);

    int8_t ndim = 4;
    int64_t shape[] = {30, 64, 50, 43};
    int64_t cshape[] = {11, 8, 12, 21};
    int64_t bshape[] = {4, 3, 3, 4};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_rewrite_cont(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                              blockshape, true, NULL));
}


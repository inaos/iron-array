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

#include "iarray_test.h"
#include <libiarray/iarray.h>
#include <src/iarray_private.h>


static ina_rc_t test_block_iterator_transpose(iarray_context_t *ctx, iarray_data_type_t dtype,
                                              int32_t type_size, int8_t ndim, const int64_t *shape,
                                              const int64_t *cshape, const int64_t *bshape,
                                              const int64_t *blockshape, bool xcontiguous, char *xurlpath, bool ycontiguous, char *yurlpath)
{
    iarray_dtshape_t xdtshape;
    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    int64_t size = 1;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
        size *= shape[i];
    }

    iarray_dtshape_t ydtshape;
    ydtshape.dtype = dtype;
    ydtshape.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        ydtshape.shape[i] = shape[ndim - 1 - i];
    }

    iarray_storage_t xstorage;
    xstorage.contiguous = xcontiguous;
    xstorage.urlpath = xurlpath;
    for (int i = 0; i < ndim; ++i) {
        xstorage.chunkshape[i] = cshape[i];
        xstorage.blockshape[i] = bshape[i];
    }

    iarray_storage_t ystorage;
    ystorage.contiguous = ycontiguous;
    ystorage.urlpath = yurlpath;
    for (int i = 0; i < ndim; ++i) {
        ystorage.chunkshape[i] = cshape[ndim - 1 - i];
        ystorage.blockshape[i] = bshape[ndim - 1 - i];
    }

    iarray_container_t *c_x;
    blosc2_remove_urlpath(xstorage.urlpath);
    blosc2_remove_urlpath(ystorage.urlpath);

    INA_TEST_ASSERT_SUCCEED(iarray_empty(ctx, &xdtshape, &xstorage, &c_x));

    // Test write iterator
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
    
    INA_TEST_ASSERT(ina_err_get_rc() == INA_RC_PACK(IARRAY_ERR_END_ITER, 0));

    uint8_t *buf = ina_mem_alloc((size_t)c_x->catarr->nitems * type_size);

    iarray_container_t *c_trans;
    INA_TEST_ASSERT_SUCCEED(iarray_linalg_transpose(ctx, c_x, &c_trans));

    iarray_container_t *c_y;
    INA_TEST_ASSERT_SUCCEED(iarray_copy(ctx, c_trans, false, &ystorage, &c_y));

    // Test read iterator
    iarray_iter_read_block_t *I2;
    iarray_iter_read_block_value_t val2;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_new(ctx, &I2, c_trans, blockshape, &val2,
                                                       false));

    iarray_iter_read_block_t *I3;
    iarray_iter_read_block_value_t val3;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_new(ctx, &I3, c_y, blockshape, &val3, false));

    while (INA_SUCCEED(iarray_iter_read_block_has_next(I2)) && INA_SUCCEED(iarray_iter_read_block_has_next(I3))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_next(I2, NULL, 0));
        INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_next(I3, NULL, 0));

        switch (dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                for (int64_t i = 0; i < val2.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_FLOATING(((double *) val2.block_pointer)[i],
                        ((double *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_FLOATING(((float *) val2.block_pointer)[i],
                                                  ((float *) val3.block_pointer)[i]);
                }
                break;
            default:
                return INA_ERR_EXCEEDED;
        }
    }

    iarray_iter_read_block_free(&I2);
    iarray_iter_read_block_free(&I3);

    INA_TEST_ASSERT(ina_err_get_rc() == INA_RC_PACK(IARRAY_ERR_END_ITER, 0));

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);
    blosc2_remove_urlpath(xstorage.urlpath);
    blosc2_remove_urlpath(ystorage.urlpath);

    ina_mem_free(buf);

    return INA_SUCCESS;
}

INA_TEST_DATA(block_iterator_transpose) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(block_iterator_transpose) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(block_iterator_transpose) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}


INA_TEST_FIXTURE(block_iterator_transpose, d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {5, 5};
    int64_t bshape[] = {5, 5};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_transpose(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                blockshape, false, NULL, false, NULL));
}


INA_TEST_FIXTURE(block_iterator_transpose, f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 2;
    int64_t shape[] = {1340, 654};
    int64_t cshape[] = {600, 50};
    int64_t bshape[] = {135, 4};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_transpose(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                blockshape, true, "xarr.iarr", false, "yarr.iarr"));
}

INA_TEST_FIXTURE(block_iterator_transpose, d_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 2;
    int64_t shape[] = {1000, 55};
    int64_t cshape[] = {250, 20};
    int64_t bshape[] = {50, 10};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_transpose(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                blockshape, false, NULL, true, "yarr.iarr"));
}


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


static ina_rc_t test_block_iterator_transpose_external(iarray_context_t *ctx,
                                                       iarray_data_type_t dtype,
                                                       int32_t type_size,
                                                       int8_t ndim,
                                                       const int64_t *shape,
                                                       const int64_t *cshape,
                                                       const int64_t *bshape,
                                                       const int64_t *blockshape,
                                                       bool xcontiguous, char *xurlpath, bool ycontiguous, char *yurlpath)
{
    iarray_dtshape_t xdtshape;
    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    int64_t size = 1;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
        size *= shape[i];
    }

    iarray_storage_t xstorage;
    xstorage.contiguous = xcontiguous;
    xstorage.urlpath = xurlpath;
    for (int i = 0; i < ndim; ++i) {
        xstorage.chunkshape[i] = cshape[i];
        xstorage.blockshape[i] = bshape[i];
    }

    iarray_storage_t ystorage;
    ystorage.contiguous = ycontiguous;
    ystorage.urlpath = yurlpath;
    for (int i = 0; i < ndim; ++i) {
        ystorage.chunkshape[i] = cshape[ndim - 1 - i];
        ystorage.blockshape[i] = bshape[ndim - 1 - i];
    }

    iarray_container_t *c_x;
    blosc2_remove_urlpath(xstorage.urlpath);
    blosc2_remove_urlpath(ystorage.urlpath);

    INA_TEST_ASSERT_SUCCEED(iarray_empty(ctx, &xdtshape, &xstorage, &c_x));

    // Test write iterator
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

    INA_TEST_ASSERT(ina_err_get_rc() == INA_RC_PACK(IARRAY_ERR_END_ITER, 0));

    iarray_container_t *c_trans;
    INA_TEST_ASSERT_SUCCEED(iarray_linalg_transpose(ctx, c_x, &c_trans));

    iarray_container_t *c_y;
    INA_TEST_ASSERT_SUCCEED(iarray_copy(ctx, c_trans, false, &ystorage, &c_y));

    // Test read iterator
    iarray_iter_read_block_t *I2;
    iarray_iter_read_block_value_t val2;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_new(ctx, &I2, c_trans, blockshape, &val2,
                                                       false));

    iarray_iter_read_block_t *I3;
    iarray_iter_read_block_value_t val3;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_new(ctx, &I3, c_y, blockshape, &val3, false));

    while (INA_SUCCEED(iarray_iter_read_block_has_next(I2)) && INA_SUCCEED(iarray_iter_read_block_has_next(I3))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_next(I2, NULL, 0));
        INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_next(I3, NULL, 0));

        switch (dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                for (int64_t i = 0; i < val2.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_FLOATING(((double *) val2.block_pointer)[i],
                                                   ((double *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_FLOATING(((float *) val2.block_pointer)[i],
                                                   ((float *) val3.block_pointer)[i]);
                }
                break;
            default:
                return INA_ERR_EXCEEDED;
        }
    }

    iarray_iter_read_block_free(&I2);
    iarray_iter_read_block_free(&I3);

    INA_TEST_ASSERT(ina_err_get_rc() == INA_RC_PACK(IARRAY_ERR_END_ITER, 0));

    blosc2_remove_urlpath(xstorage.urlpath);
    blosc2_remove_urlpath(ystorage.urlpath);
    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);
    iarray_container_free(ctx, &c_trans);

    return INA_SUCCESS;
}

INA_TEST_DATA(block_iterator_transpose_external) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(block_iterator_transpose_external) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(block_iterator_transpose_external) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}


INA_TEST_FIXTURE(block_iterator_transpose_external, f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 2;
    int64_t shape[] = {1340, 654};
    int64_t cshape[] = {600, 50};
    int64_t bshape[] = {135, 4};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_transpose_external(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                          blockshape, true, "xarr.iarr", false, "yarr.iarr"));
}

INA_TEST_FIXTURE(block_iterator_transpose_external, d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 2;
    int64_t shape[] = {1000, 55};
    int64_t cshape[] = {250, 20};
    int64_t bshape[] = {50, 10};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_transpose_external(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                          blockshape, false, NULL, true, "yarr.iarr"));
}

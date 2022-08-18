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


static ina_rc_t test_block_iterator(iarray_context_t *ctx, iarray_data_type_t dtype,
                                    int32_t type_size, int8_t ndim, const int64_t *shape,
                                    const int64_t *cshape, const int64_t *bshape, const int64_t *blockshape,
                                    bool contiguous, char *urlpath)
{
    iarray_dtshape_t xdtshape;
    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
    }

    iarray_storage_t xstorage;
    xstorage.contiguous = contiguous;
    xstorage.urlpath = urlpath;
    for (int i = 0; i < ndim; ++i) {
        xstorage.chunkshape[i] = cshape[i];
        xstorage.blockshape[i] = bshape[i];
    }

    iarray_container_t *c_x;
    blosc2_remove_urlpath(xstorage.urlpath);

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
    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, buf, (size_t)c_x->catarr->nitems * type_size));

    iarray_container_t *c_y;
    if (xstorage.urlpath != NULL) {
        xstorage.urlpath = "yarr.iarr";
        blosc2_remove_urlpath(xstorage.urlpath);
    }
    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &xdtshape, buf,
                                               (size_t) c_x->catarr->nitems * type_size, &xstorage,
                                               &c_y));

    // Test read iterator
    iarray_iter_read_block_t *I2;
    iarray_iter_read_block_value_t val2;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_new(ctx, &I2, c_x, blockshape, &val2, false));

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
            case IARRAY_DATA_TYPE_INT64:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_INT64(((int64_t *) val2.block_pointer)[i],
                                                   ((int64_t *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_INT32:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_INT(((int32_t *) val2.block_pointer)[i],
                                                   ((int32_t *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_INT16:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_INT(((int16_t *) val2.block_pointer)[i],
                                                   ((int16_t *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_INT8:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_INT(((int8_t *) val2.block_pointer)[i],
                                                   ((int8_t *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_UINT64:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_UINT64(((uint64_t *) val2.block_pointer)[i],
                                                   ((uint64_t *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_UINT32:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_UINT(((uint32_t *) val2.block_pointer)[i],
                                                   ((uint32_t *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_UINT16:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_UINT(((uint16_t *) val2.block_pointer)[i],
                                                   ((uint16_t *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_UINT8:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_UINT(((uint8_t *) val2.block_pointer)[i],
                                                   ((uint8_t *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_BOOL:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT(((bool *) val2.block_pointer)[i] ==
                                                   ((bool *) val3.block_pointer)[i]);
                }
                break;
            default:
                return INA_ERR_EXCEEDED;
        }
    }

    iarray_iter_read_block_free(&I2);
    iarray_iter_read_block_free(&I3);

    INA_TEST_ASSERT(ina_err_get_rc() == INA_RC_PACK(IARRAY_ERR_END_ITER, 0));

    blosc2_remove_urlpath(urlpath);
    blosc2_remove_urlpath(xstorage.urlpath);
    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);

    ina_mem_free(buf);

    return INA_SUCCESS;
}

INA_TEST_DATA(block_iterator) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(block_iterator) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(block_iterator) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(block_iterator, 3_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 3;
    int64_t shape[] = {100, 200, 153};
    int64_t cshape[] = {23, 45, 71};
    int64_t bshape[] = {14, 5, 12};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                blockshape, true, NULL));
}

INA_TEST_FIXTURE(block_iterator, 4_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 4;
    int64_t shape[] = {30, 64, 50, 43};
    int64_t cshape[] = {11, 8, 12, 21};
    int64_t bshape[] = {5, 6, 10, 7};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                blockshape, true, NULL));
}

INA_TEST_FIXTURE(block_iterator, 3_ll) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT64;
    int32_t type_size = sizeof(int64_t);

    int8_t ndim = 3;
    int64_t shape[] = {10, 10, 6};
    int64_t cshape[] = {2, 4, 2};
    int64_t bshape[] = {1, 3, 2};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                blockshape, true, "arr.iarr"));
}

/* Avoid heavy tests
INA_TEST_FIXTURE(block_iterator, 7_ll) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT64;
    int32_t type_size = sizeof(int64_t);

    int8_t ndim = 7;
    int64_t shape[] = {10, 10, 6, 7, 13, 9, 10};
    int64_t cshape[] = {2, 4, 2, 3, 5, 4, 7};
    int64_t bshape[] = {1, 3, 2, 2, 4, 3, 3};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                blockshape, true, "arr.iarr"));
}
*/

INA_TEST_FIXTURE(block_iterator, 3_i) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT32;
    int32_t type_size = sizeof(int32_t);

    int8_t ndim = 3;
    int64_t shape[] = {100, 200, 153};
    int64_t cshape[] = {23, 45, 71};
    int64_t bshape[] = {14, 5, 12};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                blockshape, true, NULL));
}

INA_TEST_FIXTURE(block_iterator, 4_s) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT16;
    int32_t type_size = sizeof(int16_t);

    int8_t ndim = 4;
    int64_t shape[] = {30, 64, 50, 43};
    int64_t cshape[] = {11, 8, 12, 21};
    int64_t bshape[] = {5, 6, 10, 7};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                blockshape, true, NULL));
}

INA_TEST_FIXTURE(block_iterator, 2_sc) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT8;
    int32_t type_size = sizeof(int8_t);

    int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {2, 4};
    int64_t bshape[] = {1, 3};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                blockshape, true, "arr.iarr"));
}

/* Avoid heavy tests
INA_TEST_FIXTURE(block_iterator, 7_sc) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT8;
    int32_t type_size = sizeof(int8_t);

    int8_t ndim = 7;
    int64_t shape[] = {10, 10, 6, 7, 13, 9, 10};
    int64_t cshape[] = {2, 4, 2, 3, 5, 4, 7};
    int64_t bshape[] = {1, 3, 2, 2, 4, 3, 3};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                blockshape, true, "arr.iarr"));
}
*/

INA_TEST_FIXTURE(block_iterator, 3_ull) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT64;
    int32_t type_size = sizeof(uint64_t);

    int8_t ndim = 3;
    int64_t shape[] = {100, 200, 153};
    int64_t cshape[] = {23, 45, 71};
    int64_t bshape[] = {14, 5, 12};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                blockshape, true, NULL));
}

INA_TEST_FIXTURE(block_iterator, 4_ui) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT32;
    int32_t type_size = sizeof(uint32_t);

    int8_t ndim = 4;
    int64_t shape[] = {30, 64, 50, 43};
    int64_t cshape[] = {11, 8, 12, 21};
    int64_t bshape[] = {5, 6, 10, 7};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                blockshape, true, NULL));
}

INA_TEST_FIXTURE(block_iterator, 2_us) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT16;
    int32_t type_size = sizeof(uint16_t);

    int8_t ndim = 2;
    int64_t shape[] = {13, 9};
    int64_t cshape[] = {5, 4};
    int64_t bshape[] = {4, 3};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                blockshape, true, "arr.iarr"));
}

/* Avoid heavy tests
INA_TEST_FIXTURE(block_iterator, 7_us) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT16;
    int32_t type_size = sizeof(uint16_t);

    int8_t ndim = 7;
    int64_t shape[] = {10, 10, 6, 7, 13, 9, 10};
    int64_t cshape[] = {2, 4, 2, 3, 5, 4, 7};
    int64_t bshape[] = {1, 3, 2, 2, 4, 3, 3};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                blockshape, true, "arr.iarr"));
}
 */

INA_TEST_FIXTURE(block_iterator, 3_uc) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT8;
    int32_t type_size = sizeof(uint8_t);

    int8_t ndim = 3;
    int64_t shape[] = {100, 200, 153};
    int64_t cshape[] = {23, 45, 71};
    int64_t bshape[] = {14, 5, 12};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                blockshape, true, NULL));
}

INA_TEST_FIXTURE(block_iterator, 4_b) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_BOOL;
    int32_t type_size = sizeof(bool);

    int8_t ndim = 4;
    int64_t shape[] = {30, 64, 50, 43};
    int64_t cshape[] = {11, 8, 12, 21};
    int64_t bshape[] = {5, 6, 10, 7};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                blockshape, true, NULL));
}


static ina_rc_t test_block_iterator_ext_chunk(iarray_context_t *ctx, iarray_data_type_t dtype,
                                             int32_t type_size, int8_t ndim, const int64_t *shape,
                                             const int64_t *cshape, const int64_t *bshape, const int64_t *blockshape,
                                             bool contiguous, char *urlpath)
{
    // Create dtshape
    iarray_dtshape_t xdtshape;
    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
    }

    iarray_storage_t xstore;
    xstore.contiguous = contiguous;
    xstore.urlpath = urlpath;
    for (int i = 0; i < ndim; ++i) {
        xstore.chunkshape[i] = cshape[i];
        xstore.blockshape[i] = bshape[i];
    }
    iarray_container_t *c_x;
    blosc2_remove_urlpath(xstore.urlpath);

    INA_TEST_ASSERT_SUCCEED(iarray_empty(ctx, &xdtshape, &xstore, &c_x));

    // Start Iterator
    iarray_iter_write_block_t *I;
    iarray_iter_write_block_value_t val;

    int32_t csize_x = type_size;

    for (int i = 0; i < c_x->dtshape->ndim; ++i) {
        csize_x *= (int32_t) c_x->storage->chunkshape[i];
    }

    csize_x += BLOSC2_MAX_OVERHEAD;


    INA_TEST_ASSERT_SUCCEED(iarray_iter_write_block_new(ctx, &I, c_x, blockshape, &val, true));
    uint8_t *chunk_x;

    while (INA_SUCCEED(iarray_iter_write_block_has_next(I))) {
        chunk_x = (uint8_t *) malloc(csize_x);

        INA_TEST_ASSERT_SUCCEED(iarray_iter_write_block_next(I, (void *) chunk_x, csize_x));

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
    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, buf, (size_t)c_x->catarr->nitems * type_size));

    iarray_container_t *c_y;
    if (xstore.urlpath != NULL) {
        xstore.urlpath = "yarr.iarr";
        blosc2_remove_urlpath(xstore.urlpath);
    }
    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &xdtshape, buf,
                                               (size_t) c_x->catarr->nitems * type_size, &xstore,
                                               &c_y));

    // Start Iterator
    iarray_iter_read_block_t *I2;
    iarray_iter_read_block_value_t val2;

    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_new(ctx, &I2, c_x, blockshape, &val2, true));

    iarray_iter_read_block_t *I3;
    iarray_iter_read_block_value_t val3;

    int32_t csize_y = c_y->catarr->chunknitems * type_size;

    csize_y += BLOSC2_MAX_OVERHEAD;

    uint8_t *chunk_y1 = (uint8_t *) malloc(csize_y);
    uint8_t *chunk_y2 = (uint8_t *) malloc(csize_y);

    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_new(ctx, &I3, c_y, blockshape, &val3, true));

    while (INA_SUCCEED(iarray_iter_read_block_has_next(I2)) && INA_SUCCEED(iarray_iter_read_block_has_next(I3))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_next(I2, (void *) chunk_y1, csize_x));
        INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_next(I3, (void *) chunk_y2, csize_y));

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
            case IARRAY_DATA_TYPE_INT64:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_INT64(((int64_t *) val2.block_pointer)[i],
                                                   ((int64_t *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_INT32:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_INT(((int32_t *) val2.block_pointer)[i],
                                                   ((int32_t *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_INT16:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_INT(((int16_t *) val2.block_pointer)[i],
                                                   ((int16_t *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_INT8:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_INT(((int8_t *) val2.block_pointer)[i],
                                                   ((int8_t *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_UINT64:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_UINT64(((uint64_t *) val2.block_pointer)[i],
                                                   ((uint64_t *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_UINT32:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_UINT(((uint32_t *) val2.block_pointer)[i],
                                                   ((uint32_t *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_UINT16:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_UINT(((uint16_t *) val2.block_pointer)[i],
                                                   ((uint16_t *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_UINT8:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_UINT(((uint8_t *) val2.block_pointer)[i],
                                                   ((uint8_t *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_BOOL:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT(((bool *) val2.block_pointer)[i] ==
                                                   ((bool *) val3.block_pointer)[i]);
                }
                break;
            default:
                return INA_ERR_EXCEEDED;
        }
    }

    iarray_iter_read_block_free(&I2);
    iarray_iter_read_block_free(&I3);
    INA_TEST_ASSERT(ina_err_get_rc() == INA_RC_PACK(IARRAY_ERR_END_ITER, 0));

    blosc2_remove_urlpath(urlpath);
    blosc2_remove_urlpath(xstore.urlpath);
    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);

    free(chunk_y1);
    free(chunk_y2);
    ina_mem_free(buf);

    return INA_SUCCESS;
}

INA_TEST_DATA(block_iterator_ext_chunk) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(block_iterator_ext_chunk) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(block_iterator_ext_chunk) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(block_iterator_ext_chunk, 3_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 3;
    int64_t shape[] = {120, 131, 155};
    int64_t cshape[] = {23, 32, 35};
    int64_t bshape[] = {17, 3, 4};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_ext_chunk(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                         blockshape, true, NULL));
}

INA_TEST_FIXTURE(block_iterator_ext_chunk, 2_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 2;
    int64_t shape[] = {50, 43};
    int64_t cshape[] = {12, 21};
    int64_t bshape[] = {5, 5};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_ext_chunk(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                          blockshape, false, "arr.iarr"));
}

/* Avoid heavy tests
INA_TEST_FIXTURE(block_iterator_ext_chunk, 4_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 4;
    int64_t shape[] = {30, 64, 50, 43};
    int64_t cshape[] = {11, 8, 12, 21};
    int64_t bshape[] = {5, 5, 5, 5};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_ext_chunk(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                         blockshape, false, "arr.iarr"));
}
*/

INA_TEST_FIXTURE(block_iterator_ext_chunk, 1_ll) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT64;
    int32_t type_size = sizeof(int64_t);

    int8_t ndim = 1;
    int64_t shape[] = {10};
    int64_t cshape[] = {2};
    int64_t bshape[] = {2};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_ext_chunk(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                          blockshape, false, NULL));
}

/* Avoid heavy tests
INA_TEST_FIXTURE(block_iterator_ext_chunk, 7_ll) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT64;
    int32_t type_size = sizeof(int64_t);

    int8_t ndim = 7;
    int64_t shape[] = {10, 8, 6, 7, 13, 9, 10};
    int64_t cshape[] = {2, 3, 4, 4, 2, 4, 5};
    int64_t bshape[] = {2, 1, 2, 2, 1, 2, 3};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_ext_chunk(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                         blockshape, false, NULL));
}
*/

INA_TEST_FIXTURE(block_iterator_ext_chunk, 3_i) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT32;
    int32_t type_size = sizeof(int32_t);

    int8_t ndim = 3;
    int64_t shape[] = {120, 131, 155};
    int64_t cshape[] = {23, 32, 35};
    int64_t bshape[] = {17, 3, 4};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_ext_chunk(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                          blockshape, true, NULL));
}

INA_TEST_FIXTURE(block_iterator_ext_chunk, 3_s) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT16;
    int32_t type_size = sizeof(int16_t);

    int8_t ndim = 3;
    int64_t shape[] = {64, 50, 43};
    int64_t cshape[] = {8, 12, 21};
    int64_t bshape[] = {5, 5, 5};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_ext_chunk(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                          blockshape, false, "arr.iarr"));
}

/* Avoid heavy tests
INA_TEST_FIXTURE(block_iterator_ext_chunk, 4_s) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT16;
    int32_t type_size = sizeof(int16_t);

    int8_t ndim = 4;
    int64_t shape[] = {30, 64, 50, 43};
    int64_t cshape[] = {11, 8, 12, 21};
    int64_t bshape[] = {5, 5, 5, 5};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_ext_chunk(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                          blockshape, false, "arr.iarr"));
}
*/

INA_TEST_FIXTURE(block_iterator_ext_chunk, 7_sc) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT8;
    int32_t type_size = sizeof(int8_t);

    int8_t ndim = 7;
    int64_t shape[] = {10, 8, 6, 7, 13, 9, 10};
    int64_t cshape[] = {2, 3, 4, 4, 2, 4, 5};
    int64_t bshape[] = {2, 1, 2, 2, 1, 2, 3};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_ext_chunk(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                          blockshape, false, NULL));
}

INA_TEST_FIXTURE(block_iterator_ext_chunk, 3_b) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_BOOL;
    int32_t type_size = sizeof(bool);

    int8_t ndim = 3;
    int64_t shape[] = {120, 131, 155};
    int64_t cshape[] = {23, 32, 35};
    int64_t bshape[] = {17, 3, 4};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_ext_chunk(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                          blockshape, true, NULL));
}

INA_TEST_FIXTURE(block_iterator_ext_chunk, 2_ull) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT64;
    int32_t type_size = sizeof(uint64_t);

    int8_t ndim = 2;
    int64_t shape[] = {30, 64};
    int64_t cshape[] = {11, 8};
    int64_t bshape[] = {5, 5};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_ext_chunk(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                          blockshape, false, "arr.iarr"));
}

/* Avoid heavy tests
INA_TEST_FIXTURE(block_iterator_ext_chunk, 4_ull) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT64;
    int32_t type_size = sizeof(uint64_t);

    int8_t ndim = 4;
    int64_t shape[] = {30, 64, 50, 43};
    int64_t cshape[] = {11, 8, 12, 21};
    int64_t bshape[] = {5, 5, 5, 5};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_ext_chunk(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                          blockshape, false, "arr.iarr"));
}
*/

INA_TEST_FIXTURE(block_iterator_ext_chunk, 7_ui) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT32;
    int32_t type_size = sizeof(uint32_t);

    int8_t ndim = 7;
    int64_t shape[] = {10, 8, 6, 7, 13, 9, 10};
    int64_t cshape[] = {2, 3, 4, 4, 2, 4, 5};
    int64_t bshape[] = {2, 1, 2, 2, 1, 2, 3};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_ext_chunk(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                                blockshape, false, NULL));
}


INA_TEST_FIXTURE(block_iterator_ext_chunk, 3_us) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT16;
    int32_t type_size = sizeof(uint16_t);

    int8_t ndim = 3;
    int64_t shape[] = {120, 131, 155};
    int64_t cshape[] = {23, 32, 35};
    int64_t bshape[] = {17, 3, 4};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_ext_chunk(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                          blockshape, true, NULL));
}

INA_TEST_FIXTURE(block_iterator_ext_chunk, 3_c) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT8;
    int32_t type_size = sizeof(uint8_t);

    int8_t ndim = 3;
    int64_t shape[] = {30, 64, 43};
    int64_t cshape[] = {11, 8, 21};
    int64_t bshape[] = {5, 5, 5};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_ext_chunk(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                          blockshape, false, "arr.iarr"));
}

/* Avoid heavy tests
INA_TEST_FIXTURE(block_iterator_ext_chunk, 4_c) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT8;
    int32_t type_size = sizeof(uint8_t);

    int8_t ndim = 4;
    int64_t shape[] = {30, 64, 50, 43};
    int64_t cshape[] = {11, 8, 12, 21};
    int64_t bshape[] = {5, 5, 5, 5};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_ext_chunk(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                          blockshape, false, "arr.iarr"));
}
*/

static ina_rc_t test_block_iterator_not_empty(iarray_context_t *ctx, iarray_data_type_t dtype,
                                    int32_t type_size, int8_t ndim, const int64_t *shape,
                                    const int64_t *cshape,  const int64_t *bshape, const int64_t *blockshape,
                                    bool contiguous, char *urlpath)
{
    // Create dtshape
    iarray_dtshape_t xdtshape;
    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
    }

    iarray_storage_t xstore;
    xstore.contiguous = contiguous;
    xstore.urlpath = urlpath;
    for (int i = 0; i < ndim; ++i) {
        xstore.chunkshape[i] = cshape[i];
        xstore.blockshape[i] = bshape[i];
    }

    iarray_container_t *c_x;
    blosc2_remove_urlpath(xstore.urlpath);
    INA_TEST_ASSERT_SUCCEED(iarray_arange(ctx, &xdtshape, 0, 1, &xstore, &c_x));

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
    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, buf, (size_t)c_x->catarr->nitems * type_size));

    iarray_container_t *c_y;
    if (xstore.urlpath != NULL) {
        xstore.urlpath = "yarr.iarr";
        blosc2_remove_urlpath(xstore.urlpath);
    }
    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &xdtshape, buf,
                                               (size_t) c_x->catarr->nitems * type_size, &xstore,
                                               &c_y));

    // Test read iterator
    iarray_iter_read_block_t *I2;
    iarray_iter_read_block_value_t val2;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_new(ctx, &I2, c_x, blockshape, &val2, false));

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
            case IARRAY_DATA_TYPE_INT64:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_INT64(((int64_t *) val2.block_pointer)[i],
                                                   ((int64_t *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_INT32:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_INT(((int32_t *) val2.block_pointer)[i],
                                                   ((int32_t *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_INT16:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_INT(((int16_t *) val2.block_pointer)[i],
                                                   ((int16_t *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_INT8:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_INT(((int8_t *) val2.block_pointer)[i],
                                                   ((int8_t *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_UINT64:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_UINT64(((uint64_t *) val2.block_pointer)[i],
                                                   ((uint64_t *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_UINT32:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_UINT(((uint32_t *) val2.block_pointer)[i],
                                                   ((uint32_t *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_UINT16:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_UINT(((uint16_t *) val2.block_pointer)[i],
                                                   ((uint16_t *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_UINT8:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_UINT(((uint8_t *) val2.block_pointer)[i],
                                                   ((uint8_t *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_BOOL:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT(((bool *) val2.block_pointer)[i] ==
                                                   ((bool *) val3.block_pointer)[i]);
                }
                break;
            default:
                return INA_ERR_EXCEEDED;
        }
    }

    iarray_iter_read_block_free(&I2);
    iarray_iter_read_block_free(&I3);
    INA_TEST_ASSERT(ina_err_get_rc() == INA_RC_PACK(IARRAY_ERR_END_ITER, 0));

    blosc2_remove_urlpath(urlpath);
    blosc2_remove_urlpath(xstore.urlpath);
    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);

    ina_mem_free(buf);

    return INA_SUCCESS;
}


INA_TEST_DATA(block_iterator_not_empty) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(block_iterator_not_empty) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(block_iterator_not_empty) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(block_iterator_not_empty, 2_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {5, 5};
    int64_t bshape[] =  {3, 3};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_not_empty(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                          blockshape, true, NULL));
}

INA_TEST_FIXTURE(block_iterator_not_empty, 5_ll) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT64;
    int32_t type_size = sizeof(int64_t);

    int8_t ndim = 5;
    int64_t shape[] = {10, 10, 12, 22, 15};
    int64_t cshape[] = {10, 10, 12, 22, 15};
    int64_t bshape[] = {5, 5, 5, 5, 5};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_not_empty(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                          blockshape, false, "arr.iarr"));

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_not_empty(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                          blockshape, true, "arr.iarr"));
}

INA_TEST_FIXTURE(block_iterator_not_empty, 2_i) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT32;
    int32_t type_size = sizeof(int32_t);

    int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {5, 5};
    int64_t bshape[] =  {3, 3};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_not_empty(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                          blockshape, true, NULL));
}

INA_TEST_FIXTURE(block_iterator_not_empty, 5_s) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT16;
    int32_t type_size = sizeof(int16_t);

    int8_t ndim = 5;
    int64_t shape[] = {10, 10, 12, 22, 15};
    int64_t cshape[] = {10, 10, 12, 22, 15};
    int64_t bshape[] = {5, 5, 5, 5, 5};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_not_empty(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                          blockshape, false, "arr.iarr"));

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_not_empty(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                          blockshape, true, "arr.iarr"));
}

INA_TEST_FIXTURE(block_iterator_not_empty, 2_sc) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT8;
    int32_t type_size = sizeof(int8_t);

    int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {5, 5};
    int64_t bshape[] =  {3, 3};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_not_empty(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                          blockshape, true, NULL));
}

INA_TEST_FIXTURE(block_iterator_not_empty, 5_ull) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT64;
    int32_t type_size = sizeof(uint64_t);

    int8_t ndim = 5;
    int64_t shape[] = {10, 10, 12, 22, 15};
    int64_t cshape[] = {10, 10, 12, 22, 15};
    int64_t bshape[] = {5, 5, 5, 5, 5};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_not_empty(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                          blockshape, false, "arr.iarr"));

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_not_empty(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                          blockshape, true, "arr.iarr"));
}

INA_TEST_FIXTURE(block_iterator_not_empty, 2_ui) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT32;
    int32_t type_size = sizeof(uint32_t);

    int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {5, 5};
    int64_t bshape[] =  {3, 3};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_not_empty(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                          blockshape, true, NULL));
}

INA_TEST_FIXTURE(block_iterator_not_empty, 5_us) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT16;
    int32_t type_size = sizeof(uint16_t);

    int8_t ndim = 5;
    int64_t shape[] = {10, 10, 12, 22, 15};
    int64_t cshape[] = {10, 10, 12, 22, 15};
    int64_t bshape[] = {5, 5, 5, 5, 5};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_not_empty(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                          blockshape, false, "arr.iarr"));

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_not_empty(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                          blockshape, true, "arr.iarr"));
}

INA_TEST_FIXTURE(block_iterator_not_empty, 2_uc) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT8;
    int32_t type_size = sizeof(uint8_t);

    int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {5, 5};
    int64_t bshape[] =  {3, 3};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_not_empty(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                          blockshape, true, NULL));
}

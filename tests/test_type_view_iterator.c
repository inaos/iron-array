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


static ina_rc_t test_type_block_iterator(iarray_context_t *ctx, iarray_data_type_t dtype,
                                        iarray_data_type_t view_dtype, int8_t ndim,
                                        const int64_t *shape, const int64_t *cshape, const int64_t *bshape,
                                        const int64_t *blockshape, bool contiguous, char *urlpath)
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

    iarray_container_t *c_view;
    INA_TEST_ASSERT_SUCCEED(iarray_get_type_view(ctx, c_x, view_dtype, &c_view));

    // Test read iterator
    iarray_iter_read_block_t *I2;
    iarray_iter_read_block_value_t val2;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_new(ctx, &I2, c_x, blockshape, &val2, false));

    iarray_iter_read_block_t *I3;
    iarray_iter_read_block_value_t val3;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_new(ctx, &I3, c_view, blockshape, &val3, false));

    while (INA_SUCCEED(iarray_iter_read_block_has_next(I2))
           && INA_SUCCEED(iarray_iter_read_block_has_next(I3))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_next(I2, NULL, 0));
        INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_next(I3, NULL, 0));

        switch (dtype) {
            case IARRAY_DATA_TYPE_DOUBLE: {
                double *src_ptr = (double *) val2.block_pointer;
                switch (view_dtype) {
                    case IARRAY_DATA_TYPE_INT64:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_INT64((int64_t)src_ptr[i],
                                                        ((int64_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_UINT64:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_UINT64((uint64_t)src_ptr[i],
                                                         ((uint64_t *) val3.block_pointer)[i]);
                        }
                        break;
                    default:
                        return INA_ERR_EXCEEDED;
                }
                break;
            }
            case IARRAY_DATA_TYPE_FLOAT: {
                float *src_ptr = (float *) val2.block_pointer;
                switch (view_dtype) {
                    case IARRAY_DATA_TYPE_DOUBLE:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((double)src_ptr[i],
                                                           ((double *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_INT64:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((int64_t)src_ptr[i],
                                                           ((int64_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_INT32:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((int32_t)src_ptr[i],
                                                           ((int32_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_UINT64:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((uint64_t)src_ptr[i],
                                                           ((uint64_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_UINT32:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((uint32_t)src_ptr[i],
                                                           ((uint32_t *) val3.block_pointer)[i]);
                        }
                        break;
                    default:
                        return INA_ERR_EXCEEDED;
                }
                break;
            }
            case IARRAY_DATA_TYPE_INT64: {
                int64_t *src_ptr = (int64_t *) val2.block_pointer;
                switch (view_dtype) {
                    case IARRAY_DATA_TYPE_DOUBLE: {
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((double)src_ptr[i],
                                                           ((double *) val3.block_pointer)[i]);
                        }
                        break;
                    }
                    case IARRAY_DATA_TYPE_UINT64:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((uint64_t)src_ptr[i],
                                                           ((uint64_t *) val3.block_pointer)[i]);
                        }
                        break;
                    default:
                        return INA_ERR_EXCEEDED;
                }
                break;
            }
            case IARRAY_DATA_TYPE_INT32: {
                int32_t *src_ptr = (int32_t *) val2.block_pointer;
                switch (view_dtype) {
                    case IARRAY_DATA_TYPE_DOUBLE:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((double)src_ptr[i],
                                                           ((double *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_FLOAT:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((float)src_ptr[i],
                                                           ((float *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_INT64:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING(src_ptr[i],
                                                           ((int64_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_UINT64:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((uint64_t)src_ptr[i],
                                                           ((uint64_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_UINT32:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((uint32_t)src_ptr[i],
                                                           ((uint32_t *) val3.block_pointer)[i]);
                        }
                        break;
                    default:
                        return INA_ERR_EXCEEDED;
                }
                break;
            }
            case IARRAY_DATA_TYPE_INT16: {
                int16_t *src_ptr = (int16_t *) val2.block_pointer;
                switch (view_dtype) {
                    case IARRAY_DATA_TYPE_DOUBLE:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((double)src_ptr[i],
                                                           ((double *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_FLOAT:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((float)src_ptr[i],
                                                           ((float *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_INT64:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING(src_ptr[i],
                                                           ((int64_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_INT32:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((int32_t)src_ptr[i],
                                                           ((int32_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_UINT64:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((uint64_t)src_ptr[i],
                                                           ((uint64_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_UINT32:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((uint32_t)src_ptr[i],
                                                           ((uint32_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_UINT16:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((uint16_t)src_ptr[i],
                                                           ((uint16_t *) val3.block_pointer)[i]);
                        }
                        break;
                    default:
                        return INA_ERR_EXCEEDED;
                }
                break;
            }
            case IARRAY_DATA_TYPE_INT8: {
                int8_t *src_ptr = (int8_t *) val2.block_pointer;
                switch (view_dtype) {
                    case IARRAY_DATA_TYPE_DOUBLE:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((double)src_ptr[i],
                                                           ((double *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_FLOAT:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((float)src_ptr[i],
                                                           ((float *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_INT64:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING(src_ptr[i],
                                                           ((int64_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_INT32:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((int32_t)src_ptr[i],
                                                           ((int32_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_INT16:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((int16_t)src_ptr[i],
                                                           ((int16_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_UINT64:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((uint64_t)src_ptr[i],
                                                           ((uint64_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_UINT32:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((uint32_t)src_ptr[i],
                                                           ((uint32_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_UINT16:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((uint16_t)src_ptr[i],
                                                           ((uint16_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_UINT8:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((uint8_t)src_ptr[i],
                                                           ((uint8_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_BOOL:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((bool)src_ptr[i],
                                                           ((bool *) val3.block_pointer)[i]);
                        }
                        break;
                    default:
                        return INA_ERR_EXCEEDED;
                }
                break;
            }
            case IARRAY_DATA_TYPE_UINT64: {
                uint64_t *src_ptr = (uint64_t *) val2.block_pointer;
                switch (view_dtype) {
                    case IARRAY_DATA_TYPE_DOUBLE:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((double)src_ptr[i],
                                                           ((double *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_INT64:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING(src_ptr[i],
                                                           ((int64_t *) val3.block_pointer)[i]);
                        }
                        break;
                    default:
                        return INA_ERR_EXCEEDED;
                }
                break;
            }
            case IARRAY_DATA_TYPE_UINT32: {
                uint32_t *src_ptr = (uint32_t *) val2.block_pointer;
                switch (view_dtype) {
                    case IARRAY_DATA_TYPE_DOUBLE:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((double)src_ptr[i],
                                                           ((double *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_FLOAT:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((float)src_ptr[i],
                                                           ((float *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_INT64:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING(src_ptr[i],
                                                           ((int64_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_INT32:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((int32_t)src_ptr[i],
                                                           ((int32_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_UINT64:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((uint64_t)src_ptr[i],
                                                           ((uint64_t *) val3.block_pointer)[i]);
                        }
                        break;
                    default:
                        return INA_ERR_EXCEEDED;
                }
                break;
            }
            case IARRAY_DATA_TYPE_UINT16: {
                uint16_t *src_ptr = (uint16_t *) val2.block_pointer;
                switch (view_dtype) {
                    case IARRAY_DATA_TYPE_DOUBLE:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((double)src_ptr[i],
                                                           ((double *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_FLOAT:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((float)src_ptr[i],
                                                           ((float *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_INT64:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING(src_ptr[i],
                                                           ((int64_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_INT32:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((int32_t)src_ptr[i],
                                                           ((int32_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_INT16:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((int16_t)src_ptr[i],
                                                           ((int16_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_UINT64:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((uint64_t)src_ptr[i],
                                                           ((uint64_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_UINT32:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((uint32_t)src_ptr[i],
                                                           ((uint32_t *) val3.block_pointer)[i]);
                        }
                        break;
                    default:
                        return INA_ERR_EXCEEDED;
                }
                break;
            }
            case IARRAY_DATA_TYPE_UINT8: {
                uint8_t *src_ptr = (uint8_t *) val2.block_pointer;
                switch (view_dtype) {
                    case IARRAY_DATA_TYPE_DOUBLE:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((double)src_ptr[i],
                                                           ((double *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_FLOAT:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((float)src_ptr[i],
                                                           ((float *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_INT64:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING(src_ptr[i],
                                                           ((int64_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_INT32:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((int32_t)src_ptr[i],
                                                           ((int32_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_INT16:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((int16_t)src_ptr[i],
                                                           ((int16_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_INT8:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((int8_t)src_ptr[i],
                                                           ((int8_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_UINT64:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((uint64_t)src_ptr[i],
                                                           ((uint64_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_UINT32:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((uint32_t)src_ptr[i],
                                                           ((uint32_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_UINT16:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((uint16_t)src_ptr[i],
                                                           ((uint16_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_BOOL:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((bool)src_ptr[i],
                                                           ((bool *) val3.block_pointer)[i]);
                        }
                        break;
                    default:
                        return INA_ERR_EXCEEDED;
                }
                break;
            }
            case IARRAY_DATA_TYPE_BOOL: {
                bool *src_ptr = (bool *) val2.block_pointer;
                switch (view_dtype) {
                    case IARRAY_DATA_TYPE_DOUBLE:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((double)src_ptr[i],
                                                           ((double *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_FLOAT:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((float)src_ptr[i],
                                                           ((float *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_INT64:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING(src_ptr[i],
                                                           ((int64_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_INT32:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((int32_t)src_ptr[i],
                                                           ((int32_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_INT16:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((int16_t)src_ptr[i],
                                                           ((int16_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_INT8:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((int8_t)src_ptr[i],
                                                           ((int8_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_UINT64:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((uint64_t)src_ptr[i],
                                                           ((uint64_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_UINT32:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((uint32_t)src_ptr[i],
                                                           ((uint32_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_UINT16:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((uint16_t)src_ptr[i],
                                                           ((uint16_t *) val3.block_pointer)[i]);
                        }
                        break;
                    case IARRAY_DATA_TYPE_UINT8:
                        for (int64_t i = 0; i < val2.block_size; ++i) {
                            INA_TEST_ASSERT_EQUAL_FLOATING((uint8_t)src_ptr[i],
                                                           ((uint8_t *) val3.block_pointer)[i]);
                        }
                        break;
                    default:
                        return INA_ERR_EXCEEDED;
                }
                break;
            }
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
    iarray_container_free(ctx, &c_view);

    return INA_SUCCESS;
}

INA_TEST_DATA(type_block_iterator) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(type_block_iterator) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.max_num_threads = 3;
    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(type_block_iterator) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(type_block_iterator, 3_f_ui64) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_UINT64;

    int8_t ndim = 3;
    int64_t shape[] = {100, 200, 153};
    int64_t cshape[] = {23, 45, 71};
    int64_t bshape[] = {14, 5, 12};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_type_block_iterator(data->ctx, dtype, view_dtype, ndim, shape, cshape, bshape,
                                                     blockshape, true, NULL));
}

INA_TEST_FIXTURE(type_block_iterator, 4_d_i64) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_INT64;

    int8_t ndim = 4;
    int64_t shape[] = {30, 12, 50, 43};
    int64_t cshape[] = {11, 8, 12, 21};
    int64_t bshape[] = {5, 6, 10, 7};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_type_block_iterator(data->ctx, dtype, view_dtype, ndim, shape, cshape, bshape,
                                                     blockshape, true, NULL));
}

INA_TEST_FIXTURE(type_block_iterator, 3_ll_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT64;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 3;
    int64_t shape[] = {10, 10, 6};
    int64_t cshape[] = {2, 4, 2};
    int64_t bshape[] = {1, 3, 2};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_type_block_iterator(data->ctx, dtype, view_dtype, ndim, shape, cshape, bshape,
                                                     blockshape, true, "arr.iarr"));
}

INA_TEST_FIXTURE(type_block_iterator, 3_i_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT32;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 3;
    int64_t shape[] = {100, 200, 153};
    int64_t cshape[] = {23, 45, 71};
    int64_t bshape[] = {14, 5, 12};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_type_block_iterator(data->ctx, dtype, view_dtype, ndim, shape, cshape, bshape,
                                                     blockshape, true, NULL));
}

INA_TEST_FIXTURE(type_block_iterator, 4_s_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT16;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 4;
    int64_t shape[] = {30, 64, 20, 43};
    int64_t cshape[] = {11, 8, 12, 21};
    int64_t bshape[] = {5, 6, 10, 7};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_type_block_iterator(data->ctx, dtype, view_dtype, ndim, shape, cshape, bshape,
                                                     blockshape, true, NULL));
}

INA_TEST_FIXTURE(type_block_iterator, 2_sc_b) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT8;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_BOOL;

    int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {2, 4};
    int64_t bshape[] = {1, 3};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_type_block_iterator(data->ctx, dtype, view_dtype, ndim, shape, cshape, bshape,
                                                     blockshape, true, "arr.iarr"));
}

INA_TEST_FIXTURE(type_block_iterator, 3_ull_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT64;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 3;
    int64_t shape[] = {100, 200, 153};
    int64_t cshape[] = {23, 45, 71};
    int64_t bshape[] = {14, 5, 12};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_type_block_iterator(data->ctx, dtype, view_dtype, ndim, shape, cshape, bshape,
                                                     blockshape, true, NULL));
}

INA_TEST_FIXTURE(type_block_iterator, 2_us_i) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT16;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_INT32;

    int8_t ndim = 2;
    int64_t shape[] = {13, 9};
    int64_t cshape[] = {5, 4};
    int64_t bshape[] = {4, 3};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_type_block_iterator(data->ctx, dtype, view_dtype, ndim, shape, cshape, bshape,
                                                     blockshape, true, "arr.iarr"));
}

INA_TEST_FIXTURE(type_block_iterator, 3_uc_c) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT8;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_INT8;

    int8_t ndim = 3;
    int64_t shape[] = {100, 200, 153};
    int64_t cshape[] = {23, 45, 71};
    int64_t bshape[] = {14, 5, 12};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_type_block_iterator(data->ctx, dtype, view_dtype, ndim, shape, cshape, bshape,
                                                     blockshape, true, NULL));
}

INA_TEST_FIXTURE(type_block_iterator, 4_b_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_BOOL;
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 4;
    int64_t shape[] = {30, 64, 50, 43};
    int64_t cshape[] = {11, 8, 12, 21};
    int64_t bshape[] = {5, 6, 10, 7};
    int64_t *blockshape = cshape;

    INA_TEST_ASSERT_SUCCEED(test_type_block_iterator(data->ctx, dtype, view_dtype, ndim, shape, cshape, bshape,
                                                     blockshape, true, NULL));
}

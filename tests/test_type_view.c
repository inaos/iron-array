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


static ina_rc_t execute_iarray_type_view(iarray_context_t *ctx, iarray_data_type_t src_dtype, int32_t src_type_size,
                                          iarray_data_type_t view_dtype, int8_t ndim,
                                      const int64_t *shape, const int64_t *cshape, const int64_t *bshape,
                                      bool xcontiguous, char *xurlpath) {
    void *buffer_x;
    int64_t buffer_x_len;

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

    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &xdtshape, buffer_x, buffer_x_len * src_type_size,
                                               &xstore, &c_x));
    iarray_container_t *c_out;
    INA_TEST_ASSERT_SUCCEED(iarray_get_type_view(ctx, c_x, view_dtype, &c_out));

    uint8_t *bufdes;
    switch (src_dtype) {
        case IARRAY_DATA_TYPE_DOUBLE: {
            double *bufsrc = ina_mem_alloc(buffer_x_len * src_type_size);
            INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, bufsrc, buffer_x_len * src_type_size));
            switch (view_dtype) {
                case IARRAY_DATA_TYPE_INT64:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(int64_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(int64_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_INT64(((int64_t *) bufdes)[l], (int64_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_UINT64:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(uint64_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(uint64_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_UINT64(((uint64_t *) bufdes)[l], (uint64_t) bufsrc[l]);
                    }
                    break;
                default:
                    return INA_ERR_EXCEEDED;
            }
            ina_mem_free(bufsrc);
            break;
        }
        case IARRAY_DATA_TYPE_FLOAT: {
            float *bufsrc = ina_mem_alloc(buffer_x_len * src_type_size);
            INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, bufsrc, buffer_x_len * src_type_size));
            switch (view_dtype) {
                case IARRAY_DATA_TYPE_DOUBLE:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(double));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(double)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_FLOATING(((double *) bufdes)[l], (double) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_INT64:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(int64_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(int64_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_INT64(((int64_t *) bufdes)[l], (int64_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_INT32:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(int32_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(int32_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_INT(((int32_t *) bufdes)[l], (int32_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_UINT64:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(uint64_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(uint64_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_UINT64(((uint64_t *) bufdes)[l], (uint64_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_UINT32:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(uint32_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(uint32_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_UINT(((uint32_t *) bufdes)[l], (uint32_t) bufsrc[l]);
                    }
                    break;
                default:
                    return INA_ERR_EXCEEDED;
            }
            ina_mem_free(bufsrc);
            break;
        }
        case IARRAY_DATA_TYPE_INT64: {
            int64_t *bufsrc = ina_mem_alloc(buffer_x_len * src_type_size);
            INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, bufsrc, buffer_x_len * src_type_size));
            switch (view_dtype) {
                case IARRAY_DATA_TYPE_DOUBLE: {
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(double));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(double)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_FLOATING(((double *) bufdes)[l], (double) bufsrc[l]);
                    }
                    break;
                }
                case IARRAY_DATA_TYPE_UINT64:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(uint64_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(uint64_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_UINT64(((uint64_t *) bufdes)[l], (uint64_t) bufsrc[l]);
                    }
                    break;
                default:
                    return INA_ERR_EXCEEDED;
            }
            ina_mem_free(bufsrc);
            break;
        }
        case IARRAY_DATA_TYPE_INT32: {
            int32_t *bufsrc = ina_mem_alloc(buffer_x_len * src_type_size);
            INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, bufsrc, buffer_x_len * src_type_size));
            switch (view_dtype) {
                case IARRAY_DATA_TYPE_DOUBLE:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(double));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(double)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_FLOATING(((double *) bufdes)[l], (double) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_FLOAT:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(float));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(float)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_FLOATING(((float *) bufdes)[l], (float) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_INT64:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(int64_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(int64_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_INT64(((int64_t *) bufdes)[l], (int64_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_UINT64:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(uint64_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(uint64_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_UINT64(((uint64_t *) bufdes)[l], (uint64_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_UINT32:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(uint32_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(uint32_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_UINT(((uint32_t *) bufdes)[l], (uint32_t) bufsrc[l]);
                    }
                    break;
                default:
                    return INA_ERR_EXCEEDED;
            }
            ina_mem_free(bufsrc);
            break;
        }
        case IARRAY_DATA_TYPE_INT16: {
            int16_t *bufsrc = ina_mem_alloc(buffer_x_len * src_type_size);
            INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, bufsrc, buffer_x_len * src_type_size));
            switch (view_dtype) {
                case IARRAY_DATA_TYPE_DOUBLE:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(double));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(double)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_FLOATING(((double *) bufdes)[l], (double) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_FLOAT:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(float));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(int64_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_INT64(((float *) bufdes)[l], (float) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_INT64:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(int64_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(int64_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_INT64(((int64_t *) bufdes)[l], (int64_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_INT32:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(int32_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(int32_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_INT(((int32_t *) bufdes)[l], (int32_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_UINT64:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(uint64_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(uint64_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_UINT64(((uint64_t *) bufdes)[l], (uint64_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_UINT32:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(uint32_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(uint32_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_UINT(((uint32_t *) bufdes)[l], (uint32_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_UINT16:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(uint16_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(uint16_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_UINT(((uint16_t *) bufdes)[l], (uint16_t) bufsrc[l]);
                    }
                    break;
                default:
                    return INA_ERR_EXCEEDED;
            }
            ina_mem_free(bufsrc);
            break;
        }
        case IARRAY_DATA_TYPE_INT8: {
            int8_t *bufsrc = ina_mem_alloc(buffer_x_len * src_type_size);
            INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, bufsrc, buffer_x_len * src_type_size));
            switch (view_dtype) {
                case IARRAY_DATA_TYPE_DOUBLE:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(double));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(double)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_FLOATING(((double *) bufdes)[l], (double) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_FLOAT:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(float));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(float)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_FLOATING(((float *) bufdes)[l], (float) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_INT64:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(int64_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(int64_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_INT64(((int64_t *) bufdes)[l], (int64_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_INT32:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(int32_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(int32_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_INT(((int32_t *) bufdes)[l], (int32_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_INT16:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(int16_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(int16_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_INT(((int16_t *) bufdes)[l], (int16_t ) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_UINT64:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(uint64_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(uint64_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_UINT64(((uint64_t *) bufdes)[l], (uint64_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_UINT32:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(uint32_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(uint32_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_UINT(((uint32_t *) bufdes)[l], (uint32_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_UINT16:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(uint16_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(uint16_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_UINT(((uint16_t *) bufdes)[l], (uint16_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_UINT8:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(uint8_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(uint8_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_UINT(((uint8_t *) bufdes)[l], (uint8_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_BOOL:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(bool));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(bool)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT(((bool *) bufdes)[l] == (bool) bufsrc[l]);
                    }
                    break;
                default:
                    return INA_ERR_EXCEEDED;
            }
            ina_mem_free(bufsrc);
            break;
        }
        case IARRAY_DATA_TYPE_UINT64: {
            uint64_t *bufsrc = ina_mem_alloc(buffer_x_len * src_type_size);
            INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, bufsrc, buffer_x_len * src_type_size));
            switch (view_dtype) {
                case IARRAY_DATA_TYPE_DOUBLE:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(double));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(double)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_FLOATING(((double *) bufdes)[l], (double) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_INT64:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(int64_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(int64_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_INT64(((int64_t *) bufdes)[l], (int64_t) bufsrc[l]);
                    }
                    break;
                default:
                    return INA_ERR_EXCEEDED;
            }
            ina_mem_free(bufsrc);
            break;
        }
        case IARRAY_DATA_TYPE_UINT32: {
            uint32_t *bufsrc = ina_mem_alloc(buffer_x_len * src_type_size);
            INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, bufsrc, buffer_x_len * src_type_size));
            switch (view_dtype) {
                case IARRAY_DATA_TYPE_DOUBLE:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(double));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(double)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_FLOATING(((double *) bufdes)[l], (double) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_FLOAT:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(float));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(float)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_FLOATING(((float *) bufdes)[l], (float) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_INT64:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(int64_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(int64_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_INT64(((int64_t *) bufdes)[l], (int64_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_INT32:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(int32_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(int32_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_INT(((int32_t *) bufdes)[l], (int32_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_UINT64:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(uint64_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(uint64_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_UINT64(((uint64_t *) bufdes)[l], (uint64_t) bufsrc[l]);
                    }
                    break;
                default:
                    return INA_ERR_EXCEEDED;
            }
            ina_mem_free(bufsrc);
            break;
        }
        case IARRAY_DATA_TYPE_UINT16: {
            uint16_t *bufsrc = ina_mem_alloc(buffer_x_len * src_type_size);
            INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, bufsrc, buffer_x_len * src_type_size));
            switch (view_dtype) {
                case IARRAY_DATA_TYPE_DOUBLE:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(double));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(double)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_FLOATING(((double *) bufdes)[l], (double) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_FLOAT:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(float));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(float)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_FLOATING(((float *) bufdes)[l], (float) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_INT64:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(int64_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(int64_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_INT64(((int64_t *) bufdes)[l], (int64_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_INT32:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(int32_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(int32_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_INT(((int32_t *) bufdes)[l], (int32_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_INT16:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(int16_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(int16_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_INT(((int16_t *) bufdes)[l], (int16_t ) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_UINT64:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(uint64_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(uint64_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_UINT64(((uint64_t *) bufdes)[l], (uint64_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_UINT32:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(uint32_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(uint32_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_UINT(((uint32_t *) bufdes)[l], (uint32_t) bufsrc[l]);
                    }
                    break;
                default:
                    return INA_ERR_EXCEEDED;
            }
            ina_mem_free(bufsrc);
            break;
        }
        case IARRAY_DATA_TYPE_UINT8: {
            uint8_t *bufsrc = ina_mem_alloc(buffer_x_len * src_type_size);
            INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, bufsrc, buffer_x_len * src_type_size));
            switch (view_dtype) {
                case IARRAY_DATA_TYPE_DOUBLE:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(double));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(double)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_FLOATING(((double *) bufdes)[l], (double) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_FLOAT:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(float));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(float)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_FLOATING(((float *) bufdes)[l], (float) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_INT64:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(int64_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(int64_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_INT64(((int64_t *) bufdes)[l], (int64_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_INT32:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(int32_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(int32_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_INT(((int32_t *) bufdes)[l], (int32_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_INT16:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(int16_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(int16_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_INT(((int16_t *) bufdes)[l], (int16_t ) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_INT8:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(int8_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(int8_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_INT(((int8_t *) bufdes)[l], (int8_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_UINT64:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(uint64_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(uint64_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_UINT64(((uint64_t *) bufdes)[l], (uint64_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_UINT32:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(uint32_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(uint32_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_UINT(((uint32_t *) bufdes)[l], (uint32_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_UINT16:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(uint16_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(uint16_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_UINT(((uint16_t *) bufdes)[l], (uint16_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_BOOL:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(bool));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(bool)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT(((bool *) bufdes)[l] == (bool) bufsrc[l]);
                    }
                    break;
                default:
                    return INA_ERR_EXCEEDED;
            }
            ina_mem_free(bufsrc);
            break;
        }
        case IARRAY_DATA_TYPE_BOOL: {
            bool *bufsrc = ina_mem_alloc(buffer_x_len * src_type_size);
            INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, bufsrc, buffer_x_len * src_type_size));
            switch (view_dtype) {
                case IARRAY_DATA_TYPE_DOUBLE:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(double));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(double)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_FLOATING(((double *) bufdes)[l], (double) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_FLOAT:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(float));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(float)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_FLOATING(((float *) bufdes)[l], (float) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_INT64:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(int64_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(int64_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_INT64(((int64_t *) bufdes)[l], (int64_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_INT32:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(int32_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(int32_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_INT(((int32_t *) bufdes)[l], (int32_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_INT16:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(int16_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(int16_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_INT(((int16_t *) bufdes)[l], (int16_t ) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_INT8:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(int8_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(int8_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_INT(((int8_t *) bufdes)[l], (int8_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_UINT64:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(uint64_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(uint64_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_UINT64(((uint64_t *) bufdes)[l], (uint64_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_UINT32:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(uint32_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(uint32_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_UINT(((uint32_t *) bufdes)[l], (uint32_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_UINT16:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(uint16_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(uint16_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_UINT(((uint16_t *) bufdes)[l], (uint16_t) bufsrc[l]);
                    }
                    break;
                case IARRAY_DATA_TYPE_UINT8:
                    bufdes = ina_mem_alloc(buffer_x_len * sizeof(uint8_t));
                    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_out, bufdes, buffer_x_len * sizeof(uint8_t)));
                    for (int64_t l = 0; l < buffer_x_len; ++l) {
                        INA_TEST_ASSERT_EQUAL_UINT(((uint8_t *) bufdes)[l], (uint8_t) bufsrc[l]);
                    }
                    break;
                default:
                    return INA_ERR_EXCEEDED;
            }
            ina_mem_free(bufsrc);
            break;
        }
        default:
            return INA_ERR_EXCEEDED;
    }


    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_out);
    blosc2_remove_urlpath(xstore.urlpath);

    ina_mem_free(buffer_x);
    ina_mem_free(bufdes);



    return INA_SUCCESS;
}

INA_TEST_DATA(type_view) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(type_view) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.max_num_threads = 4;
    iarray_context_new(&cfg, &data->ctx);

}

INA_TEST_TEARDOWN(type_view) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(type_view, 3_f_ll) {
    iarray_data_type_t src_dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t src_type_size = sizeof(float);
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_INT64;

    const int8_t ndim = 3;
    int64_t shape[] = {10, 10, 10};
    int64_t cshape[] = {9, 5, 4};
    int64_t bshape[] = {5, 5, 2};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_type_view(data->ctx, src_dtype, src_type_size, view_dtype, ndim, shape, cshape, bshape,
                                                      true, "xarr.iarr"));
}

INA_TEST_FIXTURE(type_view, 4_ll_d) {
    iarray_data_type_t src_dtype = IARRAY_DATA_TYPE_INT64;
    int32_t src_type_size = sizeof(int64_t);
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_DOUBLE;

    const int8_t ndim = 4;
    int64_t shape[] = {10, 10, 10, 10};
    int64_t cshape[] = {3, 5, 2, 7};
    int64_t bshape[] = {2, 2, 2, 4};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_type_view(data->ctx, src_dtype, src_type_size, view_dtype, ndim, shape, cshape, bshape,
                                                      false, NULL));
}

INA_TEST_FIXTURE(type_view, 2_uc_ll) {
    iarray_data_type_t src_dtype = IARRAY_DATA_TYPE_UINT8;
    int32_t src_type_size = sizeof(uint8_t);
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_INT64;

    const int8_t ndim = 2;
    int64_t shape[] = {100, 100};
    int64_t cshape[] = {40, 50};
    int64_t bshape[] = {20, 20};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_type_view(data->ctx, src_dtype, src_type_size, view_dtype, ndim, shape, cshape, bshape,
                                                      false, "xarr.iarr"));
}

INA_TEST_FIXTURE(type_view, 3_s_f) {
    iarray_data_type_t src_dtype = IARRAY_DATA_TYPE_INT16;
    int32_t src_type_size = sizeof(int16_t);
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_FLOAT;

    const int8_t ndim = 3;
    int64_t shape[] = {10, 10, 10};
    int64_t cshape[] = {4, 5, 1};
    int64_t bshape[] = {2, 2, 1};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_type_view(data->ctx, src_dtype, src_type_size, view_dtype, ndim, shape, cshape, bshape,
                                                      false, NULL));
}

INA_TEST_FIXTURE(type_view, 3_us_d) {
    iarray_data_type_t src_dtype = IARRAY_DATA_TYPE_UINT16;
    int32_t src_type_size = sizeof(uint16_t);
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_DOUBLE;

    const int8_t ndim = 3;
    int64_t shape[] = {10, 10, 10};
    int64_t cshape[] = {5, 4, 5};
    int64_t bshape[] = {2, 2, 1};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_type_view(data->ctx, src_dtype, src_type_size, view_dtype, ndim, shape, cshape, bshape,
                                                      true, NULL));
}

INA_TEST_FIXTURE(type_view, 2_b_ui) {
    iarray_data_type_t src_dtype = IARRAY_DATA_TYPE_BOOL;
    int32_t src_type_size = sizeof(bool);
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_UINT32;

    const int8_t ndim = 2;
    int64_t shape[] = {100, 100};
    int64_t cshape[] = {9, 50};
    int64_t bshape[] = {5, 50};



    INA_TEST_ASSERT_SUCCEED(execute_iarray_type_view(data->ctx, src_dtype, src_type_size, view_dtype, ndim, shape, cshape, bshape,
                                                     true, "xarr.iarr"));
}

INA_TEST_FIXTURE(type_view, 4_ull_d) {
    iarray_data_type_t src_dtype = IARRAY_DATA_TYPE_UINT64;
    int32_t src_type_size = sizeof(uint64_t);
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_DOUBLE;

    const int8_t ndim = 4;
    int64_t shape[] = {10, 10, 10, 10};
    int64_t cshape[] = {3, 5, 2, 7};
    int64_t bshape[] = {2, 2, 2, 4};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_type_view(data->ctx, src_dtype, src_type_size, view_dtype, ndim, shape, cshape, bshape,
                                                     false, NULL));
}

INA_TEST_FIXTURE(type_view, 3_ui_f) {
    iarray_data_type_t src_dtype = IARRAY_DATA_TYPE_UINT32;
    int32_t src_type_size = sizeof(uint32_t);
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_FLOAT;

    const int8_t ndim = 3;
    int64_t shape[] = {10, 10, 10};
    int64_t cshape[] = {4, 5, 1};
    int64_t bshape[] = {2, 2, 1};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_type_view(data->ctx, src_dtype, src_type_size, view_dtype, ndim, shape, cshape, bshape,
                                                      false, "xarr.iarr"));
}

INA_TEST_FIXTURE(type_view, 3_uc_b) {
    iarray_data_type_t src_dtype = IARRAY_DATA_TYPE_UINT8;
    int32_t src_type_size = sizeof(uint8_t);
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_BOOL;

    const int8_t ndim = 3;
    int64_t shape[] = {10, 10, 10};
    int64_t cshape[] = {5, 4, 5};
    int64_t bshape[] = {2, 2, 1};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_type_view(data->ctx, src_dtype, src_type_size, view_dtype, ndim, shape, cshape, bshape,
                                                      true, NULL));
}

INA_TEST_FIXTURE(type_view, 2_c_b) {
    iarray_data_type_t src_dtype = IARRAY_DATA_TYPE_INT8;
    int32_t src_type_size = sizeof(int8_t);
    iarray_data_type_t view_dtype = IARRAY_DATA_TYPE_BOOL;

    const int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {4, 5};
    int64_t bshape[] = {2, 2};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_type_view(data->ctx, src_dtype, src_type_size, view_dtype, ndim, shape, cshape, bshape,
                                                      false, "xarr.iarr"));
}

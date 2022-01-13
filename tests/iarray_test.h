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

#ifndef _IARRAY_TEST_COMMON_H_
#define _IARRAY_TEST_COMMON_H_

#include <libiarray/iarray.h>
#include <stdbool.h>

inline static void fill_buf(iarray_data_type_t dtype, void *x, size_t nitems)
{
    switch (dtype) {
        case IARRAY_DATA_TYPE_DOUBLE: {
            for (size_t i = 0; i < nitems; i++) {
                ((double*)x)[i] = (double) i;
            }
            break;
        }
        case IARRAY_DATA_TYPE_FLOAT: {
            for (size_t i = 0; i < nitems; i++) {
                ((float *)x)[i] = (float) i;
            }
            break;
        }
        case IARRAY_DATA_TYPE_INT64: {
            for (size_t i = 0; i < nitems; i++) {
                ((int64_t *)x)[i] = (int64_t) i;
            }
            break;
        }
        case IARRAY_DATA_TYPE_INT32: {
            int32_t *aux = (int32_t *)x;
            for (size_t i = 0; i < nitems; i++) {
                ((int32_t*)x)[i] = (int32_t) i;
            }
            break;
        }
        case IARRAY_DATA_TYPE_INT16: {
            for (size_t i = 0; i < nitems; i++) {
                ((int16_t*)x)[i] = (int16_t) i;
            }
            break;
        }
        case IARRAY_DATA_TYPE_INT8: {
            for (size_t i = 0; i < nitems; i++) {
                ((int8_t*)x)[i] = (int8_t) i;
            }
            break;
        }
        case IARRAY_DATA_TYPE_UINT64: {
            for (size_t i = 0; i < nitems; i++) {
                ((uint64_t *)x)[i] = (uint64_t) i;
            }
            break;
        }
        case IARRAY_DATA_TYPE_UINT32: {
            for (size_t i = 0; i < nitems; i++) {
                ((uint32_t*)x)[i] = (uint32_t) i;
            }
            break;
        }
        case IARRAY_DATA_TYPE_UINT16: {
            for (size_t i = 0; i < nitems; i++) {
                ((uint16_t*)x)[i] = (uint16_t) i;
            }
            break;
        }
        case IARRAY_DATA_TYPE_UINT8: {
            for (size_t i = 0; i < nitems; i++) {
                ((uint8_t*)x)[i] = (uint8_t) i;
            }
            break;
        }
        case IARRAY_DATA_TYPE_BOOL: {
            for (size_t i = 0; i < nitems; i++) {
                ((bool *)x)[i] = (bool) (i % 2);
            }
            break;
        }
    }
}

inline static ina_rc_t test_double_buffer_cmp(iarray_context_t *ctx, iarray_container_t *c, const double *buffer,
                                              size_t buffer_len, double atol, double rtol) {
    double *bufcmp = ina_mem_alloc(buffer_len);

    INA_RETURN_IF_FAILED(iarray_to_buffer(ctx, c, bufcmp, buffer_len));

    size_t len = buffer_len / sizeof(double);
    for (size_t i = 0; i < len; ++i) {
        double a = buffer[i];
        double b = bufcmp[i];
        double adiff = fabs(a - b);
        if (adiff > atol + (rtol * fabs(b)) ) {
            INA_TEST_MSG("Values differ in (%d nelem) (diff: %g) (%g - %g)\n", i, adiff, a, b);
            IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_FALSE));
        }
    }
    ina_mem_free(bufcmp);
    return INA_SUCCESS;

fail:
    ina_mem_free(bufcmp);
    return ina_err_get_rc();
}

inline static ina_rc_t test_float_buffer_cmp(iarray_context_t *ctx, iarray_container_t *c, const float *buffer,
                                             size_t buffer_len, double atol, double rtol) {
    float *bufcmp = ina_mem_alloc(buffer_len);

    INA_RETURN_IF_FAILED(iarray_to_buffer(ctx, c, bufcmp, buffer_len));

    size_t len = buffer_len / sizeof(float);
    for (size_t i = 0; i < len; ++i) {
        double a = buffer[i];
        double b = bufcmp[i];
        double adiff = fabs(a - b);
        if (adiff > atol + (rtol * fabs(b))) {
            INA_TEST_MSG("Values differ in (%d nelem) (diff: %g)(%g - %g)\n", i, adiff, a, b);
            IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_FALSE));
        }
    }
    ina_mem_free(bufcmp);
    return INA_SUCCESS;

    fail:
    ina_mem_free(bufcmp);
    return ina_err_get_rc();
}

inline static void fill_block_iter(iarray_iter_write_block_value_t val, int64_t nelem, iarray_data_type_t dtype) {
    switch (dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            for (int64_t i = 0; i < val.block_size; ++i) {
                ((double *) val.block_pointer)[i] = (double) (nelem + i);
            }
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            for (int64_t i = 0; i < val.block_size; ++i) {
                ((float *) val.block_pointer)[i] = (float) (nelem  + i);
            }
            break;
        case IARRAY_DATA_TYPE_INT64:
            for (int64_t i = 0; i < val.block_size; ++i) {
                ((int64_t *) val.block_pointer)[i] = (int64_t) nelem  + i;
            }
            break;
        case IARRAY_DATA_TYPE_INT32:
            for (int64_t i = 0; i < val.block_size; ++i) {
                ((int32_t *) val.block_pointer)[i] = (int32_t) (nelem  + i);
            }
            break;
        case IARRAY_DATA_TYPE_INT16:
            for (int64_t i = 0; i < val.block_size; ++i) {
                ((int16_t *) val.block_pointer)[i] = (int16_t) (nelem  + i);
            }
            break;
        case IARRAY_DATA_TYPE_INT8:
            for (int64_t i = 0; i < val.block_size; ++i) {
                ((int8_t *) val.block_pointer)[i] = (int8_t) (nelem  + i);
            }
            break;
        case IARRAY_DATA_TYPE_UINT64:
            for (int64_t i = 0; i < val.block_size; ++i) {
                ((uint64_t *) val.block_pointer)[i] = (uint64_t) (nelem  + i);
            }
            break;
        case IARRAY_DATA_TYPE_UINT32:
            for (int64_t i = 0; i < val.block_size; ++i) {
                ((uint32_t *) val.block_pointer)[i] = (uint32_t) (nelem  + i);
            }
            break;
        case IARRAY_DATA_TYPE_UINT16:
            for (int64_t i = 0; i < val.block_size; ++i) {
                ((uint16_t *) val.block_pointer)[i] = (uint16_t) (nelem  + i);
            }
            break;
        case IARRAY_DATA_TYPE_UINT8:
            for (int64_t i = 0; i < val.block_size; ++i) {
                ((uint8_t *) val.block_pointer)[i] = (uint8_t) (nelem  + i);
            }
            break;
        case IARRAY_DATA_TYPE_BOOL:
            for (int64_t i = 0; i < val.block_size; ++i) {
                ((bool *) val.block_pointer)[i] = (bool) ((nelem  + i)%2);
            }
            break;
    }
}

#endif

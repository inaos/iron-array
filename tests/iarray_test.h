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

inline static void ffill_buf(float *x, size_t nitems)
{
    /* Fill with even values between 0 and 10 */

    for (size_t i = 0; i < nitems; i++) {
        x[i] = (float)i;
    }
}

inline static void dfill_buf(double *x, size_t nitems)
{
    /* Fill with even values between 0 and 10 */

    for (size_t i = 0; i < nitems; i++) {
        x[i] = (double)i;
    }
}

inline static ina_rc_t _iarray_test_container_dbl_buffer_cmp(
    iarray_context_t *ctx, iarray_container_t *c, const double *buffer, size_t buffer_len, double atol)
{
    double *bufcmp = ina_mem_alloc(buffer_len);

    INA_RETURN_IF_FAILED(iarray_to_buffer(ctx, c, bufcmp, buffer_len));

    size_t len = buffer_len / sizeof(double);
    for (size_t i = 0; i < len; ++i) {
        double a = buffer[i];
        double b = bufcmp[i];
        double vdiff = fabs(a - b);
        if (vdiff > atol) {
            INA_TEST_MSG("Values differ in (%d nelem) (diff: %g)\n", i, vdiff);
            INA_FAIL_IF_ERROR(INA_ERROR(INA_ERR_FALSE));
        }
    }
    ina_mem_free(bufcmp);
    return INA_SUCCESS;

fail:
    ina_mem_free(bufcmp);
    return ina_err_get_rc();
}

#endif

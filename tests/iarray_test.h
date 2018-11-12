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

static void ffill_buf(float *x, size_t nitems)
{

    /* Fill with even values between 0 and 10 */
    float incx = (float) 10. / nitems;

    for (size_t i = 0; i < nitems; i++) {
        x[i] = incx * i;
    }
}

static void dfill_buf(double *x, size_t nitems)
{

    /* Fill with even values between 0 and 10 */
    double incx = 10. / nitems;

    for (size_t i = 0; i < nitems; i++) {
        x[i] = incx * i;
    }
}

#endif

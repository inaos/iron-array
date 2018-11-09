//
// Created by Aleix Alcacer Sales on 5/11/18.
//


#ifndef IARRAY_TEST_COMMON_H
#define IARRAY_TEST_COMMON_H

#include <libiarray/iarray.h>

void ffill_buf(float *x, size_t nitems) {

    /* Fill with even values between 0 and 10 */
    float incx = (float) 10. / nitems;

    for (size_t i = 0; i < nitems; i++) {
        x[i] = incx * i;
    }
}

void dfill_buf(double *x, size_t nitems) {

    /* Fill with even values between 0 and 10 */
    double incx = 10. / nitems;

    for (size_t i = 0; i < nitems; i++) {
        x[i] = incx * i;
    }
}

#endif //IARRAY_TEST_COMMON_H

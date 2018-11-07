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

int fmm_mul(size_t M, size_t K, size_t N, float const *a, float const *b, float *c) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t n= 0; n < N; ++n) {
            for (size_t k = 0; k < K; ++k) {
                c[m * N + n] += a[m * K + k] * b[k * N + n];
            }
        }
    }
    return 0;
}

int dmm_mul(size_t M, size_t K, size_t N, double const *a, double const *b, double *c) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t n= 0; n < N; ++n) {
            for (size_t k = 0; k < K; ++k) {
                c[m * N + n] += a[m * K + k] * b[k * N + n];
            }
        }
    }
    return 0;
}

int fmv_mul(size_t M, size_t K, float const *a, float const *b, float *c) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t k = 0; k < K; ++k) {
            c[m] += a[m * K + k] * b[k];
        }

    }
    return 0;
}

int dmv_mul(size_t M, size_t K, double const *a, double const *b, double *c) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t k = 0; k < K; ++k) {
            c[m] += a[m * K + k] * b[k];
        }

    }
    return 0;
}

#endif //IARRAY_TEST_COMMON_H

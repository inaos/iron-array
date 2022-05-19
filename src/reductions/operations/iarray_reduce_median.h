/*
 * Copyright ironArray SL 2022.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of ironArray SL
 * ("Confidential Information"). You shall not disclose such Confidential
 * Information and shall use it only in accordance with the terms of the license agreement.
 *
 */

#ifndef IARRAY_IARRAY_REDUCE_MEDIAN_H
#define IARRAY_IARRAY_REDUCE_MEDIAN_H

#include "iarray_reduce_private.h"

#define MEDIAN_R \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0);  \
    user_data_t *u_data = (user_data_t *) user_data; \
    *data0 = 0; \
    cdata1 = malloc(u_data->input_itemsize * nelem);             \
    for (int i = 0; i < nelem; ++i) { \
        cdata1[i] = *data1; \
        data1 += strides1; \
    }            \
    qsort(cdata1, nelem, u_data->input_itemsize, compare);       \
    if (nelem % 2 == 0) {      \
        *data0 = (double) ((cdata1[(int64_t) (nelem / 2 - 1)] + cdata1[(int64_t) (nelem / 2)]) * 0.5);            \
    } else {     \
        *data0 = (double) cdata1[(int64_t) (nelem / 2)];             \
    }\
    free(cdata1);


#define COMPARE_RETURN return *a > *b ? 1 : (*a < *b ? -1 : 0);
static int iarray_dmedian_compare(const double *a, const double *b) {
    COMPARE_RETURN
}
static void dmedian_red(DPARAMS_R) {
    double *cdata1;
    int (*compare)(const void *a, const void *b) = (int(*)(const void *, const void*)) iarray_dmedian_compare;
    MEDIAN_R
}

static iarray_reduce_function_t DMEDIAN = {
        .init = NULL,
        .reduction = CAST_R dmedian_red,
        .finish = NULL,
};

#define FMEDIAN_R \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0);  \
    user_data_t *u_data = (user_data_t *) user_data; \
    *data0 = 0; \
    cdata1 = malloc(u_data->input_itemsize * nelem);             \
    for (int i = 0; i < nelem; ++i) { \
        cdata1[i] = *data1; \
        data1 += strides1; \
    }            \
    qsort(cdata1, nelem, u_data->input_itemsize, compare);       \
    if (nelem % 2 == 0) {      \
        *data0 = (float) ((cdata1[(int64_t) (nelem / 2 - 1)] + cdata1[(int64_t) (nelem / 2)]) * 0.5f);            \
    } else {     \
        *data0 = (float) cdata1[(int64_t) (nelem / 2)];             \
    }\
    free(cdata1);

static int iarray_fmedian_compare(const float *a, const float *b) {
    COMPARE_RETURN
}
static void fmedian_red(FPARAMS_R) {
    float *cdata1;
    int (*compare)(const void *a, const void *b) = (int(*)(const void *, const void*)) iarray_fmedian_compare;
    FMEDIAN_R
}

static iarray_reduce_function_t FMEDIAN = {
        .init = NULL,
        .reduction = CAST_R fmedian_red,
        .finish = NULL,
};

static int iarray_i64median_compare(const int64_t *a, const int64_t *b) {
    COMPARE_RETURN
}

static void i64median_red(I64_DPARAMS_R) {
    int64_t *cdata1;
    int (*compare)(const void *a, const void *b) = (int(*)(const void *, const void*)) iarray_i64median_compare;
    MEDIAN_R
}

static iarray_reduce_function_t I64MEDIAN = {
        .init = NULL,
        .reduction = CAST_R i64median_red,
        .finish = NULL,
};

static int iarray_i32median_compare(const int32_t *a, const int32_t *b) {
    COMPARE_RETURN
}
static void i32median_red(I32_DPARAMS_R) {
    int32_t *cdata1;
    int (*compare)(const void *a, const void *b) = (int(*)(const void *, const void*)) iarray_i32median_compare;
    MEDIAN_R
}

static iarray_reduce_function_t I32MEDIAN = {
        .init = NULL,
        .reduction = CAST_R i32median_red,
        .finish = NULL,
};

static int iarray_i16median_compare(const int16_t *a, const int16_t *b) {
    COMPARE_RETURN
}
static void i16median_red(I16_DPARAMS_R) {
    int16_t *cdata1;
    int (*compare)(const void *a, const void *b) = (int(*)(const void *, const void*)) iarray_i16median_compare;
    MEDIAN_R
}

static iarray_reduce_function_t I16MEDIAN = {
        .init = NULL,
        .reduction = CAST_R i16median_red,
        .finish = NULL,
};

static int iarray_i8median_compare(const int8_t *a, const int8_t *b) {
    COMPARE_RETURN
}

static void i8median_red(I8_DPARAMS_R) {
    int8_t *cdata1;
    int (*compare)(const void *a, const void *b) = (int(*)(const void *, const void*)) iarray_i8median_compare;
    MEDIAN_R
}

static iarray_reduce_function_t I8MEDIAN = {
        .init = NULL,
        .reduction = CAST_R i8median_red,
        .finish = NULL,
};


static int iarray_ui64median_compare(const uint64_t *a, const uint64_t *b) {
    COMPARE_RETURN
}

static void ui64median_red(UI64_DPARAMS_R) {
    uint64_t *cdata1;
    int (*compare)(const void *a, const void *b) = (int(*)(const void *, const void*)) iarray_ui64median_compare;
    MEDIAN_R
}

static iarray_reduce_function_t UI64MEDIAN = {
        .init = NULL,
        .reduction = CAST_R ui64median_red,
        .finish = NULL,
};


static int iarray_ui32median_compare(const uint32_t *a, const uint32_t *b) {
    COMPARE_RETURN
}
static void ui32median_red(UI32_DPARAMS_R) {
    uint32_t *cdata1;
    int (*compare)(const void *a, const void *b) = (int(*)(const void *, const void*)) iarray_ui32median_compare;
    MEDIAN_R
}

static iarray_reduce_function_t UI32MEDIAN = {
        .init = NULL,
        .reduction = CAST_R ui32median_red,
        .finish = NULL,
};

static int iarray_ui16median_compare(const uint16_t *a, const uint16_t *b) {
    COMPARE_RETURN
}
static void ui16median_red(UI16_DPARAMS_R) {
    uint16_t *cdata1;
    int (*compare)(const void *a, const void *b) = (int(*)(const void *, const void*)) iarray_ui16median_compare;
    MEDIAN_R
}

static iarray_reduce_function_t UI16MEDIAN = {
        .init = NULL,
        .reduction = CAST_R ui16median_red,
        .finish = NULL,
};

static int iarray_ui8median_compare(const uint8_t *a, const uint8_t *b) {
    COMPARE_RETURN
}
static void ui8median_red(UI8_DPARAMS_R) {
    uint8_t *cdata1;
    int (*compare)(const void *a, const void *b) = (int(*)(const void *, const void*)) iarray_ui8median_compare;
    MEDIAN_R
}

static iarray_reduce_function_t UI8MEDIAN = {
        .init = NULL,
        .reduction = CAST_R ui8median_red,
        .finish = NULL,
};

static int iarray_boolmedian_compare(const bool *a, const bool *b) {
    COMPARE_RETURN
}
static void boolmedian_red(BOOL_DPARAMS_R) {
    bool *cdata1;
    int (*compare)(const void *a, const void *b) = (int(*)(const void *, const void*)) iarray_boolmedian_compare;
    MEDIAN_R
}

static iarray_reduce_function_t BOOLMEDIAN = {
        .init = NULL,
        .reduction = CAST_R boolmedian_red,
        .finish = NULL,
};

#endif //IARRAY_IARRAY_REDUCE_MEDIAN_H

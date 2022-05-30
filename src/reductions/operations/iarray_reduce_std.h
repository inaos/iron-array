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

#ifndef IARRAY_IARRAY_REDUCE_STD_H
#define IARRAY_IARRAY_REDUCE_STD_H

#include "iarray_reduce_private.h"

#define STD_R \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0);  \
    *data0 = 0;          \
    double mean = 0; \
    for (int i = 0; i < nelem; ++i) { \
        mean += *data1;    \
        data1 += strides1;          \
    }         \
    mean /= nelem;          \
    data1 = data1_p;       \
    for (int i = 0; i < nelem; ++i) { \
        *data0 += pow(fabs(*data1 - mean), 2); \
        data1 += strides1; \
    }         \
    *data0 /= nelem;       \
    *data0 = sqrt(*data0);

#define NAN_STD_R \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0);  \
    *data0 = 0;          \
    double mean = 0;       \
    int64_t nnans = 0;              \
    for (int i = 0; i < nelem; ++i) { \
        if (isnan(*data1)) {          \
            nnans++;  \
        }         \
        else {    \
            mean += *data1;    \
        }\
        data1 += strides1;          \
    }             \
    mean /= (nelem - nnans);          \
    data1 = data1_p;\
    for (int i = 0; i < nelem; ++i) { \
        if (!isnan(*data1)) {        \
            *data0 += pow(fabs(*data1 - mean), 2); \
        }         \
        data1 += strides1;      \
    }             \
    *data0 /= (nelem - nnans);       \
    *data0 = sqrt(*data0);      \


static void dstd_red(DPARAMS_R) {
    const double *data1_p = data1; \
    STD_R
}

static iarray_reduce_function_t DSTD = {
        .init = NULL,
        .reduction = CAST_R dstd_red,
        .finish = NULL,
};

static void nan_dstd_red(DPARAMS_R) {
    const double *data1_p = data1; \
    NAN_STD_R
}

static iarray_reduce_function_t NAN_DSTD = {
    .init = NULL,
    .reduction = CAST_R nan_dstd_red,
    .finish = NULL,
};

// Only used for float output
#define FSTD_R \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0);  \
    *data0 = 0;          \
    float mean = 0; \
    for (int i = 0; i < nelem; ++i) { \
        mean += *data1;    \
        data1 += strides1;          \
    }         \
    mean /= nelem;          \
    data1 = data1_p;       \
    for (int i = 0; i < nelem; ++i) { \
        *data0 += powf(fabsf(*data1 - mean), 2); \
        data1 += strides1; \
    }         \
    *data0 /= nelem;       \
    *data0 = sqrtf(*data0);

#define NAN_FSTD_R \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0);  \
    *data0 = 0;          \
    float mean = 0;\
    int64_t nnans = 0;              \
    for (int i = 0; i < nelem; ++i) { \
        if (isnan(*data1)) {          \
            nnans++;  \
        } else {   \
            mean += *data1;    \
        }     \
        data1 += strides1;          \
    }         \
    mean /= (nelem - nnans);          \
    data1 = data1_p;       \
    for (int i = 0; i < nelem; ++i) { \
        if (!isnan(*data1)) {        \
            *data0 += pow(fabs(*data1 - mean), 2); \
        }\
        data1 += strides1; \
    }         \
    *data0 /= (nelem - nnans);       \
    *data0 = sqrtf(*data0);\


static void fstd_red(FPARAMS_R) {
    const float *data1_p = data1; \
    FSTD_R
}

static iarray_reduce_function_t FSTD = {
        .init = NULL,
        .reduction = CAST_R fstd_red,
        .finish = NULL,
};

static void nan_fstd_red(FPARAMS_R) {
    const float *data1_p = data1; \
    NAN_FSTD_R
}

static iarray_reduce_function_t NAN_FSTD = {
    .init = NULL,
    .reduction = CAST_R nan_fstd_red,
    .finish = NULL,
};

static void i64std_red(I64_DPARAMS_R) {
    const int64_t *data1_p = data1;
    STD_R
}

static iarray_reduce_function_t I64STD = {
        .init = NULL,
        .reduction = CAST_R i64std_red,
        .finish = NULL,
};

static void i32std_red(I32_DPARAMS_R) {
    const int32_t *data1_p = data1;
    STD_R
}

static iarray_reduce_function_t I32STD = {
        .init = NULL,
        .reduction = CAST_R i32std_red,
        .finish = NULL,
};

static void i16std_red(I16_DPARAMS_R) {
    const int16_t *data1_p = data1;
    STD_R
}

static iarray_reduce_function_t I16STD = {
        .init = NULL,
        .reduction = CAST_R i16std_red,
        .finish = NULL,
};

static void i8std_red(I8_DPARAMS_R) {
    const int8_t *data1_p = data1;
    STD_R
}

static iarray_reduce_function_t I8STD = {
        .init = NULL,
        .reduction = CAST_R i8std_red,
        .finish = NULL,
};

static void ui64std_red(UI64_DPARAMS_R) {
    const uint64_t *data1_p = data1;
    STD_R
}

static iarray_reduce_function_t UI64STD = {
        .init = NULL,
        .reduction = CAST_R ui64std_red,
        .finish = NULL,
};

static void ui32std_red(UI32_DPARAMS_R) {
    const uint32_t *data1_p = data1;
    STD_R
}

static iarray_reduce_function_t UI32STD = {
        .init = NULL,
        .reduction = CAST_R ui32std_red,
        .finish = NULL,
};

static void ui16std_red(UI16_DPARAMS_R) {
    const uint16_t *data1_p = data1;
    STD_R
}

static iarray_reduce_function_t UI16STD = {
        .init = NULL,
        .reduction = CAST_R ui16std_red,
        .finish = NULL,
};

static void ui8std_red(UI8_DPARAMS_R) {
    const uint8_t *data1_p = data1;
    STD_R
}

static iarray_reduce_function_t UI8STD = {
        .init = NULL,
        .reduction = CAST_R ui8std_red,
        .finish = NULL,
};

static void boolstd_red(BOOL_DPARAMS_R) {
    const bool *data1_p = data1;
    VAR_R
}

static iarray_reduce_function_t BOOLSTD = {
        .init = NULL,
        .reduction = CAST_R boolstd_red,
        .finish = NULL,
};

#endif //IARRAY_IARRAY_REDUCE_STD_H

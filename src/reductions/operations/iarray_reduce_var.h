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

#ifndef IARRAY_IARRAY_REDUCE_VAR_H
#define IARRAY_IARRAY_REDUCE_VAR_H

#include "iarray_reduce_private.h"

#define VAR_R \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0);  \
    user_data_t *u_data = (user_data_t *) user_data; \
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
    *data0 *= u_data->inv_nelem; \

#define NAN_DVAR_R \
    INA_UNUSED(strides0);  \
    user_data_t *u_data = (user_data_t *) user_data; \
    *data0 = 0;          \
    double mean = 0;       \
    int64_t nnans = 0;              \
    for (int i = 0; i < nelem; ++i) { \
        if (!isnan(*data1)) {                         \
            mean += *data1;\
        }          \
        else {     \
            nnans++;       \
         }\
        data1 += strides1;          \
    }         \
    mean /= (nelem - nnans);          \
    data1 = data1_p;       \
    for (int i = 0; i < nelem; ++i) {                \
        if (!isnan(*data1)) {                         \
            *data0 += pow(fabs(*data1 - mean), 2); \
        }          \
        data1 += strides1;                       \
    }             \
    *data0 /= (nelem - nnans); \


static void dvar_red(DPARAMS_R) {
    const double *data1_p = data1; \
    VAR_R
}

static iarray_reduce_function_t DVAR = {
        .init = NULL,
        .reduction = CAST_R dvar_red,
        .finish = NULL,
};

static void nan_dvar_red(DPARAMS_R) {
    const double *data1_p = data1; \
    NAN_DVAR_R
}

static iarray_reduce_function_t NAN_DVAR = {
    .init = NULL,
    .reduction = CAST_R nan_dvar_red,
    .finish = NULL,
};

// Only used for float output
#define FVAR_R \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0);  \
    user_data_t *u_data = (user_data_t *) user_data; \
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
    *data0 *= (float) u_data->inv_nelem ;

#define NAN_FVAR_R \
    INA_UNUSED(strides0);  \
    user_data_t *u_data = (user_data_t *) user_data; \
    *data0 = 0;          \
    float mean = 0;\
    int64_t nnans = 0;               \
    for (int i = 0; i < nelem; ++i) { \
        if (!isnan(*data1)) {                         \
            mean += *data1;\
        }          \
        else{      \
            nnans++;           \
        }           \
        data1 += strides1;          \
    }         \
    mean /= (nelem - nnans);          \
    data1 = data1_p;       \
    for (int i = 0; i < nelem; ++i) {                \
        if (!isnan(*data1)) {                         \
            *data0 += pow(fabs(*data1 - mean), 2); \
        }          \
        data1 += strides1;                       \
    }              \
    *data0 /= (nelem -nnans);                        \

static void fvar_red(FPARAMS_R) {
    const float *data1_p = data1; \
    FVAR_R
}

static iarray_reduce_function_t FVAR = {
        .init = NULL,
        .reduction = CAST_R fvar_red,
        .finish = NULL,
};

static void nan_fvar_red(FPARAMS_R) {
    const float *data1_p = data1; \
    NAN_FVAR_R
}

static iarray_reduce_function_t NAN_FVAR = {
    .init = NULL,
    .reduction = CAST_R nan_fvar_red,
    .finish = NULL,
};

static void i64var_red(I64_DPARAMS_R) {
    const int64_t *data1_p = data1;
    VAR_R
}

static iarray_reduce_function_t I64VAR = {
        .init = NULL,
        .reduction = CAST_R i64var_red,
        .finish = NULL,
};

static void i32var_red(I32_DPARAMS_R) {
    const int32_t *data1_p = data1;
    VAR_R
}

static iarray_reduce_function_t I32VAR = {
        .init = NULL,
        .reduction = CAST_R i32var_red,
        .finish = NULL,
};

static void i16var_red(I16_DPARAMS_R) {
    const int16_t *data1_p = data1;
    VAR_R
}

static iarray_reduce_function_t I16VAR = {
        .init = NULL,
        .reduction = CAST_R i16var_red,
        .finish = NULL,
};

static void i8var_red(I8_DPARAMS_R) {
    const int8_t *data1_p = data1;
    VAR_R
}

static iarray_reduce_function_t I8VAR = {
        .init = NULL,
        .reduction = CAST_R i8var_red,
        .finish = NULL,
};

static void ui64var_red(UI64_DPARAMS_R) {
    const uint64_t *data1_p = data1;
    VAR_R
}

static iarray_reduce_function_t UI64VAR = {
        .init = NULL,
        .reduction = CAST_R ui64var_red,
        .finish = NULL,
};

static void ui32var_red(UI32_DPARAMS_R) {
    const uint32_t *data1_p = data1;
    VAR_R
}

static iarray_reduce_function_t UI32VAR = {
        .init = NULL,
        .reduction = CAST_R ui32var_red,
        .finish = NULL,
};

static void ui16var_red(UI16_DPARAMS_R) {
    const uint16_t *data1_p = data1;
    VAR_R
}

static iarray_reduce_function_t UI16VAR = {
        .init = NULL,
        .reduction = CAST_R ui16var_red,
        .finish = NULL,
};

static void ui8var_red(UI8_DPARAMS_R) {
    const uint8_t *data1_p = data1;
    VAR_R
}

static iarray_reduce_function_t UI8VAR = {
        .init = NULL,
        .reduction = CAST_R ui8var_red,
        .finish = NULL,
};

static void boolvar_red(BOOL_DPARAMS_R) {
    const bool *data1_p = data1;
    VAR_R
}

static iarray_reduce_function_t BOOLVAR = {
        .init = NULL,
        .reduction = CAST_R boolvar_red,
        .finish = NULL,
};
#endif //IARRAY_IARRAY_REDUCE_VAR_H

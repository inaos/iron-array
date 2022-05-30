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

#ifndef IARRAY_IARRAY_REDUCE_SUM_H
#define IARRAY_IARRAY_REDUCE_SUM_H

#include "iarray_reduce_private.h"

#define SUM_I \
        INA_UNUSED(user_data); \
        *res = 0;

#define NAN_SUM_I \
        INA_UNUSED(user_data); \
        *res = NAN;

#define SUM_R \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    for (int i = 0; i < nelem; ++i) { \
        *data0 = *data0 + *data1; \
        data1 += strides1; \
    }

#define NAN_SUM_R \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    for (int i = 0; i < nelem; ++i) { \
        if (!isnan(*data1)) {         \
            if (isnan(*data0)) {      \
                  *data0 = *data1;\
            }      \
            else {\
                *data0 = *data0 + *data1; \
            }         \
        }\
        data1 += strides1; \
    }

#define SUM_F \
    INA_UNUSED(user_data); \
    INA_UNUSED(res); \
    ;

static void dsum_ini(DPARAMS_I) { SUM_I }
static void dsum_red(DPARAMS_R) { SUM_R }
static void dsum_fin(DPARAMS_F) { SUM_F }

static iarray_reduce_function_t DSUM = {
        .init = CAST_I dsum_ini,
        .reduction = CAST_R dsum_red,
        .finish = CAST_F dsum_fin
};

static void nan_dsum_ini(DPARAMS_I) { NAN_SUM_I }
static void nan_dsum_red(DPARAMS_R) { NAN_SUM_R }

static iarray_reduce_function_t NAN_DSUM = {
    .init = CAST_I nan_dsum_ini,
    .reduction = CAST_R nan_dsum_red,
    .finish = CAST_F dsum_fin
};

static void fsum_ini(FPARAMS_I) { SUM_I }
static void fsum_red(FPARAMS_R) { SUM_R }
static void fsum_fin(FPARAMS_F) { SUM_F }

static iarray_reduce_function_t FSUM = {
        .init = CAST_I fsum_ini,
        .reduction = CAST_R fsum_red,
        .finish = CAST_F fsum_fin
};

static void nan_fsum_ini(FPARAMS_I) { NAN_SUM_I }
static void nan_fsum_red(FPARAMS_R) { NAN_SUM_R }

static iarray_reduce_function_t NAN_FSUM = {
    .init = CAST_I nan_fsum_ini,
    .reduction = CAST_R nan_fsum_red,
    .finish = CAST_F fsum_fin
};

static void i64sum_ini(I64PARAMS_I) { SUM_I }
static void i64sum_red(I64PARAMS_R) { SUM_R }
static void i64sum_fin(I64PARAMS_F) { SUM_F }

static iarray_reduce_function_t I64SUM = {
        .init = CAST_I i64sum_ini,
        .reduction = CAST_R i64sum_red,
        .finish = CAST_F i64sum_fin
};

static void i32sum_ini(I64PARAMS_I) { SUM_I }
static void i32sum_red(I32_64PARAMS_R) { SUM_R }
static void i32sum_fin(I64PARAMS_F) { SUM_F }

static iarray_reduce_function_t I32SUM = {
        .init = CAST_I i32sum_ini,
        .reduction = CAST_R i32sum_red,
        .finish = CAST_F i32sum_fin
};

static void i16sum_ini(I64PARAMS_I) { SUM_I }
static void i16sum_red(I16_64PARAMS_R) { SUM_R }
static void i16sum_fin(I64PARAMS_F) { SUM_F }

static iarray_reduce_function_t I16SUM = {
        .init = CAST_I i16sum_ini,
        .reduction = CAST_R i16sum_red,
        .finish = CAST_F i16sum_fin
};

static void i8sum_ini(I64PARAMS_I) { SUM_I }
static void i8sum_red(I8_64PARAMS_R) { SUM_R }
static void i8sum_fin(I64PARAMS_F) { SUM_F }

static iarray_reduce_function_t I8SUM = {
        .init = CAST_I i8sum_ini,
        .reduction = CAST_R i8sum_red,
        .finish = CAST_F i8sum_fin
};

static void ui64sum_ini(UI64PARAMS_I) { SUM_I }
static void ui64sum_red(UI64PARAMS_R) { SUM_R }
static void ui64sum_fin(UI64PARAMS_F) { SUM_F }

static iarray_reduce_function_t UI64SUM = {
        .init = CAST_I ui64sum_ini,
        .reduction = CAST_R ui64sum_red,
        .finish = CAST_F ui64sum_fin
};

static void ui32sum_ini(UI64PARAMS_I) { SUM_I }
static void ui32sum_red(UI32_64PARAMS_R) { SUM_R }
static void ui32sum_fin(UI64PARAMS_F) { SUM_F }

static iarray_reduce_function_t UI32SUM = {
        .init = CAST_I ui32sum_ini,
        .reduction = CAST_R ui32sum_red,
        .finish = CAST_F ui32sum_fin
};

static void ui16sum_ini(UI64PARAMS_I) { SUM_I }
static void ui16sum_red(UI16_64PARAMS_R) { SUM_R }
static void ui16sum_fin(UI64PARAMS_F) { SUM_F }

static iarray_reduce_function_t UI16SUM = {
        .init = CAST_I ui16sum_ini,
        .reduction = CAST_R ui16sum_red,
        .finish = CAST_F ui16sum_fin
};

static void ui8sum_ini(UI64PARAMS_I) { SUM_I }
static void ui8sum_red(UI8_64PARAMS_R) { SUM_R }
static void ui8sum_fin(UI64PARAMS_F) { SUM_F }

static iarray_reduce_function_t UI8SUM = {
        .init = CAST_I ui8sum_ini,
        .reduction = CAST_R ui8sum_red,
        .finish = CAST_F ui8sum_fin
};

static void boolsum_ini(I64PARAMS_I) { SUM_I }
static void boolsum_red(BOOL_64PARAMS_R) { SUM_R }
static void boolsum_fin(I64PARAMS_F) { SUM_F }

static iarray_reduce_function_t BOOLSUM = {
        .init = CAST_I boolsum_ini,
        .reduction = CAST_R boolsum_red,
        .finish = CAST_F boolsum_fin
};


#endif //IARRAY_IARRAY_REDUCE_SUM_H

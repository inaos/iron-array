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

#ifndef IARRAY_IARRAY_REDUCE_MAX_H
#define IARRAY_IARRAY_REDUCE_MAX_H

#include "iarray_reduce_private.h"


#define MAX_I \
    INA_UNUSED(user_data); \
    *res = -INFINITY;
#define MAX_I_I64 \
    INA_UNUSED(user_data); \
    *res = LLONG_MIN;
#define MAX_I_I32 \
    INA_UNUSED(user_data); \
    *res = INT_MIN;
#define MAX_I_I16 \
    INA_UNUSED(user_data); \
    *res = SHRT_MIN;
#define MAX_I_I8 \
    INA_UNUSED(user_data); \
    *res = SCHAR_MIN;
#define MAX_I_UI64 \
    INA_UNUSED(user_data); \
    *res = 0ULL;
#define MAX_I_UI32 \
    INA_UNUSED(user_data); \
    *res = 0U;
#define MAX_I_UI16 \
    INA_UNUSED(user_data); \
    *res = 0;
#define MAX_I_UI8 \
    INA_UNUSED(user_data); \
    *res = 0;
#define MAX_I_BOOL \
    INA_UNUSED(user_data); \
    *res = false;

#define MAX_R \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    for (int i = 0; i < nelem; ++i) { \
        if (*data1 > *data0) { \
            *data0 = *data1; \
        } \
        data1 += strides1; \
    }

#define MAX_F \
    INA_UNUSED(user_data); \
    INA_UNUSED(res); \
    ;

static void dmax_ini(DPARAMS_I) { MAX_I }
static void dmax_red(DPARAMS_R) { MAX_R }
static void dmax_fin(DPARAMS_F) { MAX_F }

static iarray_reduce_function_t DMAX = {
        .init = CAST_I dmax_ini,
        .reduction = CAST_R dmax_red,
        .finish = CAST_F dmax_fin
};

static void fmax_ini(FPARAMS_I) { MAX_I }
static void fmax_red(FPARAMS_R) { MAX_R }
static void fmax_fin(FPARAMS_F) { MAX_F }

static iarray_reduce_function_t FMAX = {
        .init = CAST_I fmax_ini,
        .reduction = CAST_R fmax_red,
        .finish = CAST_F fmax_fin
};

static void i64max_ini(I64PARAMS_I) { MAX_I_I64 }
static void i64max_red(I64PARAMS_R) { MAX_R }
static void i64max_fin(I64PARAMS_F) { MAX_F }

static iarray_reduce_function_t I64MAX = {
        .init = CAST_I i64max_ini,
        .reduction = CAST_R i64max_red,
        .finish = CAST_F i64max_fin
};

static void i32max_ini(I32PARAMS_I) { MAX_I_I32 }
static void i32max_red(I32PARAMS_R) { MAX_R }
static void i32max_fin(I32PARAMS_F) { MAX_F }

static iarray_reduce_function_t I32MAX = {
        .init = CAST_I i32max_ini,
        .reduction = CAST_R i32max_red,
        .finish = CAST_F i32max_fin
};

static void i16max_ini(I16PARAMS_I) { MAX_I_I16 }
static void i16max_red(I16PARAMS_R) { MAX_R }
static void i16max_fin(I16PARAMS_F) { MAX_F }

static iarray_reduce_function_t I16MAX = {
        .init = CAST_I i16max_ini,
        .reduction = CAST_R i16max_red,
        .finish = CAST_F i16max_fin
};

static void i8max_ini(I8PARAMS_I) { MAX_I_I8 }
static void i8max_red(I8PARAMS_R) { MAX_R }
static void i8max_fin(I8PARAMS_F) { MAX_F }

static iarray_reduce_function_t I8MAX = {
        .init = CAST_I i8max_ini,
        .reduction = CAST_R i8max_red,
        .finish = CAST_F i8max_fin
};

static void ui64max_ini(UI64PARAMS_I) { MAX_I_UI64 }
static void ui64max_red(UI64PARAMS_R) { MAX_R }
static void ui64max_fin(UI64PARAMS_F) { MAX_F }

static iarray_reduce_function_t UI64MAX = {
        .init = CAST_I ui64max_ini,
        .reduction = CAST_R ui64max_red,
        .finish = CAST_F ui64max_fin
};

static void ui32max_ini(UI32PARAMS_I) { MAX_I_UI32 }
static void ui32max_red(UI32PARAMS_R) { MAX_R }
static void ui32max_fin(UI32PARAMS_F) { MAX_F }

static iarray_reduce_function_t UI32MAX = {
        .init = CAST_I ui32max_ini,
        .reduction = CAST_R ui32max_red,
        .finish = CAST_F ui32max_fin
};

static void ui16max_ini(UI16PARAMS_I) { MAX_I_UI16 }
static void ui16max_red(UI16PARAMS_R) { MAX_R }
static void ui16max_fin(UI16PARAMS_F) { MAX_F }

static iarray_reduce_function_t UI16MAX = {
        .init = CAST_I ui16max_ini,
        .reduction = CAST_R ui16max_red,
        .finish = CAST_F ui16max_fin
};

static void ui8max_ini(UI8PARAMS_I) { MAX_I_UI8 }
static void ui8max_red(UI8PARAMS_R) { MAX_R }
static void ui8max_fin(UI8PARAMS_F) { MAX_F }

static iarray_reduce_function_t UI8MAX = {
        .init = CAST_I ui8max_ini,
        .reduction = CAST_R ui8max_red,
        .finish = CAST_F ui8max_fin
};

static void boolmax_ini(BOOLPARAMS_I) { MAX_I_BOOL }
static void boolmax_red(BOOLPARAMS_R) { MAX_R }
static void boolmax_fin(BOOLPARAMS_F) { MAX_F }

static iarray_reduce_function_t BOOLMAX = {
        .init = CAST_I boolmax_ini,
        .reduction = CAST_R boolmax_red,
        .finish = CAST_F boolmax_fin
};

#endif //IARRAY_IARRAY_REDUCE_MAX_H

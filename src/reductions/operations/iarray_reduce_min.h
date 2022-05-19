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

#ifndef IARRAY_IARRAY_REDUCE_MIN_H
#define IARRAY_IARRAY_REDUCE_MIN_H

#include "iarray_reduce_private.h"


#define MIN_I \
    INA_UNUSED(user_data); \
    *res = INFINITY;
#define MIN_I_I64 \
    INA_UNUSED(user_data); \
    *res = LLONG_MAX;
#define MIN_I_I32 \
    INA_UNUSED(user_data); \
    *res = INT_MAX;
#define MIN_I_I16 \
    INA_UNUSED(user_data); \
    *res = SHRT_MAX;
#define MIN_I_I8 \
    INA_UNUSED(user_data); \
    *res = SCHAR_MAX;
#define MIN_I_UI64 \
    INA_UNUSED(user_data); \
    *res = ULLONG_MAX;
#define MIN_I_UI32 \
    INA_UNUSED(user_data); \
    *res = UINT_MAX;
#define MIN_I_UI16 \
    INA_UNUSED(user_data); \
    *res = USHRT_MAX;
#define MIN_I_UI8 \
    INA_UNUSED(user_data); \
    *res = UCHAR_MAX;
#define MIN_I_BOOL \
    INA_UNUSED(user_data); \
    *res = true;

#define MIN_R \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    for (int i = 0; i < nelem; ++i) { \
        if (*data1 < *data0) {    \
            *data0 = *data1; \
        } \
        data1 += strides1; \
    }

#define MIN_F \
    INA_UNUSED(user_data); \
    INA_UNUSED(res); \
    ;

static void dmin_ini(DPARAMS_I) { MIN_I }
static void dmin_red(DPARAMS_R) { MIN_R }
static void dmin_fin(DPARAMS_F) { MIN_F }

static iarray_reduce_function_t DMIN = {
        .init = CAST_I dmin_ini,
        .reduction = CAST_R dmin_red,
        .finish = CAST_F dmin_fin
};

static void fmin_ini(FPARAMS_I) { MIN_I }
static void fmin_red(FPARAMS_R) { MIN_R }
static void fmin_fin(FPARAMS_F) { MIN_F }

static iarray_reduce_function_t FMIN = {
        .init = CAST_I fmin_ini,
        .reduction =  CAST_R fmin_red,
        .finish = CAST_F fmin_fin
};

static void i64min_ini(I64PARAMS_I) { MIN_I_I64 }
static void i64min_red(I64PARAMS_R) { MIN_R }
static void i64min_fin(I64PARAMS_F) { MIN_F }

static iarray_reduce_function_t I64MIN = {
        .init = CAST_I i64min_ini,
        .reduction =  CAST_R i64min_red,
        .finish = CAST_F i64min_fin
};

static void i32min_ini(I32PARAMS_I) { MIN_I_I32 }
static void i32min_red(I32PARAMS_R) { MIN_R }
static void i32min_fin(I32PARAMS_F) { MIN_F }

static iarray_reduce_function_t I32MIN = {
        .init = CAST_I i32min_ini,
        .reduction =  CAST_R i32min_red,
        .finish = CAST_F i32min_fin
};

static void i16min_ini(I16PARAMS_I) { MIN_I_I16 }
static void i16min_red(I16PARAMS_R) { MIN_R }
static void i16min_fin(I16PARAMS_F) { MIN_F }

static iarray_reduce_function_t I16MIN = {
        .init = CAST_I i16min_ini,
        .reduction =  CAST_R i16min_red,
        .finish = CAST_F i16min_fin
};

static void i8min_ini(I8PARAMS_I) { MIN_I_I8 }
static void i8min_red(I8PARAMS_R) { MIN_R }
static void i8min_fin(I8PARAMS_F) { MIN_F }

static iarray_reduce_function_t I8MIN = {
        .init = CAST_I i8min_ini,
        .reduction =  CAST_R i8min_red,
        .finish = CAST_F i8min_fin
};

static void ui64min_ini(UI64PARAMS_I) { MIN_I_UI64 }
static void ui64min_red(UI64PARAMS_R) { MIN_R }
static void ui64min_fin(UI64PARAMS_F) { MIN_F }

static iarray_reduce_function_t UI64MIN = {
        .init = CAST_I ui64min_ini,
        .reduction =  CAST_R ui64min_red,
        .finish = CAST_F ui64min_fin
};

static void ui32min_ini(UI32PARAMS_I) { MIN_I_UI32 }
static void ui32min_red(UI32PARAMS_R) { MIN_R }
static void ui32min_fin(UI32PARAMS_F) { MIN_F }

static iarray_reduce_function_t UI32MIN = {
        .init = CAST_I ui32min_ini,
        .reduction =  CAST_R ui32min_red,
        .finish = CAST_F ui32min_fin
};

static void ui16min_ini(UI16PARAMS_I) { MIN_I_UI16 }
static void ui16min_red(UI16PARAMS_R) { MIN_R }
static void ui16min_fin(UI16PARAMS_F) { MIN_F }

static iarray_reduce_function_t UI16MIN = {
        .init = CAST_I ui16min_ini,
        .reduction =  CAST_R ui16min_red,
        .finish = CAST_F ui16min_fin
};

static void ui8min_ini(UI8PARAMS_I) { MIN_I_UI8 }
static void ui8min_red(UI8PARAMS_R) { MIN_R }
static void ui8min_fin(UI8PARAMS_F) { MIN_F }

static iarray_reduce_function_t UI8MIN = {
        .init = CAST_I ui8min_ini,
        .reduction =  CAST_R ui8min_red,
        .finish = CAST_F ui8min_fin
};

static void boolmin_ini(BOOLPARAMS_I) { MIN_I_BOOL }
static void boolmin_red(BOOLPARAMS_R) { MIN_R }
static void boolmin_fin(BOOLPARAMS_F) { MIN_F }

static iarray_reduce_function_t BOOLMIN = {
        .init = CAST_I boolmin_ini,
        .reduction =  CAST_R boolmin_red,
        .finish = CAST_F boolmin_fin
};

#endif //IARRAY_IARRAY_REDUCE_MIN_H

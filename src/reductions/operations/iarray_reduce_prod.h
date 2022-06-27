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

#ifndef IARRAY_IARRAY_REDUCE_PROD_H
#define IARRAY_IARRAY_REDUCE_PROD_H

#include "iarray_reduce_private.h"

#define PROD_I \
    INA_UNUSED(user_data); \
    *res = 1;

#define PROD_R \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    for (int i = 0; i < nelem; ++i) { \
        *data0 = *data0 * *data1; \
        data1 += strides1; \
    }

#define NAN_PROD_R \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    for (int i = 0; i < nelem; ++i) { \
        if (!isnan(*data1)) {          \
            *data0 = *data0 * *data1; \
        }\
        data1 += strides1; \
    }

#define PROD_F \
    INA_UNUSED(user_data); \
    INA_UNUSED(res); \
    ;

static void dprod_ini(DPARAMS_I) { PROD_I }
static void dprod_red(DPARAMS_R) { PROD_R }
static void dprod_fin(DPARAMS_F) { PROD_F }

static iarray_reduce_function_t DPROD = {
        .init = CAST_I dprod_ini,
        .reduction = CAST_R dprod_red,
        .finish = CAST_F dprod_fin
};

static void nan_dprod_red(DPARAMS_R) { NAN_PROD_R }

static iarray_reduce_function_t NAN_DPROD = {
    .init = CAST_I dprod_ini,
    .reduction = CAST_R nan_dprod_red,
    .finish = CAST_F dprod_fin
};

static void fprod_ini(FPARAMS_I) { PROD_I }
static void fprod_red(FPARAMS_R) { PROD_R }
static void fprod_fin(FPARAMS_F) { PROD_F }

static iarray_reduce_function_t FPROD = {
        .init = CAST_I fprod_ini,
        .reduction = CAST_R fprod_red,
        .finish = CAST_F fprod_fin
};

static void nan_fprod_red(FPARAMS_R) { NAN_PROD_R }

static iarray_reduce_function_t NAN_FPROD = {
    .init = CAST_I fprod_ini,
    .reduction = CAST_R nan_fprod_red,
    .finish = CAST_F fprod_fin
};

static void i64prod_ini(I64PARAMS_I) { PROD_I }
static void i64prod_red(I64PARAMS_R) { PROD_R }
static void i64prod_fin(I64PARAMS_F) { PROD_F }

static iarray_reduce_function_t I64PROD = {
        .init = CAST_I i64prod_ini,
        .reduction = CAST_R i64prod_red,
        .finish = CAST_F i64prod_fin
};

static void i32prod_ini(I64PARAMS_I) { PROD_I }
static void i32prod_red(I32_64PARAMS_R) { PROD_R }
static void i32prod_fin(I64PARAMS_F) { PROD_F }

static iarray_reduce_function_t I32PROD = {
        .init = CAST_I i32prod_ini,
        .reduction = CAST_R i32prod_red,
        .finish = CAST_F i32prod_fin
};

static void i16prod_ini(I64PARAMS_I) { PROD_I }
static void i16prod_red(I16_64PARAMS_R) { PROD_R }
static void i16prod_fin(I64PARAMS_F) { PROD_F }

static iarray_reduce_function_t I16PROD = {
        .init = CAST_I i16prod_ini,
        .reduction = CAST_R i16prod_red,
        .finish = CAST_F i16prod_fin
};

static void i8prod_ini(I64PARAMS_I) { PROD_I }
static void i8prod_red(I8_64PARAMS_R) { PROD_R }
static void i8prod_fin(I64PARAMS_F) { PROD_F }

static iarray_reduce_function_t I8PROD = {
        .init = CAST_I i8prod_ini,
        .reduction = CAST_R i8prod_red,
        .finish = CAST_F i8prod_fin
};

static void ui64prod_ini(UI64PARAMS_I) { PROD_I }
static void ui64prod_red(UI64PARAMS_R) { PROD_R }
static void ui64prod_fin(UI64PARAMS_F) { PROD_F }

static iarray_reduce_function_t UI64PROD = {
        .init = CAST_I ui64prod_ini,
        .reduction = CAST_R ui64prod_red,
        .finish = CAST_F ui64prod_fin
};

static void ui32prod_ini(UI64PARAMS_I) { PROD_I }
static void ui32prod_red(UI32_64PARAMS_R) { PROD_R }
static void ui32prod_fin(UI64PARAMS_F) { PROD_F }

static iarray_reduce_function_t UI32PROD = {
        .init = CAST_I ui32prod_ini,
        .reduction = CAST_R ui32prod_red,
        .finish = CAST_F ui32prod_fin
};

static void ui16prod_ini(UI64PARAMS_I) { PROD_I }
static void ui16prod_red(UI16_64PARAMS_R) { PROD_R }
static void ui16prod_fin(UI64PARAMS_F) { PROD_F }

static iarray_reduce_function_t UI16PROD = {
        .init = CAST_I ui16prod_ini,
        .reduction = CAST_R ui16prod_red,
        .finish = CAST_F ui16prod_fin
};

static void ui8prod_ini(UI64PARAMS_I) { PROD_I }
static void ui8prod_red(UI8_64PARAMS_R) { PROD_R }
static void ui8prod_fin(UI64PARAMS_F) { PROD_F }

static iarray_reduce_function_t UI8PROD = {
        .init = CAST_I ui8prod_ini,
        .reduction = CAST_R ui8prod_red,
        .finish = CAST_F ui8prod_fin
};

static void boolprod_ini(I64PARAMS_I) { PROD_I }
static void boolprod_red(BOOL_64PARAMS_R) { PROD_R }
static void boolprod_fin(I64PARAMS_F) { PROD_F }

static iarray_reduce_function_t BOOLPROD = {
        .init = CAST_I boolprod_ini,
        .reduction = CAST_R boolprod_red,
        .finish = CAST_F boolprod_fin
};


#endif //IARRAY_IARRAY_REDUCE_PROD_H

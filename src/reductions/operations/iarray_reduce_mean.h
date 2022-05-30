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

#ifndef IARRAY_IARRAY_REDUCE_MEAN_H
#define IARRAY_IARRAY_REDUCE_MEAN_H

#include "iarray_reduce_private.h"


#define MEAN_I \
    INA_UNUSED(user_data); \
    *res = 0;

#define MEAN_R \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    for (int i = 0; i < nelem; ++i) { \
        *data0 = *data0 + *data1; \
        data1 += strides1; \
    }

#define NAN_MEAN_R \
    INA_UNUSED(strides0); \
    user_data_t *u_data = (user_data_t *) user_data; \
    /* In a NAN_MEAN u_data->aux_nelem is the not NAN nelems */        \
    for (int i = 0; i < nelem; ++i) { \
        if (isnan(*data1)) {          \
            u_data->aux_nelem--;       \
        }\
        else {     \
            *data0 = *data0 + *data1; \
        }           \
        data1 += strides1; \
    }\

#define MEAN_F \
    user_data_t *u_data = (user_data_t *) user_data; \
    *res = *res * u_data->inv_nelem;

#define NAN_MEAN_F \
    user_data_t *u_data = (user_data_t *) user_data; \
    *res = *res / u_data->aux_nelem;

static void dmean_ini(DPARAMS_I) { MEAN_I }
static void dmean_red(DPARAMS_R) { MEAN_R }
static void dmean_fin(DPARAMS_F) { MEAN_F }

static iarray_reduce_function_t DMEAN = {
        .init = CAST_I dmean_ini,
        .reduction = CAST_R dmean_red,
        .finish = CAST_F dmean_fin
};

static void nan_dmean_red(DPARAMS_R) { NAN_MEAN_R }
static void nan_dmean_fin(DPARAMS_F) { NAN_MEAN_F }

static iarray_reduce_function_t NAN_DMEAN = {
    .init = CAST_I dmean_ini,
    .reduction = CAST_R nan_dmean_red,
    .finish = CAST_F nan_dmean_fin
};

static void fmean_ini(FPARAMS_I) { MEAN_I }
static void fmean_red(FPARAMS_R) { MEAN_R }
static void fmean_fin(FPARAMS_F) { MEAN_F }

static iarray_reduce_function_t FMEAN = {
        .init = CAST_I fmean_ini,
        .reduction = CAST_R fmean_red,
        .finish = CAST_F fmean_fin
};

static void nan_fmean_red(FPARAMS_R) { NAN_MEAN_R }
static void nan_fmean_fin(FPARAMS_F) { NAN_MEAN_F }

static iarray_reduce_function_t NAN_FMEAN = {
    .init = CAST_I fmean_ini,
    .reduction = CAST_R nan_fmean_red,
    .finish = CAST_F nan_fmean_fin
};

static void i64mean_ini(DPARAMS_I) { MEAN_I }
static void i64mean_red(I64_DPARAMS_R) { MEAN_R }
static void i64mean_fin(DPARAMS_F) { MEAN_F }

static iarray_reduce_function_t I64MEAN = {
        .init = CAST_I i64mean_ini,
        .reduction = CAST_R i64mean_red,
        .finish = CAST_F i64mean_fin
};
static void i32mean_ini(DPARAMS_I) { MEAN_I }
static void i32mean_red(I32_DPARAMS_R) { MEAN_R }
static void i32mean_fin(DPARAMS_F) { MEAN_F }

static iarray_reduce_function_t I32MEAN = {
        .init = CAST_I i32mean_ini,
        .reduction = CAST_R i32mean_red,
        .finish = CAST_F i32mean_fin
};

static void i16mean_ini(DPARAMS_I) { MEAN_I }
static void i16mean_red(I16_DPARAMS_R) { MEAN_R }
static void i16mean_fin(DPARAMS_F) { MEAN_F }

static iarray_reduce_function_t I16MEAN = {
        .init = CAST_I i16mean_ini,
        .reduction = CAST_R i16mean_red,
        .finish = CAST_F i16mean_fin
};

static void i8mean_ini(DPARAMS_I) { MEAN_I }
static void i8mean_red(I8_DPARAMS_R) { MEAN_R }
static void i8mean_fin(DPARAMS_F) { MEAN_F }

static iarray_reduce_function_t I8MEAN = {
        .init = CAST_I i8mean_ini,
        .reduction = CAST_R i8mean_red,
        .finish = CAST_F i8mean_fin
};

static void ui64mean_ini(DPARAMS_I) { MEAN_I }
static void ui64mean_red(UI64_DPARAMS_R) { MEAN_R }
static void ui64mean_fin(DPARAMS_F) { MEAN_F }

static iarray_reduce_function_t UI64MEAN = {
        .init = CAST_I ui64mean_ini,
        .reduction = CAST_R ui64mean_red,
        .finish = CAST_F ui64mean_fin
};
static void ui32mean_ini(DPARAMS_I) { MEAN_I }
static void ui32mean_red(UI32_DPARAMS_R) { MEAN_R }
static void ui32mean_fin(DPARAMS_F) { MEAN_F }

static iarray_reduce_function_t UI32MEAN = {
        .init = CAST_I ui32mean_ini,
        .reduction = CAST_R ui32mean_red,
        .finish = CAST_F ui32mean_fin
};

static void ui16mean_ini(DPARAMS_I) { MEAN_I }
static void ui16mean_red(UI16_DPARAMS_R) { MEAN_R }
static void ui16mean_fin(DPARAMS_F) { MEAN_F }

static iarray_reduce_function_t UI16MEAN = {
        .init = CAST_I ui16mean_ini,
        .reduction = CAST_R ui16mean_red,
        .finish = CAST_F ui16mean_fin
};

static void ui8mean_ini(DPARAMS_I) { MEAN_I }
static void ui8mean_red(UI8_DPARAMS_R) { MEAN_R }
static void ui8mean_fin(DPARAMS_F) { MEAN_F }

static iarray_reduce_function_t UI8MEAN = {
        .init = CAST_I ui8mean_ini,
        .reduction = CAST_R ui8mean_red,
        .finish = CAST_F ui8mean_fin
};

static void boolmean_ini(DPARAMS_I) { MEAN_I }
static void boolmean_red(BOOL_DPARAMS_R) { MEAN_R }
static void boolmean_fin(DPARAMS_F) { MEAN_F }

static iarray_reduce_function_t BOOLMEAN = {
        .init = CAST_I boolmean_ini,
        .reduction = CAST_R boolmean_red,
        .finish = CAST_F boolmean_fin
};

#endif //IARRAY_IARRAY_REDUCE_MEAN_H


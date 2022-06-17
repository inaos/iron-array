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

#define VAR_I \
    user_data_os_t *u_data = (user_data_os_t *) user_data; \
    *res = 0; \
    u_data->not_nan_nelems[u_data->i] = 0;

#define VAR_R \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0);  \
    INA_UNUSED(strides1);  \
    INA_UNUSED(nelem);  \
    user_data_os_t *u_data = (user_data_os_t *) user_data; \
    const double *mean = (double *) u_data->mean;    \
    double dif = (double) *data1 - *mean;          \
    *data0 += dif * dif;

#define VAR_F \
    INA_UNUSED(user_data); \
    user_data_os_t *u_data = (user_data_os_t *) user_data; \
    *res = *res * u_data->inv_nelem;


static void dvar_red(DPARAMS_R) { VAR_R }
static void dvar_init(DPARAMS_I) { VAR_I }
static void dvar_finish(DPARAMS_F) { VAR_F }

static iarray_reduce_function_t DVAR = {
        .init = CAST_I dvar_init,
        .reduction = CAST_R dvar_red,
        .finish = CAST_F dvar_finish,
};

#define NANVAR_I \
    VAR_I

#define NANVAR_R \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0);  \
    INA_UNUSED(strides1);  \
    INA_UNUSED(nelem);  \
    user_data_os_t *u_data = (user_data_os_t *) user_data; \
    const double *mean = (double *) u_data->mean;    \
    if (!isnan(*data1)) {  \
        double dif = (double) *data1 - *mean;          \
        *data0 += dif * dif;         \
        u_data->not_nan_nelems[u_data->i]++; \
    }

#define NANVAR_F \
    INA_UNUSED(user_data); \
    user_data_os_t *u_data = (user_data_os_t *) user_data; \
    *res = *res / u_data->not_nan_nelems[u_data->i];\

static void dnanvar_red(DPARAMS_R) { NANVAR_R }
static void dnanvar_init(DPARAMS_I) { NANVAR_I }
static void dnanvar_finish(DPARAMS_F) { NANVAR_F }

static iarray_reduce_function_t DNANVAR = {
        .init = CAST_I dnanvar_init,
        .reduction = CAST_R dnanvar_red,
        .finish = CAST_F dnanvar_finish,
};

// Only used for float output
#define FVAR_R \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0);  \
    INA_UNUSED(strides1);  \
    INA_UNUSED(nelem);  \
    user_data_os_t *u_data = (user_data_os_t *) user_data; \
    const float *mean = (float *) u_data->mean;      \
    float dif = (float) *data1 - *mean;          \
    *data0 += dif * dif;         \

#define FVAR_I \
    VAR_I

#define FVAR_F \
    VAR_F

static void fvar_red(FPARAMS_R) { FVAR_R }
static void fvar_ini(FPARAMS_I) { FVAR_I }
static void fvar_fin(FPARAMS_F) { FVAR_F }

static iarray_reduce_function_t FVAR = {
        .init = CAST_I fvar_ini,
        .reduction = CAST_R fvar_red,
        .finish = CAST_F fvar_fin,
};

#define FNANVAR_I \
    NANVAR_I

#define FNANVAR_R \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0);  \
    INA_UNUSED(strides1);  \
    INA_UNUSED(nelem);  \
    user_data_os_t *u_data = (user_data_os_t *) user_data; \
    const float *mean = (float *) u_data->mean;    \
    if (!isnan(*data1)) {  \
        float dif = (float) *data1 - *mean;          \
        *data0 += dif * dif;         \
        u_data->not_nan_nelems[u_data->i]++; \
    }

#define FNANVAR_F \
    NANVAR_F

static void fnanvar_red(FPARAMS_R) { FNANVAR_R }
static void fnanvar_init(FPARAMS_I) { FNANVAR_I }
static void fnanvar_finish(FPARAMS_F) { FNANVAR_F }

static iarray_reduce_function_t FNANVAR = {
        .init = CAST_I fnanvar_init,
        .reduction = CAST_R fnanvar_red,
        .finish = CAST_F fnanvar_finish,
};

static void i64var_red(I64_DPARAMS_R) { VAR_R }
static void i64var_ini(DPARAMS_I) { VAR_I }
static void i64var_fin(DPARAMS_F) { VAR_F }

static iarray_reduce_function_t I64VAR = {
        .init = CAST_I i64var_ini,
        .reduction = CAST_R i64var_red,
        .finish = CAST_F i64var_fin,
};

static void i32var_red(I32_DPARAMS_R) { VAR_R }
static void i32var_ini(DPARAMS_I) { VAR_I }
static void i32var_fin(DPARAMS_F) { VAR_F }

static iarray_reduce_function_t I32VAR = {
        .init = CAST_I i32var_ini,
        .reduction = CAST_R i32var_red,
        .finish = CAST_F i32var_fin,
};

static void i16var_red(I16_DPARAMS_R) { VAR_R }
static void i16var_ini(DPARAMS_I) { VAR_I }
static void i16var_fin(DPARAMS_F) { VAR_F }

static iarray_reduce_function_t I16VAR = {
        .init = CAST_I i16var_ini,
        .reduction = CAST_R i16var_red,
        .finish = CAST_F i16var_fin,
};

static void i8var_red(I8_DPARAMS_R) { VAR_R }
static void i8var_ini(DPARAMS_I) { VAR_I }
static void i8var_fin(DPARAMS_F) { VAR_F }

static iarray_reduce_function_t I8VAR = {
        .init = CAST_I i8var_ini,
        .reduction = CAST_R i8var_red,
        .finish = CAST_F i8var_fin,
};

static void ui64var_red(UI64_DPARAMS_R) { VAR_R }
static void ui64var_ini(DPARAMS_I) { VAR_I }
static void ui64var_fin(DPARAMS_F) { VAR_F }

static iarray_reduce_function_t UI64VAR = {
        .init = CAST_I ui64var_ini,
        .reduction = CAST_R ui64var_red,
        .finish = CAST_F ui64var_fin,
};

static void ui32var_red(UI32_DPARAMS_R) { VAR_R }
// static void u32var_ini(DPARAMS_I) { VAR_I }
// static void u32var_fin(DPARAMS_F) { VAR_F }

static iarray_reduce_function_t UI32VAR = {
        .init = CAST_I ui64var_ini,
        .reduction = CAST_R ui32var_red,
        .finish = CAST_F ui64var_fin,
};

static void ui16var_red(UI16_DPARAMS_R) { VAR_R }
static void ui16var_ini(DPARAMS_I) { VAR_I }
static void ui16var_fin(DPARAMS_F) { VAR_F }

static iarray_reduce_function_t UI16VAR = {
        .init = CAST_I ui16var_ini,
        .reduction = CAST_R ui16var_red,
        .finish = CAST_F ui16var_fin,
};

static void ui8var_red(UI8_DPARAMS_R) { VAR_R }
static void ui8var_ini(DPARAMS_I) { VAR_I }
static void ui8var_fin(DPARAMS_F) { VAR_F }

static iarray_reduce_function_t UI8VAR = {
        .init = CAST_I ui8var_ini,
        .reduction = CAST_R ui8var_red,
        .finish = CAST_F ui8var_fin,
};

static void boolvar_red(BOOL_DPARAMS_R) { VAR_R }
static void boolvar_ini(DPARAMS_I) { VAR_I }
static void boolvar_fin(DPARAMS_F) { VAR_F }

static iarray_reduce_function_t BOOLVAR = {
        .init = CAST_I boolvar_ini,
        .reduction = CAST_R boolvar_red,
        .finish = CAST_F boolvar_fin,
};

#endif //IARRAY_IARRAY_REDUCE_VAR_H

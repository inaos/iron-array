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

#define STD_I \
    VAR_I

#define STD_R \
    VAR_R

#define STD_F \
    VAR_F     \
    *res = sqrt(*res);

static void dstd_red(DPARAMS_R) { STD_R }
static void dstd_init(DPARAMS_I) { STD_I }
static void dstd_finish(DPARAMS_F) { STD_F }


static iarray_reduce_function_t DSTD = {
        .init = CAST_I dstd_init,
        .reduction = CAST_R dstd_red,
        .finish = CAST_F dstd_finish,
};

#define NANSTD_I \
    NANVAR_I

#define NANSTD_R \
    NANVAR_R

#define NANSTD_F \
    NANVAR_F     \
    *res = sqrt(*res);

static void dnanstd_red(DPARAMS_R) { NANSTD_R }
static void dnanstd_init(DPARAMS_I) { NANSTD_I }
static void dnanstd_finish(DPARAMS_F) { NANSTD_F }


static iarray_reduce_function_t DNANSTD = {
        .init = CAST_I dnanstd_init,
        .reduction = CAST_R dnanstd_red,
        .finish = CAST_F dnanstd_finish,
};

// Only used for float output
#define FSTD_R \
    FVAR_R

#define FSTD_I \
    FVAR_I

#define FSTD_F \
    FVAR_F \
    *res = sqrtf(*res);

static void fstd_red(FPARAMS_R) { FSTD_R }
static void fstd_ini(FPARAMS_I) { FSTD_I }
static void fstd_fin(FPARAMS_F) { FSTD_F }

static iarray_reduce_function_t FSTD = {
        .init = CAST_I fstd_ini,
        .reduction = CAST_R fstd_red,
        .finish = CAST_F fstd_fin,
};

#define FNANSTD_R \
    FNANVAR_R

#define FNANSTD_I \
    FNANVAR_I

#define FNANSTD_F \
    FNANVAR_F \
    *res = sqrtf(*res);

static void fnanstd_red(FPARAMS_R) { FNANSTD_R }
static void fnanstd_ini(FPARAMS_I) { FNANSTD_I }
static void fnanstd_fin(FPARAMS_F) { FNANSTD_F }

static iarray_reduce_function_t FNANSTD = {
        .init = CAST_I fnanstd_ini,
        .reduction = CAST_R fnanstd_red,
        .finish = CAST_F fnanstd_fin,
};

static void i64std_red(I64_DPARAMS_R) { STD_R }
static void i64std_ini(DPARAMS_I) { STD_I }
static void i64std_fin(DPARAMS_F) { STD_F }

static iarray_reduce_function_t I64STD = {
        .init = CAST_I i64std_ini,
        .reduction = CAST_R i64std_red,
        .finish = CAST_F i64std_fin,
};

static void i32std_red(I32_DPARAMS_R) { STD_R }
static void i32std_ini(DPARAMS_I) { STD_I }
static void i32std_fin(DPARAMS_F) { STD_F }

static iarray_reduce_function_t I32STD = {
        .init = CAST_I i32std_ini,
        .reduction = CAST_R i32std_red,
        .finish = CAST_F i32std_fin,
};

static void i16std_red(I16_DPARAMS_R) { STD_R }
static void i16std_ini(DPARAMS_I) { STD_I }
static void i16std_fin(DPARAMS_F) { STD_F }

static iarray_reduce_function_t I16STD = {
        .init = CAST_I i16std_ini,
        .reduction = CAST_R i16std_red,
        .finish = CAST_F i16std_fin,
};

static void i8std_red(I8_DPARAMS_R) { STD_R }
static void i8std_ini(DPARAMS_I) { STD_I }
static void i8std_fin(DPARAMS_F) { STD_F }

static iarray_reduce_function_t I8STD = {
        .init = CAST_I i8std_ini,
        .reduction = CAST_R i8std_red,
        .finish = CAST_F i8std_fin,
};

static void ui64std_red(UI64_DPARAMS_R) { STD_R }
static void ui64std_ini(DPARAMS_I) { STD_I }
static void ui64std_fin(DPARAMS_F) { STD_F }

static iarray_reduce_function_t UI64STD = {
        .init = CAST_I ui64std_ini,
        .reduction = CAST_R ui64std_red,
        .finish = CAST_F ui64std_fin,
};

static void ui32std_red(UI32_DPARAMS_R) { STD_R }
// static void u32std_ini(DPARAMS_I) { STD_I }
// static void u32std_fin(DPARAMS_F) { STD_F }

static iarray_reduce_function_t UI32STD = {
        .init = CAST_I ui64std_ini,
        .reduction = CAST_R ui32std_red,
        .finish = CAST_F ui64std_fin,
};

static void ui16std_red(UI16_DPARAMS_R) { STD_R }
static void ui16std_ini(DPARAMS_I) { STD_I }
static void ui16std_fin(DPARAMS_F) { STD_F }

static iarray_reduce_function_t UI16STD = {
        .init = CAST_I ui16std_ini,
        .reduction = CAST_R ui16std_red,
        .finish = CAST_F ui16std_fin,
};

static void ui8std_red(UI8_DPARAMS_R) { STD_R }
static void ui8std_ini(DPARAMS_I) { STD_I }
static void ui8std_fin(DPARAMS_F) { STD_F }

static iarray_reduce_function_t UI8STD = {
        .init = CAST_I ui8std_ini,
        .reduction = CAST_R ui8std_red,
        .finish = CAST_F ui8std_fin,
};

static void boolstd_red(BOOL_DPARAMS_R) { STD_R }
static void boolstd_ini(DPARAMS_I) { STD_I }
static void boolstd_fin(DPARAMS_F) { STD_F }

static iarray_reduce_function_t BOOLSTD = {
        .init = CAST_I boolstd_ini,
        .reduction = CAST_R boolstd_red,
        .finish = CAST_F boolstd_fin,
};

#endif //IARRAY_IARRAY_REDUCE_STD_H

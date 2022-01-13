/*
 * Copyright INAOS GmbH, Thalwil, 2019.
 * Copyright Francesc Alted, 2019.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of INAOS GmbH
 * and Francesc Alted ("Confidential Information"). You shall not disclose such Confidential
 * Information and shall use it only in accordance with the terms of the license agreement.
 *
 */

#include <math.h>
#include <limits.h>
#include <stdint.h>


struct iarray_reduce_function_s {
    void (*init)(void *, void *);
    void (*reduction)(void *, int64_t, void *, int64_t, int64_t, void *);
    void (*finish)(void *, void *);
};


#define CAST_I (void (*)(void *, void *))
#define CAST_R (void (*)(void *, int64_t, void *, int64_t, int64_t, void *))
#define CAST_F (void (*)(void *, void *))

#define DPARAMS_I double *res, void *user_data

#define DPARAMS_R double *data0, int64_t strides0, \
                  double *data1, int64_t strides1, \
                  int64_t nelem, void *user_data

#define DPARAMS_F double *res, void *user_data

#define FPARAMS_I float *res, void *user_data
#define FPARAMS_R float *data0, int64_t strides0, \
                  float *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
#define FPARAMS_F float *res, void *user_data

#define I64PARAMS_I int64_t *res, void *user_data
#define I64PARAMS_R int64_t *data0, int64_t strides0, \
                  int64_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as double
#define I64_DPARAMS_R double *data0, int64_t strides0, \
                  int64_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
#define I64PARAMS_F int64_t *res, void *user_data

#define I32PARAMS_I int32_t *res, void *user_data
#define I32PARAMS_R int32_t *data0, int64_t strides0, \
                  int32_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as int64_t
#define I32_64PARAMS_R int64_t *data0, int64_t strides0, \
                  int32_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as double
#define I32_DPARAMS_R double *data0, int64_t strides0, \
                  int32_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
#define I32PARAMS_F int32_t *res, void *user_data

#define I16PARAMS_I int16_t *res, void *user_data
#define I16PARAMS_R int16_t *data0, int64_t strides0, \
                  int16_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as int64_t
#define I16_64PARAMS_R int64_t *data0, int64_t strides0, \
                  int16_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as double
#define I16_DPARAMS_R double *data0, int64_t strides0, \
                  int16_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
#define I16PARAMS_F int16_t *res, void *user_data

#define I8PARAMS_I int8_t *res, void *user_data
#define I8PARAMS_R int8_t *data0, int64_t strides0, \
                  int8_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as int64_t
#define I8_64PARAMS_R int64_t *data0, int64_t strides0, \
                  int8_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as double
#define I8_DPARAMS_R double *data0, int64_t strides0, \
                  int8_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
#define I8PARAMS_F int8_t *res, void *user_data

#define UI64PARAMS_I uint64_t *res, void *user_data
#define UI64PARAMS_R uint64_t *data0, int64_t strides0, \
                  uint64_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as double
#define UI64_DPARAMS_R double *data0, int64_t strides0, \
                  uint64_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
#define UI64PARAMS_F uint64_t *res, void *user_data

#define UI32PARAMS_I uint32_t *res, void *user_data
#define UI32PARAMS_R uint32_t *data0, int64_t strides0, \
                  uint32_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as uint64_t
#define UI32_64PARAMS_R uint64_t *data0, int64_t strides0, \
                  uint32_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as double
#define UI32_DPARAMS_R double *data0, int64_t strides0, \
                  uint32_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
#define UI32PARAMS_F uint32_t *res, void *user_data

#define UI16PARAMS_I uint16_t *res, void *user_data
#define UI16PARAMS_R uint16_t *data0, int64_t strides0, \
                  uint16_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as uint64_t
#define UI16_64PARAMS_R uint64_t *data0, int64_t strides0, \
                  uint16_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as double
#define UI16_DPARAMS_R double *data0, int64_t strides0, \
                  uint16_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
#define UI16PARAMS_F uint16_t *res, void *user_data

#define UI8PARAMS_I uint8_t *res, void *user_data
#define UI8PARAMS_R uint8_t *data0, int64_t strides0, \
                  uint8_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as uint64_t
#define UI8_64PARAMS_R uint64_t *data0, int64_t strides0, \
                  uint8_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as double
#define UI8_DPARAMS_R double *data0, int64_t strides0, \
                  uint8_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
#define UI8PARAMS_F uint8_t *res, void *user_data

#define BOOLPARAMS_I bool *res, void *user_data
#define BOOLPARAMS_R bool *data0, int64_t strides0, \
                  bool *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as int64_t
#define BOOL_64PARAMS_R int64_t *data0, int64_t strides0, \
                  bool *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as double
#define BOOL_DPARAMS_R double *data0, int64_t strides0, \
                  bool *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
#define BOOLPARAMS_F bool *res, void *user_data

/* SUM REDUCTION */

#define SUM_I \
        INA_UNUSED(user_data); \
        *res = 0;

#define SUM_R \
    INA_UNUSED(user_data); \
    for (int i = 0; i < nelem; ++i) { \
        *data0 = *data0 + *data1; \
        data1 += strides1; \
    }

#define SUM_F \
    INA_UNUSED(user_data); \
    ;

static void dsum_ini(DPARAMS_I) { SUM_I }
static void dsum_red(DPARAMS_R) { SUM_R }
static void dsum_fin(DPARAMS_F) { SUM_F }

static iarray_reduce_function_t DSUM = {
        .init = CAST_I dsum_ini,
        .reduction = CAST_R dsum_red,
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

/* PROD REDUCTION */

#define PROD_I \
    INA_UNUSED(user_data); \
    *res = 1;

#define PROD_R \
    INA_UNUSED(user_data); \
    for (int i = 0; i < nelem; ++i) { \
        *data0 = *data0 * *data1; \
        data1 += strides1; \
    }

#define PROD_F \
    INA_UNUSED(user_data); \
    ;

static void dprod_ini(DPARAMS_I) { PROD_I }
static void dprod_red(DPARAMS_R) { PROD_R }
static void dprod_fin(DPARAMS_F) { PROD_F }

static iarray_reduce_function_t DPROD = {
        .init = CAST_I dprod_ini,
        .reduction = CAST_R dprod_red,
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


/* MAX REDUCTION */

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
    for (int i = 0; i < nelem; ++i) { \
        if (*data1 > *data0) { \
            *data0 = *data1; \
        } \
        data1 += strides1; \
    }

#define MAX_F \
    INA_UNUSED(user_data); \
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

/* MIN REDUCTION */

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
    for (int i = 0; i < nelem; ++i) { \
        if (*data1 < *data0) {    \
            *data0 = *data1; \
        } \
        data1 += strides1; \
    }

#define MIN_F \
    INA_UNUSED(user_data); \
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

/* MEAN REDUCTION */

#define MEAN_I \
    INA_UNUSED(user_data); \
    *res = 0;

#define MEAN_R \
    INA_UNUSED(user_data); \
     for (int i = 0; i < nelem; ++i) { \
        *data0 = *data0 + *data1; \
        data1 += strides1; \
    }

typedef struct user_data_s {
    double inv_nelem;
} user_data_t;

#define MEAN_F \
    INA_UNUSED(user_data); \
    user_data_t *u_data = (user_data_t *) user_data; \
    *res = *res * u_data->inv_nelem;

static void dmean_ini(DPARAMS_I) { MEAN_I }
static void dmean_red(DPARAMS_R) { MEAN_R }
static void dmean_fin(DPARAMS_F) { MEAN_F }

static iarray_reduce_function_t DMEAN = {
        .init = CAST_I dmean_ini,
        .reduction = CAST_R dmean_red,
        .finish = CAST_F dmean_fin
};

static void fmean_ini(FPARAMS_I) { MEAN_I }
static void fmean_red(FPARAMS_R) { MEAN_R }
static void fmean_fin(FPARAMS_F) { MEAN_F }

static iarray_reduce_function_t FMEAN = {
        .init = CAST_I fmean_ini,
        .reduction = CAST_R fmean_red,
        .finish = CAST_F fmean_fin
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

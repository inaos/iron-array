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
#include <stdint.h>


struct iarray_reduce_function_s {
    void (*init)(void *, void *);
    void (*reduction)(void *, void *, int64_t, int64_t, void *);
    void (*finish)(void *, void *);
};


#define CAST_I (void (*)(void *, void *))
#define CAST_R (void (*)(void *, void *, int64_t, int64_t, void *))
#define CAST_F (void (*)(void *, void *))

#define DPARAMS_I double *res, void *user_data

#define DPARAMS_R double *data0, \
                  double *data1, int64_t strides1, \
                  int64_t nelem, void *user_data

#define DPARAMS_F double *res, void *user_data

#define FPARAMS_I float *res, void *user_data
#define FPARAMS_R float *data0, \
                  float *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
#define FPARAMS_F float *res, void *user_data


/* SUM REDUCTION */

#define SUM_I \
    INA_UNUSED(user_data); \
    *res = 0;

#define SUM_R(func) \
    INA_UNUSED(user_data); \
    *data0 += func(nelem, data1, strides1);

#define SUM_F \
    INA_UNUSED(res); \
    INA_UNUSED(user_data);


static void dsum_ini(DPARAMS_I) { SUM_I }
static void dsum_red(DPARAMS_R) { SUM_R(cblas_dasum); }
static void dsum_fin(DPARAMS_F) { SUM_F }

static iarray_reduce_function_t DSUM = {
        .init = CAST_I dsum_ini,
        .reduction = CAST_R dsum_red,
        .finish = CAST_F dsum_fin
};

static void fsum_ini(FPARAMS_I) { SUM_I }
static void fsum_red(FPARAMS_R) { SUM_R(cblas_sasum); }
static void fsum_fin(FPARAMS_F) { SUM_F }

static iarray_reduce_function_t FSUM = {
        .init = CAST_I fsum_ini,
        .reduction = CAST_R fsum_red,
        .finish = CAST_F fsum_fin
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
    INA_UNUSED(res); \
    INA_UNUSED(user_data);


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


/* MAX REDUCTION */

#define MAX_I \
    INA_UNUSED(user_data); \
    *res = -INFINITY;

#define MAX_R \
    INA_UNUSED(user_data); \
    for (int i = 0; i < nelem; ++i) { \
        if (*data1 > *data0) { \
            *data0 = *data1; \
        } \
        data1 += strides1; \
    }

#define MAX_F \
    INA_UNUSED(res); \
    INA_UNUSED(user_data);

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


/* MIN REDUCTION */

#define MIN_I \
    INA_UNUSED(user_data); \
    *res = INFINITY;

#define MIN_R \
    INA_UNUSED(user_data); \
    for (int i = 0; i < nelem; ++i) { \
        if (*data1 < *data0) {    \
            *data0 = *data1; \
        } \
        data1 += strides1; \
    }

#define MIN_F \
    INA_UNUSED(res); \
    INA_UNUSED(user_data);

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

/* MEAN REDUCTION */

#define MEAN_I \
    INA_UNUSED(user_data); \
    *res = 0;

#define MEAN_R(func) \
    *data0 += func(nelem, data1, strides1);

typedef struct user_data_s {
    double inv_nelem;
} user_data_t;

#define MEAN_F \
    INA_UNUSED(user_data); \
    user_data_t *u_data = (user_data_t *) user_data; \
    *res = *res * u_data->inv_nelem;

static void dmean_ini(DPARAMS_I) { MEAN_I }
static void dmean_red(DPARAMS_R) { MEAN_R(cblas_dasum) }
static void dmean_fin(DPARAMS_F) { MEAN_F }

static iarray_reduce_function_t DMEAN = {
        .init = CAST_I dmean_ini,
        .reduction = CAST_R dmean_red,
        .finish = CAST_F dmean_fin
};

static void fmean_ini(FPARAMS_I) { MEAN_I }
static void fmean_red(FPARAMS_R) { MEAN_R(cblas_sasum) }
static void fmean_fin(FPARAMS_F) { MEAN_F }

static iarray_reduce_function_t FMEAN = {
        .init = CAST_I fmean_ini,
        .reduction = CAST_R fmean_red,
        .finish = CAST_F fmean_fin
};

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

#define VAR_I(itype, otype, nan) \
    user_data_os_t *u_data = (user_data_os_t *) user_data; \
    *res = 0; \
    u_data->not_nan_nelems[u_data->i] = 0;

#define nanVAR_I(itype, otype, nan) \
    VAR_I(itype, otype, nan)

#define VAR_R(itype, otype, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0);  \
    INA_UNUSED(strides1);  \
    INA_UNUSED(nelem);  \
    user_data_os_t *u_data = (user_data_os_t *) user_data; \
    const double *mean = (double *) u_data->mean;    \
    double dif = (double) *data1 - *mean;          \
    *data0 += dif * dif;

#define nanVAR_R(itype, otype, nan) \
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

#define VAR_F(itype, otype, nan) \
    INA_UNUSED(user_data); \
    user_data_os_t *u_data = (user_data_os_t *) user_data; \
    *res = *res * u_data->inv_nelem;

#define nanVAR_F(itype, otype, nan) \
    INA_UNUSED(user_data); \
    user_data_os_t *u_data = (user_data_os_t *) user_data; \
    if (u_data->not_nan_nelems[u_data->i] - u_data->rparams->correction < 0){\
        *res = *res / u_data->not_nan_nelems[u_data->i];\
    }\
    else {\
        *res = *res / (u_data->not_nan_nelems[u_data->i] - u_data->rparams->correction);\
    }\

// Only used for float output
#define FVAR_I(itype, otype, nan) \
    VAR_I(itype, otype, nan)

#define FnanVAR_I(itype, otype, nan) \
    nanVAR_I(itype, otype, nan)

#define FVAR_R(itype, otype, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0);  \
    INA_UNUSED(strides1);  \
    INA_UNUSED(nelem);  \
    user_data_os_t *u_data = (user_data_os_t *) user_data; \
    const float *mean = (float *) u_data->mean;      \
    float dif = (float) *data1 - *mean;          \
    *data0 += dif * dif;

#define FnanVAR_R(itype, otype, nan) \
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

#define FVAR_F(itype, otype, nan) \
    VAR_F(itype, otype, nan)

#define FnanVAR_F(itype, otype, nan) \
    nanVAR_F(itype, otype, nan)


#define VAR(itype, otype, nan, f) \
    static void itype##_##nan##_var_ini(PARAMS_O_I(itype, otype)) { \
        f##nan##VAR_I(itype, otype, nan) \
    } \
    static void itype##_##nan##_var_red(PARAMS_O_R(itype, otype)) { \
        f##nan##VAR_R(itype, otype, nan) \
    } \
    static void itype##_##nan##_var_fin(PARAMS_O_F(itype, otype)) { \
        f##nan##VAR_F(itype, otype, nan) \
    } \
    static iarray_reduce_function_t itype##nan##_VAR = { \
            .init = CAST_I itype##_##nan##_var_ini, \
            .reduction = CAST_R itype##_##nan##_var_red, \
            .finish = CAST_F itype##_##nan##_var_fin, \
    };

VAR(double, double, , )
VAR(float, float, , F)
VAR(int64_t, double, , )
VAR(int32_t, double, , )
VAR(int16_t, double, , )
VAR(int8_t, double, , )
VAR(uint64_t, double, , )
VAR(uint32_t, double, , )
VAR(uint16_t, double, , )
VAR(uint8_t, double, , )
VAR(bool, double, , )
VAR(double, double, nan, )
VAR(float, float, nan, F)

#endif //IARRAY_IARRAY_REDUCE_VAR_H

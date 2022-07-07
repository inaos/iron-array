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


#define MEAN_I(itype, otype, nan) \
    INA_UNUSED(user_data); \
    user_data_os_t *u_data = (user_data_os_t *) user_data; \
    *res = 0; \
    u_data->not_nan_nelems[u_data->i] = 0;

#define nanMEAN_I(itype, otype, nan) \
    user_data_os_t *u_data = (user_data_os_t *) user_data; \
    u_data->not_nan_nelems[u_data->i] = 0; \
    *res = 0;

#define MEAN_R(itype, otype, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    for (int i = 0; i < nelem; ++i) { \
        *data0 = *data0 + *data1; \
        data1 += strides1; \
    }

#define nanMEAN_R(itype, otype, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0);  \
    user_data_os_t *u_data = (user_data_os_t *) user_data; \
    if (!isnan(*data1)) {          \
        *data0 = *data0 + *data1;                      \
        u_data->not_nan_nelems[u_data->i]++;             \
   }

#define MEAN_F(itype, otype, nan) \
    user_data_os_t *u_data = (user_data_os_t *) user_data; \
    *res = *res * u_data->inv_nelem;

#define nanMEAN_F(itype, otype, nan) \
    user_data_os_t *u_data = (user_data_os_t *) user_data; \
    *res = *res / u_data->not_nan_nelems[u_data->i];


#define MEAN(itype, otype, nan) \
    static void itype##_##nan##_mean_ini(PARAMS_O_I(itype, otype)) { \
        nan##MEAN_I(itype, otype, nan) \
    } \
    static void itype##_##nan##_mean_red(PARAMS_O_R(itype, otype)) { \
        nan##MEAN_R(itype, otype, nan) \
    } \
    static void itype##_##nan##_mean_fin(PARAMS_O_F(itype, otype)) { \
        nan##MEAN_F(itype, otype, nan) \
    } \
    static iarray_reduce_function_t itype##nan##_MEAN = { \
            .init = CAST_I itype##_##nan##_mean_ini, \
            .reduction = CAST_R itype##_##nan##_mean_red, \
            .finish = CAST_F itype##_##nan##_mean_fin, \
    };

MEAN(double, double, )
MEAN(float, float, )
MEAN(int64_t, double, )
MEAN(int32_t, double, )
MEAN(int16_t, double, )
MEAN(int8_t, double, )
MEAN(uint64_t, double, )
MEAN(uint32_t, double, )
MEAN(uint16_t, double, )
MEAN(uint8_t, double, )
MEAN(bool, double, )
MEAN(double, double, nan)
MEAN(float, float, nan)

#endif //IARRAY_IARRAY_REDUCE_MEAN_H


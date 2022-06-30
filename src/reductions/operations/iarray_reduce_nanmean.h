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

#ifndef IARRAY_IARRAY_REDUCE_NANMEAN_H
#define IARRAY_IARRAY_REDUCE_NANMEAN_H


#include "iarray_reduce_private.h"

#define NANMEAN_I(itype, otype) \
    user_data_os_t *u_data = (user_data_os_t *) user_data; \
    u_data->not_nan_nelems[u_data->i] = 0; \
    *res = 0;

#define NANMEAN_R(itype, otype) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0);  \
    user_data_os_t *u_data = (user_data_os_t *) user_data; \
    if (!isnan(*data1)) {          \
        *data0 = *data0 + *data1;                      \
        u_data->not_nan_nelems[u_data->i]++;             \
   }

#define NANMEAN_F(itype, otype) \
    user_data_os_t *u_data = (user_data_os_t *) user_data; \
    *res = *res / u_data->not_nan_nelems[u_data->i];

#define NANMEAN(itype, otype) \
    static void itype##_nanmean_ini(PARAMS_O_I(itype, otype)) { \
        NANMEAN_I(itype, otype) \
    } \
    static void itype##_nanmean_red(PARAMS_O_R(itype, otype)) { \
        NANMEAN_R(itype, otype) \
    } \
    static void itype##_nanmean_fin(PARAMS_O_F(itype, otype)) { \
        NANMEAN_F(itype, otype) \
    } \
    static iarray_reduce_function_t itype##_NANMEAN = { \
            .init = CAST_I itype##_nanmean_ini, \
            .reduction = CAST_R itype##_nanmean_red, \
            .finish = CAST_F itype##_nanmean_fin, \
    };

NANMEAN(double, double)
NANMEAN(float, float)

#endif //IARRAY_IARRAY_REDUCE_NANMEAN_H

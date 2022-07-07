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

#ifndef IARRAY_IARRAY_REDUCE_SUM_H
#define IARRAY_IARRAY_REDUCE_SUM_H

#include "iarray_reduce_private.h"

#define SUM_I(itype, otype, nan) \
    INA_UNUSED(user_data); \
    *res = 0;

#define SUM_R(itype, otype, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    for (int i = 0; i < nelem; ++i) { \
        *data0 = *data0 + *data1; \
        data1 += strides1; \
    }

#define nanSUM_R(itype, otype, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    for (int i = 0; i < nelem; ++i) { \
        if (!isnan(*data1)) {         \
            *data0 = *data0 + *data1; \
        }\
        data1 += strides1; \
    }

#define SUM_F(itype, otype, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(res); \
    ;


#define SUM(itype, otype, nan) \
    static void itype##_##nan##_sum_ini(PARAMS_O_I(itype, otype)) { \
        SUM_I(itype, otype, nan) \
    } \
    static void itype##_##nan##_sum_red(PARAMS_O_R(itype, otype)) { \
        nan##SUM_R(itype, otype, nan) \
    } \
    static void itype##_##nan##_sum_fin(PARAMS_O_F(itype, otype)) { \
        SUM_F(itype, otype, nan) \
    } \
    static iarray_reduce_function_t itype##nan##_SUM = { \
            .init = CAST_I itype##_##nan##_sum_ini, \
            .reduction = CAST_R itype##_##nan##_sum_red, \
            .finish = CAST_F itype##_##nan##_sum_fin, \
    };

SUM(double, double, )
SUM(float, float, )
SUM(int64_t, int64_t, )
SUM(int32_t, int64_t, )
SUM(int16_t, int64_t, )
SUM(int8_t, int64_t, )
SUM(uint64_t, uint64_t, )
SUM(uint32_t, uint64_t, )
SUM(uint16_t, uint64_t, )
SUM(uint8_t, uint64_t, )
SUM(bool, int64_t, )
SUM(double, double, nan)
SUM(float, float, nan)

#endif //IARRAY_IARRAY_REDUCE_SUM_H

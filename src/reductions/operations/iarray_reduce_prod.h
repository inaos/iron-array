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

#define PROD_I(itype, otype, nan) \
    INA_UNUSED(user_data); \
    *res = 1;

#define PROD_R(itype, otype, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    for (int i = 0; i < nelem; ++i) { \
        *data0 = *data0 * *data1; \
        data1 += strides1; \
    }

#define oneshotPROD_R(itype, otype, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    INA_UNUSED(strides1);  \
    INA_UNUSED(nelem); \
    *data0 = *data0 * *data1;

#define nanPROD_R(itype, otype, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    for (int i = 0; i < nelem; ++i) { \
        if (!isnan(*data1)) {          \
            *data0 = *data0 * *data1; \
        }\
        data1 += strides1; \
    }

#define oneshotnanPROD_R(itype, otype, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    INA_UNUSED(strides1);  \
    INA_UNUSED(nelem); \
    if (!isnan(*data1)) {          \
        *data0 = *data0 * *data1; \
    }

#define PROD_F(itype, otype, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(res); \
    ;


#define PROD(itype, otype, nan, oneshot) \
    static void itype##_##oneshot##_##nan##_prod_ini(PARAMS_O_I(itype, otype)) { \
        PROD_I(itype, otype, nan) \
    } \
    static void itype##_##oneshot##_##nan##_prod_red(PARAMS_O_R(itype, otype)) { \
        oneshot##nan##PROD_R(itype, otype, nan) \
    } \
    static void itype##_##oneshot##_##nan##_prod_fin(PARAMS_O_F(itype, otype)) { \
        PROD_F(itype, otype, nan) \
    } \
    static iarray_reduce_function_t itype##oneshot##nan##_PROD = { \
            .init = CAST_I itype##_##oneshot##_##nan##_prod_ini, \
            .reduction = CAST_R itype##_##oneshot##_##nan##_prod_red, \
            .finish = CAST_F itype##_##oneshot##_##nan##_prod_fin, \
    };

PROD(double, double, , )
PROD(float, float, , )
PROD(int64_t, int64_t, , )
PROD(int32_t, int64_t, , )
PROD(int16_t, int64_t, , )
PROD(int8_t, int64_t, , )
PROD(uint64_t, uint64_t, , )
PROD(uint32_t, uint64_t, , )
PROD(uint16_t, uint64_t, , )
PROD(uint8_t, uint64_t, , )
PROD(bool, int64_t, , )
PROD(double, double, nan, )
PROD(float, float, nan, )

PROD(double, double, , oneshot)
PROD(float, float, , oneshot)
PROD(int64_t, int64_t, , oneshot)
PROD(int32_t, int64_t, , oneshot)
PROD(int16_t, int64_t, , oneshot)
PROD(int8_t, int64_t, , oneshot)
PROD(uint64_t, uint64_t, , oneshot)
PROD(uint32_t, uint64_t, , oneshot)
PROD(uint16_t, uint64_t, , oneshot)
PROD(uint8_t, uint64_t, , oneshot)
PROD(bool, int64_t, , oneshot)
PROD(double, double, nan, oneshot)
PROD(float, float, nan, oneshot)



#endif //IARRAY_IARRAY_REDUCE_PROD_H

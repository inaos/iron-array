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

#ifndef IARRAY_IARRAY_REDUCE_MAX_H
#define IARRAY_IARRAY_REDUCE_MAX_H

#include "iarray_reduce_private.h"


#define MAX_I(type, inival, nan) \
    INA_UNUSED(user_data); \
    *res = inival;

#define MAX_R(type, inival, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    for (int i = 0; i < nelem; ++i) { \
        if (*data1 > *data0) { \
            *data0 = *data1; \
        } \
        data1 += strides1; \
    }

#define DFMAX_R(type, inival, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    for (int i = 0; i < nelem; ++i) { \
        if(isnan(*data1)) {\
            *data0 = NAN;  \
            break;      \
        }\
        if (*data1 > *data0) { \
            *data0 = *data1; \
        } \
        data1 += strides1; \
    }

#define nanMAX_R(type, inival, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    for (int i = 0; i < nelem; ++i) { \
        if (!isnan(*data1) && (*data1 > *data0 || isnan(*data0))) {    \
            *data0 = *data1; \
        } \
        data1 += strides1; \
    }

#define MAX_F(type, inival, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(res); \
    ;


#define MAX(type, inival, nan, rprefix) \
    static void type##_##nan##_max_ini(PARAMS_O_I(type, type)) { \
        MAX_I(type, inival, nan) \
    } \
    static void type##_##nan##_max_red(PARAMS_O_R(type, type)) { \
        rprefix##nan##MAX_R(type, inival, nan) \
    } \
    static void type##_##nan##_max_fin(PARAMS_O_F(type, type)) { \
        MAX_F(type, inival, nan) \
    } \
    static iarray_reduce_function_t type##nan##_MAX = { \
            .init = CAST_I type##_##nan##_max_ini, \
            .reduction = CAST_R type##_##nan##_max_red, \
            .finish = CAST_F type##_##nan##_max_fin, \
    };

MAX(double, -INFINITY, , DF)
MAX(float, -INFINITY, , DF)
MAX(int64_t, LLONG_MIN, , )
MAX(int32_t, INT_MIN, , )
MAX(int16_t, SHRT_MIN, , )
MAX(int8_t, SCHAR_MIN, , )
MAX(uint64_t, 0ULL, , )
MAX(uint32_t, 0U, , )
MAX(uint16_t, 0, , )
MAX(uint8_t, 0, , )
MAX(bool, false, , )
MAX(double, NAN, nan, )
MAX(float, NAN, nan, )

#endif //IARRAY_IARRAY_REDUCE_MAX_H

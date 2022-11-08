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

#ifndef IARRAY_IARRAY_REDUCE_ALL_H
#define IARRAY_IARRAY_REDUCE_ALL_H

#include "iarray_reduce_private.h"


#define ALL_I(type, inival, nan) \
    INA_UNUSED(user_data);       \
    *res = inival;

#define ALL_R(type, inival, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0);        \
    bool value;                              \
    for (int i = 0; i < nelem; ++i) {        \
        if (!(bool)(*data1)) {   \
            *data0 = 0;      \
            break;               \
        } \
        data1 += strides1; \
    }

#define oneshotALL_R(type, inival, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    INA_UNUSED(strides1);  \
    INA_UNUSED(nelem); \
    if (!(bool)(*data1)) { \
        *data0 = false;  \
    }

#define DFALL_R(type, inival, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0);          \
    for (int i = 0; i < nelem; ++i) { \
        if (isfinite(*data1) && !(bool)(*data1)) { \
            *data0 = false;        \
            break;                  \
        } \
        data1 += strides1; \
    }

#define oneshotDFALL_R(type, inival, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    INA_UNUSED(strides1);  \
    INA_UNUSED(nelem); \
    if (isfinite(*data1) && !(bool)(*data1)) { \
        *data0 = false;                   \
    }                                 \

#define ALL_F(type, inival, nan) \
    INA_UNUSED(user_data);       \
    INA_UNUSED(res); \
    ;


#define ALL(type, inival, nan, rprefix, oneshot) \
    static void type##_##oneshot##_##nan##_all_ini(PARAMS_O_I(type, bool)) { \
        ALL_I(type, inival, nan) \
    } \
    static void type##_##oneshot##_##nan##_all_red(PARAMS_O_R(type, bool)) { \
        oneshot##rprefix##nan##ALL_R(type, inival, nan) \
    } \
    static void type##_##oneshot##_##nan##_all_fin(PARAMS_O_F(type, bool)) { \
        ALL_F(type, inival, nan) \
    } \
    static iarray_reduce_function_t type##oneshot##nan##_ALL = { \
            .init = CAST_I type##_##oneshot##_##nan##_all_ini, \
            .reduction = CAST_R type##_##oneshot##_##nan##_all_red, \
            .finish = CAST_F type##_##oneshot##_##nan##_all_fin, \
    };

ALL(double, true, , DF, )
ALL(float, true, , DF, )
ALL(int64_t, true, , , )
ALL(int32_t, true, , , )
ALL(int16_t, true, , , )
ALL(int8_t, true, , , )
ALL(uint64_t, true, , , )
ALL(uint32_t, true, , , )
ALL(uint16_t, true, , , )
ALL(uint8_t, true, , , )
ALL(bool, true, , , )

// Oneshot
ALL(double, true, , DF, oneshot)
ALL(float, true, , DF, oneshot)
ALL(int64_t, true, , , oneshot)
ALL(int32_t, true, , , oneshot)
ALL(int16_t, true, , , oneshot)
ALL(int8_t, true, , , oneshot)
ALL(uint64_t, true, , , oneshot)
ALL(uint32_t, true, , , oneshot)
ALL(uint16_t, true, , , oneshot)
ALL(uint8_t, true, , , oneshot)
ALL(bool, true, , , oneshot)

#endif //IARRAY_IARRAY_REDUCE_ALL_H

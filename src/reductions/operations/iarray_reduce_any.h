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

#ifndef IARRAY_IARRAY_REDUCE_ANY_H
#define IARRAY_IARRAY_REDUCE_ANY_H

#include "iarray_reduce_private.h"


#define ANY_I(type, inival, nan) \
    INA_UNUSED(user_data); \
    *res = inival;

#define ANY_R(type, inival, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    for (int i = 0; i < nelem; ++i) { \
        if ((bool)(*data1)) { \
            *data0 = true;      \
            break;               \
        } \
        data1 += strides1; \
    }

#define oneshotANY_R(type, inival, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    INA_UNUSED(strides1);  \
    INA_UNUSED(nelem); \
    if ((bool)(*data1)) { \
        *data0 = true;  \
    }

#define DFANY_R(type, inival, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    for (int i = 0; i < nelem; ++i) { \
        if (!isfinite(*data1) || (bool)(*data1)) { \
            *data0 = true;        \
            break;                  \
        } \
        data1 += strides1; \
    }

#define oneshotDFANY_R(type, inival, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    INA_UNUSED(strides1);  \
    INA_UNUSED(nelem); \
    if (!isfinite(*data1) || (bool)(*data1)) { \
        *data0 = true;                   \
    }                                 \

#define ANY_F(type, inival, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(res); \
    ;


#define ANY(type, inival, nan, rprefix, oneshot) \
    static void type##_##oneshot##_##nan##_any_ini(PARAMS_O_I(type, bool)) { \
        ANY_I(type, inival, nan) \
    } \
    static void type##_##oneshot##_##nan##_any_red(PARAMS_O_R(type, bool)) { \
        oneshot##rprefix##nan##ANY_R(type, inival, nan) \
    } \
    static void type##_##oneshot##_##nan##_any_fin(PARAMS_O_F(type, bool)) { \
        ANY_F(type, inival, nan) \
    } \
    static iarray_reduce_function_t type##oneshot##nan##_ANY = { \
            .init = CAST_I type##_##oneshot##_##nan##_any_ini, \
            .reduction = CAST_R type##_##oneshot##_##nan##_any_red, \
            .finish = CAST_F type##_##oneshot##_##nan##_any_fin, \
    };

ANY(double, false, , DF, )
ANY(float, false, , DF, )
ANY(int64_t, false, , , )
ANY(int32_t, false, , , )
ANY(int16_t, false, , , )
ANY(int8_t, false, , , )
ANY(uint64_t, false, , , )
ANY(uint32_t, false, , , )
ANY(uint16_t, false, , , )
ANY(uint8_t, false, , , )
ANY(bool, false, , , )

// Oneshot
ANY(double, false, , DF, oneshot)
ANY(float, false, , DF, oneshot)
ANY(int64_t, false, , , oneshot)
ANY(int32_t, false, , , oneshot)
ANY(int16_t, false, , , oneshot)
ANY(int8_t, false, , , oneshot)
ANY(uint64_t, false, , , oneshot)
ANY(uint32_t, false, , , oneshot)
ANY(uint16_t, false, , , oneshot)
ANY(uint8_t, false, , , oneshot)
ANY(bool, false, , , oneshot)

#endif //IARRAY_IARRAY_REDUCE_ANY_H

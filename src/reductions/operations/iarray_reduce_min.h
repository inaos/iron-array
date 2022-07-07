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

#ifndef IARRAY_IARRAY_REDUCE_MIN_H
#define IARRAY_IARRAY_REDUCE_MIN_H

#include "iarray_reduce_private.h"


#define MIN_I(type, inival, nan) \
    INA_UNUSED(user_data); \
    *res = inival;

#define MIN_R(type, inival, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    for (int i = 0; i < nelem; ++i) { \
        if (*data1 < *data0) {    \
            *data0 = *data1; \
        } \
        data1 += strides1; \
    }

#define oneshotMIN_R(type, inival, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    INA_UNUSED(strides1);  \
    INA_UNUSED(nelem); \
    if (*data1 < *data0) {    \
        *data0 = *data1; \
    }

#define DFMIN_R(type, inival, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    for (int i = 0; i < nelem; ++i) { \
        if(isnan(*data1)) {\
            *data0 = NAN;  \
            break;      \
        }\
        if (*data1 < *data0) {    \
            *data0 = *data1; \
        } \
        data1 += strides1; \
    }

#define oneshotDFMIN_R(type, inival, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    INA_UNUSED(strides1);  \
    INA_UNUSED(nelem); \
    if(isnan(*data1)) {\
        *data0 = NAN;  \
    }                                 \
    else {                            \
        if (*data1 < *data0) {    \
            *data0 = *data1; \
        }\
    }

#define nanMIN_R(type, inival, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    for (int i = 0; i < nelem; ++i) { \
        if (!isnan(*data1) && (isnan(*data0) || *data1 < *data0)) {    \
            *data0 = *data1; \
        } \
        data1 += strides1; \
    }

#define oneshotnanMIN_R(type, inival, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0); \
    INA_UNUSED(strides1);  \
    INA_UNUSED(nelem); \
    if (!isnan(*data1) && (isnan(*data0) || *data1 < *data0)) {    \
        *data0 = *data1; \
    }

#define MIN_F(type, inival, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(res); \
    ;


#define MIN(type, inival, nan, rprefix, oneshot) \
    static void type##_##oneshot##_##nan##_min_ini(PARAMS_O_I(type, type)) { \
        MIN_I(type, inival, nan) \
    } \
    static void type##_##oneshot##_##nan##_min_red(PARAMS_O_R(type, type)) { \
        oneshot##rprefix##nan##MIN_R(type, inival, nan) \
    } \
    static void type##_##oneshot##_##nan##_min_fin(PARAMS_O_F(type, type)) { \
        MIN_F(type, inival, nan) \
    } \
    static iarray_reduce_function_t type##oneshot##nan##_MIN = { \
            .init = CAST_I type##_##oneshot##_##nan##_min_ini, \
            .reduction = CAST_R type##_##oneshot##_##nan##_min_red, \
            .finish = CAST_F type##_##oneshot##_##nan##_min_fin, \
    };

MIN(double, INFINITY, , DF, )
MIN(float, INFINITY, , DF, )
MIN(int64_t, LLONG_MAX, , , )
MIN(int32_t, INT_MAX, , , )
MIN(int16_t, SHRT_MAX, , , )
MIN(int8_t, SCHAR_MAX, , , )
MIN(uint64_t, ULLONG_MAX, , , )
MIN(uint32_t, UINT_MAX, , , )
MIN(uint16_t, USHRT_MAX, , , )
MIN(uint8_t, UCHAR_MAX, , , )
MIN(bool, true, , , )
MIN(double, NAN, nan, , )
MIN(float, NAN, nan, , )

MIN(double, INFINITY, , DF, oneshot)
MIN(float, INFINITY, , DF, oneshot)
MIN(int64_t, LLONG_MAX, , , oneshot)
MIN(int32_t, INT_MAX, , , oneshot)
MIN(int16_t, SHRT_MAX, , , oneshot)
MIN(int8_t, SCHAR_MAX, , , oneshot)
MIN(uint64_t, ULLONG_MAX, , , oneshot)
MIN(uint32_t, UINT_MAX, , , oneshot)
MIN(uint16_t, USHRT_MAX, , , oneshot)
MIN(uint8_t, UCHAR_MAX, , , oneshot)
MIN(bool, true, , , oneshot)
MIN(double, NAN, nan, , oneshot)
MIN(float, NAN, nan, , oneshot)

#endif //IARRAY_IARRAY_REDUCE_MIN_H

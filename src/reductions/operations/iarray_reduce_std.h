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

#ifndef IARRAY_IARRAY_REDUCE_STD_H
#define IARRAY_IARRAY_REDUCE_STD_H

#include "iarray_reduce_private.h"

#define STD_I(itype, otype, nan) \
    VAR_I(itype, otype, nan)

#define nanSTD_I(itype, otype, nan) \
    nanVAR_I(itype, otype, nan)

#define STD_R(itype, otype, nan) \
    VAR_R(itype, otype, nan)

#define nanSTD_R(itype, otype, nan) \
    nanVAR_R(itype, otype, nan)

#define STD_F(itype, otype, nan) \
    VAR_F(itype, otype, nan)     \
    *res = sqrt(*res);

#define nanSTD_F(itype, otype, nan) \
    nanVAR_F(itype, otype, nan)     \
    *res = sqrt(*res);

// Only used for float output
#define FSTD_R(itype, otype, nan) \
    FVAR_R(itype, otype, nan)

#define FSTD_I(itype, otype, nan) \
    FVAR_I(itype, otype, nan)

#define FSTD_F(itype, otype, nan) \
    FVAR_F(itype, otype, nan) \
    *res = sqrtf(*res);

#define FnanSTD_R(itype, otype, nan) \
    FnanVAR_R(itype, otype, nan)

#define FnanSTD_I(itype, otype, nan) \
    FnanVAR_I(itype, otype, nan)

#define FnanSTD_F(itype, otype, nan) \
    FnanVAR_F(itype, otype, nan) \
    *res = sqrtf(*res);


#define STD(itype, otype, nan, f) \
    static void itype##_##nan##_std_ini(PARAMS_O_I(itype, otype)) { \
        f##nan##STD_I(itype, otype, nan) \
    } \
    static void itype##_##nan##_std_red(PARAMS_O_R(itype, otype)) { \
        f##nan##STD_R(itype, otype, nan) \
    } \
    static void itype##_##nan##_std_fin(PARAMS_O_F(itype, otype)) { \
        f##nan##STD_F(itype, otype, nan) \
    } \
    static iarray_reduce_function_t itype##nan##_STD = { \
            .init = CAST_I itype##_##nan##_std_ini, \
            .reduction = CAST_R itype##_##nan##_std_red, \
            .finish = CAST_F itype##_##nan##_std_fin, \
    };

STD(double, double, , )
STD(float, float, , F)
STD(int64_t, double, , )
STD(int32_t, double, , )
STD(int16_t, double, , )
STD(int8_t, double, , )
STD(uint64_t, double, , )
STD(uint32_t, double, , )
STD(uint16_t, double, , )
STD(uint8_t, double, , )
STD(bool, double, , )
STD(double, double, nan, )
STD(float, float, nan, F)

#endif //IARRAY_IARRAY_REDUCE_STD_H

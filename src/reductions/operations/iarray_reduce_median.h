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

#ifndef IARRAY_IARRAY_REDUCE_MEDIAN_H
#define IARRAY_IARRAY_REDUCE_MEDIAN_H

#include "iarray_reduce_private.h"


#define COMPARE(type, nan) \
    static int iarray_##type##_##nan##_median_compare(const type *a, const type *b) { \
        return *a > *b ? 1 : (*a < *b ? -1 : 0); \
    }

#define MEDIAN_I(itype, otype, nan) \
    user_data_os_t *u_data = (user_data_os_t *) user_data; \
    u_data->medians[u_data->i] = malloc(u_data->reduced_items * u_data->pparams->out_typesize); \
    u_data->not_nan_nelems[u_data->i] = 0;


#define MEDIAN_R(itype, otype, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0);  \
    user_data_os_t *u_data = (user_data_os_t *) user_data; \
    *((itype *) u_data->median) = *((itype *) data1);       \
    u_data->median += u_data->rparams->input->catarr->itemsize; \
    u_data->median_nelems[u_data->i]++; \
    u_data->not_nan_nelems[u_data->i]++;


#define MEDIAN_F(itype, otype, nan) \
    int (*compare)(const void *a, const void *b) = (int(*)(const void *, const void*)) iarray_##itype##_##nan##_median_compare; \
    user_data_os_t *u_data = (user_data_os_t *) user_data;                                                             \
    int64_t nelem = u_data->not_nan_nelems[u_data->i];                                                                         \
    if(nelem != 0) {                        \
        qsort(u_data->medians[u_data->i], nelem, u_data->input_itemsize, compare);                                                     \
        if (nelem % 2 == 0) {      \
            *res = (otype) ((((itype *) u_data->medians[u_data->i])[(int64_t) (nelem / 2 - 1)] + \
                              ((itype *) u_data->medians[u_data->i])[(int64_t) (nelem / 2)]) * 0.5); \
        } else {     \
            *res = (otype) ((itype *) u_data->medians[u_data->i])[(int64_t) (nelem / 2)];             \
    }                       \
    } else {                \
        *res = NAN;                        \
    }                       \
    free(u_data->medians[u_data->i]); \

#define nanMEDIAN_I(itype, otype, nan) \
    MEDIAN_I(itype, otype, nan)

#define nanMEDIAN_R(itype, otype, nan) \
    INA_UNUSED(user_data); \
    INA_UNUSED(user_data); \
    INA_UNUSED(strides0);  \
    user_data_os_t *u_data = (user_data_os_t *) user_data; \
    itype d1 = *((itype *) data1);                \
    if(!isnan(d1)) {                           \
        *((itype *) u_data->median) = d1;       \
        u_data->median += u_data->rparams->input->catarr->itemsize; \
        u_data->median_nelems[u_data->i]++; \
        u_data->not_nan_nelems[u_data->i]++;                   \
    }
#define nanMEDIAN_F(itype, otype, nan) \
    MEDIAN_F(itype, otype, nan)


#define MEDIAN(itype, otype, nan) \
    COMPARE(itype, nan) \
    static void itype##_##nan##_median_ini(PARAMS_O_I(itype, otype)) { \
        nan##MEDIAN_I(itype, otype, nan) \
    } \
    static void itype##_##nan##_median_red(PARAMS_O_R(itype, otype)) { \
        nan##MEDIAN_R(itype, otype, nan) \
    } \
    static void itype##_##nan##_median_fin(PARAMS_O_F(itype, otype)) { \
        nan##MEDIAN_F(itype, otype, nan) \
    } \
    static iarray_reduce_function_t itype##nan##_MEDIAN = { \
            .init = CAST_I itype##_##nan##_median_ini, \
            .reduction = CAST_R itype##_##nan##_median_red, \
            .finish = CAST_F itype##_##nan##_median_fin, \
    };

MEDIAN(double, double,)
MEDIAN(double, double, nan)
MEDIAN(float,float, )
MEDIAN(float, float, nan)
MEDIAN(int64_t, double,)
MEDIAN(int32_t, double,)
MEDIAN(int16_t, double,)
MEDIAN(int8_t, double,)
MEDIAN(uint64_t, double,)
MEDIAN(uint32_t, double,)
MEDIAN(uint16_t, double,)
MEDIAN(uint8_t, double,)
MEDIAN(bool, double,)

#endif //IARRAY_IARRAY_REDUCE_MEDIAN_H

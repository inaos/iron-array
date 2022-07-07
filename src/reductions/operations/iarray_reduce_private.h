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

#ifndef IARRAY_IARRAY_REDUCE_PRIVATE_H
#define IARRAY_IARRAY_REDUCE_PRIVATE_H

#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <iarray_private.h>


struct iarray_reduce_function_s {
    void (*init)(void *, void *);
    void (*reduction)(void *, int64_t, void *, int64_t, int64_t, void *);
    void (*finish)(void *, void *);
};

typedef struct iarray_reduce_params_s {
    iarray_reduce_function_t *ufunc;
    iarray_container_t *input;
    iarray_container_t *result;
    int8_t axis;
    int64_t *out_chunkshape;
    int64_t nchunk;
} iarray_reduce_params_t;

typedef struct iarray_reduce_os_params_s {
    iarray_reduce_function_t *ufunc;
    iarray_reduce_func_t func;
    iarray_container_t *input;
    iarray_container_t *result;
    int8_t naxis;
    const int8_t *axis;
    int64_t *out_chunkshape;
    int64_t nchunk;
    uint8_t *aux_chunk;
    int32_t aux_csize;
    iarray_container_t *aux;
} iarray_reduce_os_params_t;

typedef struct user_data_os_s {
    blosc2_prefilter_params *pparams;
    iarray_reduce_os_params_t *rparams;
    int64_t reduced_items;
    int64_t i;
    double inv_nelem;
    uint8_t input_itemsize;
    int64_t *not_nan_nelems;
    int64_t *nan_nelems;
    uint8_t *mean;
    uint8_t **medians;
    uint8_t *median;
    int64_t *median_nelems;
} user_data_os_t;


#define REDUCTION(name, type) \
    type##_##name
#define NANREDUCTION(name, type) \
    type##nan_##name

#define CAST_I (void (*)(void *, void *))
#define CAST_R (void (*)(void *, int64_t, void *, int64_t, int64_t, void *))
#define CAST_F (void (*)(void *, void *))


#define PARAMS_O_I(itype, otype) \
    otype *res, void *user_data


#define PARAMS_O_R(itype, otype) \
                      \
    otype *data0, int64_t strides0, \
    const itype *data1, int64_t strides1, \
    int64_t nelem, void *user_data

#define PARAMS_O_F(itype, otype) \
    otype *res, void *user_data


ina_rc_t _iarray_reduce_oneshot(iarray_context_t *ctx,
                                iarray_container_t *a,
                                iarray_reduce_func_t func,
                                int8_t naxis,
                                const int8_t *axis,
                                iarray_storage_t *storage,
                                iarray_container_t **b);


#endif //IARRAY_IARRAY_REDUCE_PRIVATE_H

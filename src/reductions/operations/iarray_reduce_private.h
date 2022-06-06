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

typedef struct user_data_s {
    double inv_nelem;
    uint8_t input_itemsize;
    int64_t *not_nan_nelem;
} user_data_t;


#define CAST_I (void (*)(void *, void *))
#define CAST_R (void (*)(void *, int64_t, void *, int64_t, int64_t, void *))
#define CAST_F (void (*)(void *, void *))

#define DPARAMS_I double *res, void *user_data

#define DPARAMS_R double *data0, int64_t strides0, \
                  const double *data1, int64_t strides1, \
                  int64_t nelem, void *user_data

#define DPARAMS_F double *res, void *user_data

#define FPARAMS_I float *res, void *user_data
#define FPARAMS_R float *data0, int64_t strides0, \
                  const float *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
#define FPARAMS_F float *res, void *user_data

#define I64PARAMS_I int64_t *res, void *user_data
#define I64PARAMS_R int64_t *data0, int64_t strides0, \
                  const int64_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as double
#define I64_DPARAMS_R double *data0, int64_t strides0, \
                  const int64_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
#define I64PARAMS_F int64_t *res, void *user_data

#define I32PARAMS_I int32_t *res, void *user_data
#define I32PARAMS_R int32_t *data0, int64_t strides0, \
                  const int32_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as int64_t
#define I32_64PARAMS_R int64_t *data0, int64_t strides0, \
                  const int32_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as double
#define I32_DPARAMS_R double *data0, int64_t strides0, \
                  const int32_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
#define I32PARAMS_F int32_t *res, void *user_data

#define I16PARAMS_I int16_t *res, void *user_data
#define I16PARAMS_R int16_t *data0, int64_t strides0, \
                  const int16_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as int64_t
#define I16_64PARAMS_R int64_t *data0, int64_t strides0, \
                  const int16_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as double
#define I16_DPARAMS_R double *data0, int64_t strides0, \
                  const int16_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
#define I16PARAMS_F int16_t *res, void *user_data

#define I8PARAMS_I int8_t *res, void *user_data
#define I8PARAMS_R int8_t *data0, int64_t strides0, \
                  const int8_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as int64_t
#define I8_64PARAMS_R int64_t *data0, int64_t strides0, \
                  const int8_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as double
#define I8_DPARAMS_R double *data0, int64_t strides0, \
                  const int8_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
#define I8PARAMS_F int8_t *res, void *user_data

#define UI64PARAMS_I uint64_t *res, void *user_data
#define UI64PARAMS_R uint64_t *data0, int64_t strides0, \
                  const uint64_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as double
#define UI64_DPARAMS_R double *data0, int64_t strides0, \
                  const uint64_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
#define UI64PARAMS_F uint64_t *res, void *user_data

#define UI32PARAMS_I uint32_t *res, void *user_data
#define UI32PARAMS_R uint32_t *data0, int64_t strides0, \
                  const uint32_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as uint64_t
#define UI32_64PARAMS_R uint64_t *data0, int64_t strides0, \
                  const uint32_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as double
#define UI32_DPARAMS_R double *data0, int64_t strides0, \
                  const uint32_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
#define UI32PARAMS_F uint32_t *res, void *user_data

#define UI16PARAMS_I uint16_t *res, void *user_data
#define UI16PARAMS_R uint16_t *data0, int64_t strides0, \
                  const uint16_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as uint64_t
#define UI16_64PARAMS_R uint64_t *data0, int64_t strides0, \
                  const uint16_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as double
#define UI16_DPARAMS_R double *data0, int64_t strides0, \
                  const uint16_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
#define UI16PARAMS_F uint16_t *res, void *user_data

#define UI8PARAMS_I uint8_t *res, void *user_data
#define UI8PARAMS_R uint8_t *data0, int64_t strides0, \
                  const uint8_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as uint64_t
#define UI8_64PARAMS_R uint64_t *data0, int64_t strides0, \
                  const uint8_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as double
#define UI8_DPARAMS_R double *data0, int64_t strides0, \
                  const uint8_t *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
#define UI8PARAMS_F uint8_t *res, void *user_data

#define BOOLPARAMS_I bool *res, void *user_data
#define BOOLPARAMS_R bool *data0, int64_t strides0, \
                  const bool *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as int64_t
#define BOOL_64PARAMS_R int64_t *data0, int64_t strides0, \
                  const bool *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
// Needed when we want the result as double
#define BOOL_DPARAMS_R double *data0, int64_t strides0, \
                  const bool *data1, int64_t strides1, \
                  int64_t nelem, void *user_data
#define BOOLPARAMS_F bool *res, void *user_data

#endif //IARRAY_IARRAY_REDUCE_PRIVATE_H

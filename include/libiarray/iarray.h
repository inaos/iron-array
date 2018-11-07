/*
 * Copyright INAOS GmbH, Thalwil, 2018.
 * Copyright Francesc Alted, 2018.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of INAOS GmbH
 * and Francesc Alted ("Confidential Information"). You shall not disclose such Confidential
 * Information and shall use it only in accordance with the terms of the license agreement.
 *
 */
#ifndef PROJECT_IARRAY_H
#define PROJECT_IARRAY_H

#include <blosc.h>
#include <caterva.h>
#include <libinac/lib.h>

typedef struct iarray_context_s iarray_context_t;

typedef struct iarray_container_s iarray_container_t;

typedef struct iarray_expression_s iarray_expression_t;

typedef struct iarray_config_s {
    int flags;
    int max_num_threads; /* Maximum number of threads to use */
    blosc2_cparams *cparams;
    blosc2_dparams *dparams;
    caterva_pparams *pparams;
} iarray_config_t;

typedef enum iarray_rng_e {
    IARRAY_RNG_MERSENNE_TWISTER,
    IARRAY_RNG_SOBOL,
} iarray_rng_t;

typedef enum iarray_data_type_e {
    IARRAY_DATA_TYPE_DOUBLE,
    IARRAY_DATA_TYPE_FLOAT
} iarray_data_type_t;

typedef struct iarray_dtshape_s {
    iarray_data_type_t dtype;
    int ndim;     /* IF ndim = 0 THEN it is a scalar */
    int dims[8];  // a fixed size simplifies the code and should be enough for most IronArray cases
} iarray_dtshape_t;

typedef struct iarray_slice_param_s {
    int axis;
    int idx;
} iarray_slice_param_t;

typedef enum iarray_config_flags_e {
    IARRAY_EXPR_EVAL_BLOCK = 0x1,
    IARRAY_EXPR_EVAL_CHUNK = 0x2
} iarray_config_flags_t;

typedef enum iarray_bind_flags_e {
    IARRAY_BIND_UPDATE_CONTAINER = 0x1
} iarray_bind_flags_t;

INA_API(ina_rc_t) iarray_ctx_new(iarray_config_t *cfg, iarray_context_t **ctx);
INA_API(void) iarray_ctx_free(iarray_context_t **ctx);

INA_API(ina_rc_t) iarray_arange(iarray_context_t *ctx, iarray_dtshape_t *dtshape, int start, int stop, int step, iarray_data_type_t dtype, iarray_container_t **container);
INA_API(ina_rc_t) iarray_zeros(iarray_context_t *ctx, iarray_dtshape_t *dtshape, iarray_data_type_t dtype, iarray_container_t **container);
INA_API(ina_rc_t) iarray_ones(iarray_context_t *ctx, iarray_dtshape_t *dtshape, iarray_data_type_t dtype, iarray_container_t **container);
INA_API(ina_rc_t) iarray_fill_float(iarray_context_t *ctx, iarray_dtshape_t *dtshape, float value, iarray_container_t **container);
INA_API(ina_rc_t) iarray_fill_double(iarray_context_t *ctx, iarray_dtshape_t *dtshape, double value, iarray_container_t **container);
INA_API(ina_rc_t) iarray_rand(iarray_context_t *ctx, iarray_dtshape_t *dtshape, iarray_rng_t rng, iarray_data_type_t dtype, iarray_container_t **container);
INA_API(ina_rc_t) iarray_slice(iarray_context_t *ctx, iarray_container_t *c, iarray_slice_param_t *params, iarray_container_t **container);

INA_API(ina_rc_t) iarray_expr_new(iarray_context_t *ctx, iarray_expression_t **e);
INA_API(void) iarray_expr_free(iarray_context_t *ctx, iarray_expression_t **e);
INA_API(ina_rc_t) iarray_expr_bind(iarray_expression_t *e, const char *var, iarray_container_t *val);
INA_API(ina_rc_t) iarray_expr_bind_scalar_float(iarray_expression_t *e, const char *var, float val);
INA_API(ina_rc_t) iarray_expr_bind_scalar_double(iarray_expression_t *e, const char *var, double val);

INA_API(ina_rc_t) iarray_expr_compile(iarray_expression_t *e, const char *expr);

INA_API(ina_rc_t) iarray_eval(iarray_expression_t *e, iarray_container_t *ret); /* e.g. IARRAY_BIND_UPDATE_CONTAINER */

INA_API(ina_rc_t) iarray_from_ctarray(iarray_context_t *ctx, caterva_array *ctarray, iarray_data_type_t dtype, iarray_container_t **container);

//FIXME: remove
INA_API(ina_rc_t) iarray_from_sc(iarray_context_t *ctx, blosc2_schunk *sc, iarray_data_type_t dtype, iarray_container_t **container);
INA_API(ina_rc_t) iarray_expr_get_mp(iarray_expression_t *e, ina_mempool_t **mp);

//FIXME: Move to private header

typedef struct iarray_variable_s {
    const char *name;
    const void *address;
    iarray_dtshape_t dtshape;
    void *context;
} iarray_variable_t;

ina_rc_t iarray_eval_chunk(iarray_context_t *ctx, char* expr, iarray_variable_t *vars, int vars_count, iarray_variable_t out, iarray_data_type_t dtype, int *err);
ina_rc_t iarray_eval_block(iarray_context_t *ctx, char* expr, iarray_variable_t *vars, int vars_count, iarray_variable_t out, iarray_data_type_t dtype, int *err);

INA_API(ina_rc_t) iarray_equal_data(iarray_container_t *a, iarray_container_t *b);
INA_API(ina_rc_t) iarray_gemm(iarray_container_t *a, iarray_container_t *b, iarray_container_t *c);
INA_API(ina_rc_t) iarray_gemv(iarray_container_t *a, iarray_container_t *b, iarray_container_t *c);

#endif //PROJECT_IARRAY_H

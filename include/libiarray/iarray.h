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
#ifndef _IARRAY_H_
#define _IARRAY_H_

#include <libinac/lib.h>

#define IARRAY_DIMENSION_MAX 8 /* A fixed size simplifies the code and should be enough for most IronArray cases */

typedef struct iarray_context_s iarray_context_t;

typedef struct iarray_container_s iarray_container_t;

typedef struct iarray_expression_s iarray_expression_t;

typedef enum iarray_rng_e {
    IARRAY_RNG_MERSENNE_TWISTER,
    IARRAY_RNG_SOBOL,
} iarray_rng_t;

typedef enum iarray_data_type_e {
    IARRAY_DATA_TYPE_DOUBLE,
    IARRAY_DATA_TYPE_FLOAT
} iarray_data_type_t;

typedef enum iarray_storage_format_e {
    IARRAY_STORAGE_ROW_WISE = 0,
    IARRAY_STORAGE_COL_WISE
} iarray_storage_format_t;

typedef enum iarray_config_flags_e {
    IARRAY_EXPR_EVAL_BLOCK = 0x1,
    IARRAY_EXPR_EVAL_CHUNK = 0x2,
    IARRAY_COMP_SHUFFLE    = 0x4,
    IARRAY_COMP_BITSHUFFLE = 0x8,
    IARRAY_COMP_DELTA      = 0x10,
    IARRAY_COMP_TRUNC_PREC = 0x20,
} iarray_config_flags_t;

typedef enum iarray_bind_flags_e {
    IARRAY_BIND_UPDATE_CONTAINER = 0x1
} iarray_bind_flags_t;

typedef enum iarray_container_flags_e {
    IARRAY_CONTAINER_PERSIST = 0x1
} iarray_container_flags_t;

typedef enum iarray_compression_codec_e {
    IARRAY_COMPRESSION_DEFAULT = 0,
    IARRAY_COMPRESSION_LZ4,
    IARRAY_COMPRESSION_LZ4HC,
    IARRAY_COMPRESSION_SNAPPY,
    IARRAY_COMPRESSION_ZLIB,
    IARRAY_COMPRESSION_ZSTD,
    IARRAY_COMPRESSION_LIZARD
} iarray_compression_codec_t;

typedef struct iarray_config_s {
    iarray_compression_codec_t compression_codec;
    int compression_level;
    int flags;
    int max_num_threads; /* Maximum number of threads to use */
    int fp_mantissa_bits; /* Only useful together with flag: IARRAY_COMP_TRUNC_PREC */
} iarray_config_t;

typedef struct iarray_dtshape_s {
    iarray_data_type_t dtype;
    int ndim;     /* IF ndim = 0 THEN it is a scalar */
    int dims[IARRAY_DIMENSION_MAX];
    int partshape[IARRAY_DIMENSION_MAX]; /* Partition-Shape, optional in the future */
} iarray_dtshape_t;

typedef struct iarray_slice_param_s {
    int axis;
    int idx;
} iarray_slice_param_t;

INA_API(ina_rc_t) iarray_init();
INA_API(void) iarray_destroy();

INA_API(ina_rc_t) iarray_context_new(iarray_config_t *cfg, iarray_context_t **ctx);
INA_API(void) iarray_context_free(iarray_context_t **ctx);

INA_API(ina_rc_t) iarray_container_new(iarray_context_t *ctx,
                                       iarray_dtshape_t *dtshape,
                                       const char *name,
                                       int flags,
                                       iarray_container_t **container);

INA_API(ina_rc_t) iarray_arange(iarray_context_t *ctx, 
                                iarray_dtshape_t *dtshape, 
                                int start, 
                                int stop, 
                                int step, 
                                const char *name,
                                int flags,
                                iarray_container_t **container);

INA_API(ina_rc_t) iarray_zeros(iarray_context_t *ctx, 
                               iarray_dtshape_t *dtshape, 
                               const char *name,
                               int flags,
                               iarray_container_t **container);

INA_API(ina_rc_t) iarray_ones(iarray_context_t *ctx, 
                              iarray_dtshape_t *dtshape, 
                              const char *name,
                              int flags,
                              iarray_container_t **container);

INA_API(ina_rc_t) iarray_fill_float(iarray_context_t *ctx, 
                                    iarray_dtshape_t *dtshape, 
                                    float value, 
                                    const char *name,
                                    int flags,
                                    iarray_container_t **container);

INA_API(ina_rc_t) iarray_fill_double(iarray_context_t *ctx, 
                                     iarray_dtshape_t *dtshape, 
                                     double value, 
                                     const char *name,
                                     int flags,
                                     iarray_container_t **container);

INA_API(ina_rc_t) iarray_rand(iarray_context_t *ctx, 
                              iarray_dtshape_t *dtshape, 
                              iarray_rng_t rng, 
                              const char *name,
                              int flags,
                              iarray_container_t **container);

INA_API(ina_rc_t) iarray_slice(iarray_context_t *ctx, 
                               iarray_container_t *c, 
                               iarray_slice_param_t *params, 
                               iarray_container_t **container);

INA_API(ina_rc_t) iarray_from_buffer(iarray_context_t *ctx,
                                     iarray_dtshape_t *dtshape,
                                     const void *buffer,
                                     size_t buffer_len,
                                     iarray_storage_format_t fmt,
                                     const char *name,
                                     int flags,
                                     iarray_container_t **container);

INA_API(ina_rc_t) iarray_to_buffer(iarray_context_t *ctx,
                                   iarray_container_t *container,
                                   void *buffer,
                                   size_t buffer_len,
                                   iarray_storage_format_t fmt);


INA_API(ina_rc_t) iarray_container_info(iarray_container_t *c, 
                                        size_t *size_in_bytes, 
                                        size_t *compressed_size_in_bytes);

INA_API(void) iarray_container_free(iarray_context_t *ctx, iarray_container_t **container);

INA_API(ina_rc_t) iarray_expr_new(iarray_context_t *ctx, iarray_expression_t **e);
INA_API(void) iarray_expr_free(iarray_context_t *ctx, iarray_expression_t **e);
INA_API(ina_rc_t) iarray_expr_bind(iarray_expression_t *e, const char *var, iarray_container_t *val);
INA_API(ina_rc_t) iarray_expr_bind_scalar_float(iarray_expression_t *e, const char *var, float val);
INA_API(ina_rc_t) iarray_expr_bind_scalar_double(iarray_expression_t *e, const char *var, double val);

INA_API(ina_rc_t) iarray_expr_compile(iarray_expression_t *e, const char *expr);

INA_API(ina_rc_t) iarray_eval(iarray_expression_t *e, iarray_container_t *ret); /* e.g. IARRAY_BIND_UPDATE_CONTAINER */

INA_API(ina_rc_t) iarray_equal_data(iarray_container_t *a, iarray_container_t *b);

INA_API(ina_rc_t) iarray_gemm(iarray_container_t *a, iarray_container_t *b, iarray_container_t *c);
INA_API(ina_rc_t) iarray_gemv(iarray_container_t *a, iarray_container_t *b, iarray_container_t *c);

//FIXME: remove
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

#endif

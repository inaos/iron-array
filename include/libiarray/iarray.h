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
#include <stdbool.h>


#define IARRAY_DIMENSION_MAX 8 /* A fixed size simplifies the code and should be enough for most IronArray cases */

#define IARRAY_ES_CONTAINER (INA_ES_USER_DEFINED + 1)
#define IARRAY_ES_DTSHAPE (INA_ES_USER_DEFINED + 2)
#define IARRAY_ES_SHAPE (INA_ES_USER_DEFINED + 3)
#define IARRAY_ES_PSHAPE (INA_ES_USER_DEFINED + 4)
#define IARRAY_ES_NDIM (INA_ES_USER_DEFINED + 5)
#define IARRAY_ES_DTYPE (INA_ES_USER_DEFINED + 6)
#define IARRAY_ES_STORAGE (INA_ES_USER_DEFINED + 7)
#define IARRAY_ES_PERSISTENCY (INA_ES_USER_DEFINED + 8)
#define IARRAY_ES_BUFFER (INA_ES_USER_DEFINED + 9)
#define IARRAY_ES_CATERVA (INA_ES_USER_DEFINED + 10)
#define IARRAY_ES_BLOSC (INA_ES_USER_DEFINED + 11)
#define IARRAY_ES_ASSERTION (INA_ES_USER_DEFINED + 12)
#define IARRAY_ES_BSHAPE (INA_ES_USER_DEFINED + 13)
#define IARRAY_ES_RNG_METHOD (INA_ES_USER_DEFINED + 14)
#define IARRAY_ES_RAND_METHOD (INA_ES_USER_DEFINED + 15)
#define IARRAY_ES_RAND_PARAM (INA_ES_USER_DEFINED + 16)
#define IARRAY_ES_ITER (INA_ES_USER_DEFINED + 16)
#define IARRAY_ES_MALLOC (INA_ES_USER_DEFINED + 17)

#define IARRAY_ERR_EMPTY_CONTAINER (INA_ERR_EMPTY | IARRAY_ES_CONTAINER)
#define IARRAY_ERR_FULL_CONTAINER (INA_ERR_FULL | IARRAY_ES_CONTAINER)

#define IARRAY_ERR_INVALID_DTSHAPE (INA_ERR_INVALID | IARRAY_ES_DTSHAPE)

#define IARRAY_ERR_INVALID_DTYPE (INA_ERR_INVALID | IARRAY_ES_DTYPE)
#define IARRAY_ERR_INVALID_SHAPE (INA_ERR_INVALID | IARRAY_ES_SHAPE)
#define IARRAY_ERR_INVALID_PSHAPE (INA_ERR_INVALID | IARRAY_ES_PSHAPE)
#define IARRAY_ERR_INVALID_BSHAPE (INA_ERR_INVALID | IARRAY_ES_BSHAPE)
#define IARRAY_ERR_INVALID_NDIM (INA_ERR_INVALID | IARRAY_ES_NDIM)

#define IARRAY_ERR_INVALID_RNG_METHOD (IARRAY_ES_RNG_METHOD | INA_ERR_INVALID)
#define IARRAY_ERR_INVALID_RAND_METHOD (IARRAY_ES_RAND_METHOD | INA_ERR_INVALID)
#define IARRAY_ERR_INVALID_RAND_PARAM (IARRAY_ES_RAND_PARAM | INA_ERR_INVALID)

#define IARRAY_ERR_INVALID_STORAGE (INA_ERR_INVALID | IARRAY_ES_STORAGE)
#define IARRAY_ERR_INVALID_PERSISTENCY (INA_ERR_INVALID | IARRAY_ES_PERSISTENCY)
#define IARRAY_ERR_TOO_SMALL_BUFFER (INA_ERR_TOO_SMALL | IARRAY_ES_BUFFER)

#define IARRAY_ERR_CATERVA_FAILED (INA_ERR_FAILED | IARRAY_ES_CATERVA)
#define IARRAY_ERR_BLOSC_FAILED (INA_ERR_FAILED | IARRAY_ES_BLOSC)
#define IARRAY_ERR_RAND_METHOD_FAILED (IARRAY_ES_RAND_METHOD | INA_ERR_FAILED)
#define IARRAY_ERR_ASSERTION_FAILED (IARRAY_ES_ASSERTION | INA_ERR_FAILED)
#define IARRAY_ERR_MALLOC_FAILED (INA_ERR_FAILED | IARRAY_ES_MALLOC)

#define IARRAY_ERR_END_ITER (IARRAY_ES_ITER | INA_ERR_COMPLETE)


typedef struct iarray_context_s iarray_context_t;
typedef struct iarray_container_s iarray_container_t;

typedef struct iarray_iter_write_s iarray_iter_write_t;

typedef struct iarray_iter_read_s iarray_iter_read_t;
typedef struct iarray_iter_read_block_s iarray_iter_read_block_t;
typedef struct iarray_iter_write_block_s iarray_iter_write_block_t;

typedef struct iarray_expression_s iarray_expression_t;

typedef enum iarray_random_rng_e {
    IARRAY_RANDOM_RNG_MERSENNE_TWISTER,
    IARRAY_RANDOM_RNG_SOBOL,
} iarray_random_rng_t;

typedef enum iarray_random_dist_parameter_e {
    IARRAY_RANDOM_DIST_PARAM_MU,
    IARRAY_RANDOM_DIST_PARAM_SIGMA,
    IARRAY_RANDOM_DIST_PARAM_ALPHA,
    IARRAY_RANDOM_DIST_PARAM_BETA,
    IARRAY_RANDOM_DIST_PARAM_LAMBDA,
    IARRAY_RANDOM_DIST_PARAM_A,
    IARRAY_RANDOM_DIST_PARAM_B,
    IARRAY_RANDOM_DIST_PARAM_P,
    IARRAY_RANDOM_DIST_PARAM_M,
    IARRAY_RANDOM_DIST_PARAM_SENTINEL /* marks end of list */
} iarray_random_dist_parameter_t;

typedef enum iarray_data_type_e {
    IARRAY_DATA_TYPE_DOUBLE,
    IARRAY_DATA_TYPE_FLOAT,
    IARRAY_DATA_TYPE_MAX  // marker; must be the last entry
} iarray_data_type_t;

typedef enum iarray_storage_format_e {
    IARRAY_STORAGE_ROW_WISE = 0,
    IARRAY_STORAGE_COL_WISE
} iarray_storage_format_t;

typedef struct iarray_store_properties_s {
    const char *id;
} iarray_store_properties_t;

typedef enum iarray_eval_flags_e {
    IARRAY_EXPR_EVAL_ITERCHUNK = 1,
    IARRAY_EXPR_EVAL_ITERBLOCK = 2,
    IARRAY_EXPR_EVAL_ITERBLOSC = 3,
} iarray_eval_flags_t;

typedef enum iarray_filter_flags_e {
    IARRAY_COMP_SHUFFLE    = 0x1,
    IARRAY_COMP_BITSHUFFLE = 0x2,
    IARRAY_COMP_DELTA      = 0x4,
    IARRAY_COMP_TRUNC_PREC = 0x8,
} iarray_filter_flags_t;

typedef enum iarray_bind_flags_e {
    IARRAY_BIND_UPDATE_CONTAINER = 0x1
} iarray_bind_flags_t;

typedef enum iarray_container_flags_e {
    IARRAY_CONTAINER_PERSIST = 0x1
} iarray_container_flags_t;

typedef enum iarray_operator_hint_e {
    IARRAY_OPERATOR_GENERAL = 0,
    IARRAY_OPERATOR_SYMMETRIC,
    IARRAY_OPERATOR_TRIANGULAR
} iarray_operator_hint_t;

typedef enum iarray_compression_codec_e {
    IARRAY_COMPRESSION_BLOSCLZ = 0,
    IARRAY_COMPRESSION_LZ4,
    IARRAY_COMPRESSION_LZ4HC,
    IARRAY_COMPRESSION_SNAPPY,
    IARRAY_COMPRESSION_ZLIB,
    IARRAY_COMPRESSION_ZSTD,
    IARRAY_COMPRESSION_LIZARD
} iarray_compression_codec_t;

typedef enum iarray_linalg_norm_e {
    IARRAY_LINALG_NORM_NONE,
    IARRAY_LINALG_NORM_FROBENIUS,
    IARRAY_LINALG_NORM_NUCLEAR,
    IARRAY_LINALG_NORM_MAX_ROWS,
    IARRAY_LINALG_NORM_MAX_COLS,
    IARRAY_LINALG_NORM_MIN_ROWS,
    IARRAY_LINALG_NORM_MIN_COLS,
    IARRAY_LINALG_NORM_SING_MAX,
    IARRAY_LINALG_NORM_SING_MIN
} iarray_linalg_norm_t;

typedef struct iarray_config_s {
    iarray_compression_codec_t compression_codec;
    int compression_level;
    int use_dict;
    int filter_flags;
    unsigned int eval_flags;
    int max_num_threads; /* Maximum number of threads to use */
    uint8_t fp_mantissa_bits; /* Only useful together with flag: IARRAY_COMP_TRUNC_PREC */
    int blocksize; /* Advanced Tuning Parameter */
} iarray_config_t;

typedef struct iarray_dtshape_s {
    iarray_data_type_t dtype;
    int8_t ndim;     /* if ndim = 0 it is a scalar */
    int64_t shape[IARRAY_DIMENSION_MAX];
    int64_t pshape[IARRAY_DIMENSION_MAX]; /* partition shape */
} iarray_dtshape_t;

typedef struct iarray_iter_write_value_s {
    void *elem_pointer;
    int64_t *elem_index;
    int64_t elem_flat_index;
} iarray_iter_write_value_t;


typedef struct iarray_iter_read_value_s {
    void *elem_pointer;
    int64_t *elem_index;
    int64_t elem_flat_index;
} iarray_iter_read_value_t;

typedef struct iarray_iter_write_block_value_s {
    void *block_pointer;
    int64_t *block_index;
    int64_t *elem_index;
    int64_t nblock;
    int64_t* block_shape;
    int64_t block_size;
} iarray_iter_write_block_value_t;

typedef struct iarray_iter_read_block_value_s {
    void *block_pointer;
    int64_t *block_index;
    int64_t *elem_index;
    int64_t nblock;
    int64_t* block_shape;
    int64_t block_size;
} iarray_iter_read_block_value_t;

typedef struct iarray_slice_param_s {
    int axis;
    int idx;
} iarray_slice_param_t;

typedef struct iarray_random_ctx_s iarray_random_ctx_t;

static const iarray_config_t IARRAY_CONFIG_DEFAULTS = {
    .compression_codec=IARRAY_COMPRESSION_LZ4,
    .compression_level=5,
    .use_dict=0,
    .filter_flags=IARRAY_COMP_SHUFFLE,
    .eval_flags=IARRAY_EXPR_EVAL_ITERCHUNK,
    .max_num_threads=1,
    .fp_mantissa_bits=0,
    .blocksize=0 };

static const iarray_config_t IARRAY_CONFIG_NO_COMPRESSION = {
    .compression_codec=IARRAY_COMPRESSION_LZ4,
    .compression_level=0,
    .use_dict=0,
    .filter_flags=0,
    .eval_flags=0,
    .max_num_threads=1,
    .fp_mantissa_bits=0,
    .blocksize=0 };

INA_API(ina_rc_t) iarray_init(void);
INA_API(void) iarray_destroy(void);

INA_API(ina_rc_t) iarray_context_new(iarray_config_t *cfg, iarray_context_t **ctx);
INA_API(void) iarray_context_free(iarray_context_t **ctx);

/*
 *  Provide advice for the partition shape of a `dtshape`.
 *
 *  If success, dtshape.pshape will contain the advice.
 *
 *  `low` and `high` contain low and high values for the partition size.  If `low` is 0, it defaults
 *  to a fraction of L2 cache size.  If `high` is 0, it defaults to a fraction of L3 cache size.
 */
INA_API(ina_rc_t) iarray_partition_advice(iarray_context_t *ctx, iarray_dtshape_t *dtshape,
                                          int64_t low, int64_t high);

/*
 * Provide advice for the block shapes for performing a matrix-matrix multiplication.
 *
 * `a` and `b` are supposed to have (M, K) and (K, N) dimensions respectively
 * `c` is supposed to have a partition size of (m, n)
 * The hint for the block shapes are going to be (m, k) and (k, n) respectively
 *
 * The hints will be stored in `bshape_a` and `bshape_b`, which needs to be provided by the user.
 * The number of components for the block shapes is 2.
 *
 *  `low` and `high` contain low and high values for the partition size.  If `low` is 0, it defaults
 *  to a fraction of L2 cache size.  If `high` is 0, it defaults to a fraction of L3 cache size.
 *
 * Note: When performing matrix-*vector* operations, just pass the N dimension as 1.  The `k` hint
 * will be valid for this case too.  In this case, always pass `bshape_a` and `bshape_b` with
 * 2-components too (even if `bshape_b` only has a dimension in this case).
 *
 */
INA_API(ina_rc_t) iarray_matmul_advice(iarray_context_t *ctx,
                                       iarray_container_t *a, iarray_container_t *b, iarray_container_t *c,
                                       int64_t *bshape_a, int64_t *bshape_b,
                                       int64_t low, int64_t high);

INA_API(ina_rc_t) iarray_random_ctx_new(iarray_context_t *ctx,
                                        uint32_t seed,
                                        iarray_random_rng_t rng,
                                        iarray_random_ctx_t **rng_ctx);

INA_API(void) iarray_random_ctx_free(iarray_context_t *ctx, iarray_random_ctx_t **rng_ctx);

INA_API(ina_rc_t) iarray_random_dist_set_param_float(iarray_random_ctx_t *ctx,
                                                     iarray_random_dist_parameter_t key,
                                                     float value);

INA_API(ina_rc_t) iarray_random_dist_set_param_double(iarray_random_ctx_t *ctx,
                                                      iarray_random_dist_parameter_t key,
                                                      double value);

INA_API(ina_rc_t) iarray_container_new(iarray_context_t *ctx,
                                       iarray_dtshape_t *dtshape,
                                       iarray_store_properties_t *store,
                                       int flags,
                                       iarray_container_t **container);

INA_API(ina_rc_t) iarray_arange(iarray_context_t *ctx,
                                iarray_dtshape_t *dtshape,
                                double start,
                                double stop,
                                double step,
                                iarray_store_properties_t *store,
                                int flags,
                                iarray_container_t **container);

INA_API(ina_rc_t) iarray_linspace(iarray_context_t *ctx,
                                  iarray_dtshape_t *dtshape,
                                  int64_t nelem,
                                  double start,
                                  double stop,
                                  iarray_store_properties_t *store,
                                  int flags,
                                  iarray_container_t **container);

INA_API(ina_rc_t) iarray_zeros(iarray_context_t *ctx,
                               iarray_dtshape_t *dtshape,
                               iarray_store_properties_t *store,
                               int flags,
                               iarray_container_t **container);

INA_API(ina_rc_t) iarray_ones(iarray_context_t *ctx,
                              iarray_dtshape_t *dtshape,
                              iarray_store_properties_t *store,
                              int flags,
                              iarray_container_t **container);

INA_API(ina_rc_t) iarray_fill_float(iarray_context_t *ctx,
                                    iarray_dtshape_t *dtshape,
                                    float value,
                                    iarray_store_properties_t *store,
                                    int flags,
                                    iarray_container_t **container);

INA_API(ina_rc_t) iarray_fill_double(iarray_context_t *ctx,
                                     iarray_dtshape_t *dtshape,
                                     double value,
                                     iarray_store_properties_t *store,
                                     int flags,
                                     iarray_container_t **container);

INA_API(ina_rc_t) iarray_random_rand(iarray_context_t *ctx,
                                     iarray_dtshape_t *dtshape,
                                     iarray_random_ctx_t *rand_ctx,
                                     iarray_store_properties_t *store,
                                     int flags,
                                     iarray_container_t **container);

INA_API(ina_rc_t) iarray_random_randn(iarray_context_t *ctx,
                                      iarray_dtshape_t *dtshape,
                                      iarray_random_ctx_t *rand_ctx,
                                      iarray_store_properties_t *store,
                                      int flags,
                                      iarray_container_t **container);

INA_API(ina_rc_t) iarray_random_beta(iarray_context_t *ctx,
                                     iarray_dtshape_t *dtshape,
                                     iarray_random_ctx_t *rand_ctx,
                                     iarray_store_properties_t *store,
                                     int flags,
                                     iarray_container_t **container);

INA_API(ina_rc_t) iarray_random_lognormal(iarray_context_t *ctx,
                                          iarray_dtshape_t *dtshape,
                                          iarray_random_ctx_t *rand_ctx,
                                          iarray_store_properties_t *store,
                                          int flags,
                                          iarray_container_t **container);

INA_API(ina_rc_t) iarray_random_exponential(iarray_context_t *ctx,
                                            iarray_dtshape_t *dtshape,
                                            iarray_random_ctx_t *random_ctx,
                                            iarray_store_properties_t *store,
                                            int flags,
                                            iarray_container_t **container);

INA_API(ina_rc_t) iarray_random_uniform(iarray_context_t *ctx,
                                            iarray_dtshape_t *dtshape,
                                            iarray_random_ctx_t *random_ctx,
                                            iarray_store_properties_t *store,
                                            int flags,
                                            iarray_container_t **container);

INA_API(ina_rc_t) iarray_random_normal(iarray_context_t *ctx,
                                       iarray_dtshape_t *dtshape,
                                       iarray_random_ctx_t *random_ctx,
                                       iarray_store_properties_t *store,
                                       int flags,
                                       iarray_container_t **container);

INA_API(ina_rc_t) iarray_random_bernoulli(iarray_context_t *ctx,
                                          iarray_dtshape_t *dtshape,
                                          iarray_random_ctx_t *random_ctx,
                                          iarray_store_properties_t *store,
                                          int flags,
                                          iarray_container_t **container);

INA_API(ina_rc_t) iarray_random_binomial(iarray_context_t *ctx,
                                         iarray_dtshape_t *dtshape,
                                         iarray_random_ctx_t *random_ctx,
                                         iarray_store_properties_t *store,
                                         int flags,
                                         iarray_container_t **container);

INA_API(ina_rc_t) iarray_random_poisson(iarray_context_t *ctx,
                                        iarray_dtshape_t *dtshape,
                                        iarray_random_ctx_t *random_ctx,
                                        iarray_store_properties_t *store,
                                        int flags,
                                        iarray_container_t **container);

INA_API(ina_rc_t) iarray_random_kstest(iarray_context_t *ctx,
                                       iarray_container_t *c1,
                                       iarray_container_t *c2,
                                       bool *res);

INA_API(ina_rc_t) iarray_get_slice(iarray_context_t *ctx,
                                   iarray_container_t *c,
                                   int64_t *start,
                                   int64_t *stop,
                                   const int64_t *pshape,
                                   iarray_store_properties_t *store,
                                   int flags,
                                   bool view,
                                   iarray_container_t **container);


INA_API(ina_rc_t) iarray_set_slice(iarray_context_t *ctx,
                                   iarray_container_t *c,
                                   const int64_t *start,
                                   const int64_t *stop,
                                   iarray_container_t *slice);


INA_API(ina_rc_t) iarray_get_slice_buffer(iarray_context_t *ctx,
                                          iarray_container_t *c,
                                          const int64_t *start,
                                          const int64_t *stop,
                                          void *buffer,
                                          int64_t buflen);

INA_API(ina_rc_t) iarray_set_slice_buffer(iarray_context_t *ctx,
                                          iarray_container_t *c,
                                          const int64_t *start,
                                          const int64_t *stop,
                                          void *buffer,
                                          int64_t buflen);

INA_API(ina_rc_t) iarray_from_file(iarray_context_t *ctx,
                                   iarray_store_properties_t *store,
                                   iarray_container_t **container,
                                   bool load_in_mem);

INA_API(ina_rc_t) iarray_to_file(iarray_context_t *ctx,
                                 iarray_store_properties_t *store,
                                 iarray_container_t **container);

INA_API(ina_rc_t) iarray_squeeze(iarray_context_t *ctx,
                                 iarray_container_t *container);

INA_API(ina_rc_t) iarray_get_dtshape(iarray_context_t *ctx,
                                     iarray_container_t *c,
                                     iarray_dtshape_t *dtshape);

INA_API(ina_rc_t) iarray_get_sframe(iarray_container_t *container,
                                    uint8_t **sframe,
                                    int64_t *len,
                                    bool *shared);

INA_API(ina_rc_t) iarray_from_buffer(iarray_context_t *ctx,
                                     iarray_dtshape_t *dtshape,
                                     void *buffer,
                                     size_t buflen,
                                     iarray_store_properties_t *store,
                                     int flags,
                                     iarray_container_t **container);

INA_API(ina_rc_t) iarray_to_buffer(iarray_context_t *ctx,
                                   iarray_container_t *container,
                                   void *buffer,
                                   size_t buflen);

INA_API(bool) iarray_is_empty(iarray_container_t *container);

INA_API(ina_rc_t) iarray_container_dtshape_equal(iarray_dtshape_t *a, iarray_dtshape_t *b);
INA_API(ina_rc_t) iarray_container_info(iarray_container_t *c, int64_t *nbytes, int64_t *cbytes);

INA_API(void) iarray_container_free(iarray_context_t *ctx, iarray_container_t **container);

/* Comparison operators -> not supported yet as we only support float and double and return would be int8 */
INA_API(ina_rc_t) iarray_container_gt(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result);
INA_API(ina_rc_t) iarray_container_lt(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result);
INA_API(ina_rc_t) iarray_container_gte(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result);
INA_API(ina_rc_t) iarray_container_lte(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result);
INA_API(ina_rc_t) iarray_container_eq(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result);

INA_API(ina_rc_t) iarray_container_almost_equal(iarray_container_t *a, iarray_container_t *b, double tol);

INA_API(ina_rc_t) iarray_container_is_symmetric(iarray_container_t *a);
INA_API(ina_rc_t) iarray_container_is_triangular(iarray_container_t *a);

/* Logical operators -> not supported yet as we only support float and double and return would be int8 */
INA_API(ina_rc_t) iarray_operator_and(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_or(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_xor(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_nand(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_not(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result);

/* Arithmetic operators -> element-wise */
INA_API(ina_rc_t) iarray_operator_add(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_sub(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_mul(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_div(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result);

/* linear algebra */
INA_API(ina_rc_t) iarray_linalg_transpose(iarray_context_t *ctx, iarray_container_t *a);
INA_API(ina_rc_t) iarray_linalg_inverse(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_linalg_matmul(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result,
                                       int64_t *bshape_a, int64_t *bshape_b, iarray_operator_hint_t hint);
INA_API(ina_rc_t) iarray_linalg_dot(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result, iarray_operator_hint_t hint);
INA_API(ina_rc_t) iarray_linalg_det(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_linalg_eigen(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_linalg_norm(iarray_context_t *ctx, iarray_container_t *a, iarray_linalg_norm_t ord, iarray_container_t *result);
INA_API(ina_rc_t) iarray_linalg_solve(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result);
INA_API(ina_rc_t) iarray_linalg_lstsq(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result);
INA_API(ina_rc_t) iarray_linalg_svd(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_linalg_qr(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result); // Not clear to which MKL function we need to map
INA_API(ina_rc_t) iarray_linalg_lu(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result); // ?getrf (MKL) - Not clear to which MKL function we need to map
INA_API(ina_rc_t) iarray_linalg_cholesky(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);

/* Function operators -> element-wise */
INA_API(ina_rc_t) iarray_operator_abs(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_acos(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_asin(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_atanc(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_atan2(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_ceil(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_cos(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_cosh(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_exp(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_floor(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_log(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_log10(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_pow(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_sin(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_sinh(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_sqrt(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_tan(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_tanh(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_erf(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_erfc(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_cdfnorm(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_erfinv(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_erfcinv(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_cdfnorminv(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_lgamma(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_tgamma(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_expint1(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_operator_cumsum(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);

/* Reductions */
INA_API(ina_rc_t) iarray_reduction_sum(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_reduction_min(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_reduction_max(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
INA_API(ina_rc_t) iarray_reduction_mul(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);

/* Iterators */

INA_API(ina_rc_t) iarray_iter_write_new(iarray_context_t *ctx,
                                        iarray_iter_write_t **itr,
                                        iarray_container_t *cont,
                                        iarray_iter_write_value_t *val);
INA_API(void) iarray_iter_write_free(iarray_iter_write_t **itr);
INA_API(ina_rc_t) iarray_iter_write_next(iarray_iter_write_t *itr);
INA_API(ina_rc_t) iarray_iter_write_has_next(iarray_iter_write_t *itr);


INA_API(ina_rc_t) iarray_iter_read_new(iarray_context_t *ctx,
                                       iarray_iter_read_t **itr,
                                       iarray_container_t *cont,
                                       iarray_iter_read_value_t *val);
INA_API(void) iarray_iter_read_free(iarray_iter_read_t **itr);
INA_API(ina_rc_t) iarray_iter_read_next(iarray_iter_read_t *itr);
INA_API(ina_rc_t) iarray_iter_read_has_next(iarray_iter_read_t *itr);

INA_API(ina_rc_t) iarray_iter_read_block_new(iarray_context_t *ctx,
                                             iarray_iter_read_block_t **itr,
                                             iarray_container_t *cont,
                                             const int64_t *blockshape,
                                             iarray_iter_read_block_value_t *value,
                                             bool external_buffer);

INA_API(void) iarray_iter_read_block_free(iarray_iter_read_block_t **itr);
INA_API(ina_rc_t) iarray_iter_read_block_next(iarray_iter_read_block_t *itr, void *buffer, int32_t bufsize);
INA_API(ina_rc_t) iarray_iter_read_block_has_next(iarray_iter_read_block_t *itr);

INA_API(ina_rc_t) iarray_iter_write_block_new(iarray_context_t *ctx,
                                              iarray_iter_write_block_t **itr,
                                              iarray_container_t *cont,
                                              const int64_t *blockshape,
                                              iarray_iter_write_block_value_t *value,
                                              bool external_buffer);

INA_API(void) iarray_iter_write_block_free(iarray_iter_write_block_t **itr);
INA_API(ina_rc_t) iarray_iter_write_block_next(iarray_iter_write_block_t *itr, void *buffer, int32_t bufsize);
INA_API(ina_rc_t) iarray_iter_write_block_has_next(iarray_iter_write_block_t *itr);

/* Expressions */
INA_API(ina_rc_t) iarray_expr_new(iarray_context_t *ctx, iarray_expression_t **e);
INA_API(void) iarray_expr_free(iarray_context_t *ctx, iarray_expression_t **e);

INA_API(ina_rc_t) iarray_expr_bind(iarray_expression_t *e, const char *var, iarray_container_t *val);
INA_API(ina_rc_t) iarray_expr_bind_scalar_float(iarray_expression_t *e, const char *var, float val);
INA_API(ina_rc_t) iarray_expr_bind_scalar_double(iarray_expression_t *e, const char *var, double val);
INA_API(ina_rc_t) iarray_expr_compile(iarray_expression_t *e, const char *expr);

INA_API(ina_rc_t) iarray_eval(iarray_expression_t *e, iarray_container_t *ret); /* e.g. IARRAY_BIND_UPDATE_CONTAINER */

//FIXME: remove
INA_API(ina_rc_t) iarray_expr_get_mp(iarray_expression_t *e, ina_mempool_t **mp);
INA_API(ina_rc_t) iarray_expr_get_nthreads(iarray_expression_t *e, int *nthreads);

#endif

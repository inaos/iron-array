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

#ifndef _IARRAY_H_
#define _IARRAY_H_

#include <libinac/lib.h>
#include <stdbool.h>

#define IARRAY_METALAYER_VERSION 0

#define IARRAY_DIMENSION_MAX 8  /* A fixed size simplifies the code and should be enough for most IronArray cases */

#define IARRAY_EXPR_OPERANDS_MAX (128)
// The maximum number of input arrays in expressions

#define IARRAY_EXPR_USER_PARAMS_MAX (128)
// The maximum number of input user parameters in expressions

#define IARRAY_ES_CONTAINER (INA_ES_USER_DEFINED + 1)
#define IARRAY_ES_DTSHAPE (INA_ES_USER_DEFINED + 2)
#define IARRAY_ES_SHAPE (INA_ES_USER_DEFINED + 3)
#define IARRAY_ES_CHUNKSHAPE (INA_ES_USER_DEFINED + 4)
#define IARRAY_ES_NDIM (INA_ES_USER_DEFINED + 5)
#define IARRAY_ES_DTYPE (INA_ES_USER_DEFINED + 6)
#define IARRAY_ES_STORAGE (INA_ES_USER_DEFINED + 7)
#define IARRAY_ES_PERSISTENCY (INA_ES_USER_DEFINED + 8)
#define IARRAY_ES_BUFFER (INA_ES_USER_DEFINED + 9)
#define IARRAY_ES_CATERVA (INA_ES_USER_DEFINED + 10)
#define IARRAY_ES_BLOSC (INA_ES_USER_DEFINED + 11)
#define IARRAY_ES_ASSERTION (INA_ES_USER_DEFINED + 12)
#define IARRAY_ES_BLOCKSHAPE (INA_ES_USER_DEFINED + 13)
#define IARRAY_ES_RNG_METHOD (INA_ES_USER_DEFINED + 14)
#define IARRAY_ES_RAND_METHOD (INA_ES_USER_DEFINED + 15)
#define IARRAY_ES_RAND_PARAM (INA_ES_USER_DEFINED + 16)
#define IARRAY_ES_ITER (INA_ES_USER_DEFINED + 17)
#define IARRAY_ES_EVAL_METHOD (INA_ES_USER_DEFINED + 18)
#define IARRAY_ES_EVAL_ENGINE (INA_ES_USER_DEFINED + 19)
#define IARRAY_ES_ITERSHAPE (INA_ES_USER_DEFINED + 20)
#define IARRAY_ES_NCORES (INA_ES_USER_DEFINED + 21)
#define IARRAY_ES_CACHE_SIZES (INA_ES_USER_DEFINED + 22)
#define IARRAY_ES_AXIS (INA_ES_USER_DEFINED + 23)


#define IARRAY_ERR_EMPTY_CONTAINER (INA_ERR_EMPTY | IARRAY_ES_CONTAINER)
#define IARRAY_ERR_FULL_CONTAINER (INA_ERR_FULL | IARRAY_ES_CONTAINER)

#define IARRAY_ERR_INVALID_DTSHAPE (INA_ERR_INVALID | IARRAY_ES_DTSHAPE)

#define IARRAY_ERR_INVALID_DTYPE (INA_ERR_INVALID | IARRAY_ES_DTYPE)
#define IARRAY_ERR_INVALID_SHAPE (INA_ERR_INVALID | IARRAY_ES_SHAPE)
#define IARRAY_ERR_INVALID_CHUNKSHAPE (INA_ERR_INVALID | IARRAY_ES_CHUNKSHAPE)
#define IARRAY_ERR_INVALID_BLOCKSHAPE (INA_ERR_INVALID | IARRAY_ES_BLOCKSHAPE)
#define IARRAY_ERR_INVALID_ITERSHAPE (INA_ERR_INVALID | IARRAY_ES_ITERSHAPE)
#define IARRAY_ERR_INVALID_NDIM (INA_ERR_INVALID | IARRAY_ES_NDIM)
#define IARRAY_ERR_INVALID_AXIS (INA_ERR_INVALID | IARRAY_ES_AXIS)

#define IARRAY_ERR_INVALID_RNG_METHOD (INA_ERR_INVALID | IARRAY_ES_RNG_METHOD)
#define IARRAY_ERR_INVALID_RAND_METHOD (INA_ERR_INVALID | IARRAY_ES_RAND_METHOD)
#define IARRAY_ERR_INVALID_RAND_PARAM (INA_ERR_INVALID | IARRAY_ES_RAND_PARAM)

#define IARRAY_ERR_INVALID_EVAL_METHOD (INA_ERR_INVALID | IARRAY_ES_EVAL_METHOD)

#define IARRAY_ERR_INVALID_EVAL_ENGINE (INA_ERR_INVALID | IARRAY_ES_EVAL_ENGINE)
#define IARRAY_ERR_EVAL_ENGINE_FAILED (INA_ERR_FAILED | IARRAY_ES_EVAL_ENGINE)
#define IARRAY_ERR_EVAL_ENGINE_NOT_COMPILED (INA_ERR_COMPILED | IARRAY_ES_EVAL_ENGINE)
#define IARRAY_ERR_EVAL_ENGINE_OUT_OF_RANGE (INA_ERR_OUT_OF_RANGE | IARRAY_ES_EVAL_ENGINE)

#define IARRAY_ERR_INVALID_STORAGE (INA_ERR_INVALID | IARRAY_ES_STORAGE)
#define IARRAY_ERR_TOO_SMALL_BUFFER (INA_ERR_TOO_SMALL | IARRAY_ES_BUFFER)

#define IARRAY_ERR_GET_NCORES (INA_ERR_FAILED | IARRAY_ES_NCORES)
#define IARRAY_ERR_GET_CACHE_SIZES (INA_ERR_FAILED | IARRAY_ES_CACHE_SIZES)

#define IARRAY_ERR_CATERVA_FAILED (INA_ERR_FAILED | IARRAY_ES_CATERVA)
#define IARRAY_ERR_BLOSC_FAILED (INA_ERR_FAILED | IARRAY_ES_BLOSC)

#define IARRAY_ERR_RAND_METHOD_FAILED (IARRAY_ES_RAND_METHOD | INA_ERR_FAILED)
#define IARRAY_ERR_ASSERTION_FAILED (IARRAY_ES_ASSERTION | INA_ERR_FAILED)

#define IARRAY_ERR_END_ITER (IARRAY_ES_ITER | INA_ERR_COMPLETE)
#define IARRAY_ERR_NOT_END_ITER (IARRAY_ES_ITER | INA_ERR_NOT_COMPLETE)

#ifdef __WIN32__
#define access _access
#endif

#define IARRAY_TRACE1(cat, fmt, ...) INA_TRACE1(cat, "%s:%d\n" fmt, __FILE__, __LINE__, ##__VA_ARGS__)
#define IARRAY_TRACE2(cat, fmt, ...) INA_TRACE2(cat, "%s:%d\n" fmt, __FILE__, __LINE__, ##__VA_ARGS__)
#define IARRAY_TRACE3(cat, fmt, ...) INA_TRACE3(cat, "%s:%d\n" fmt, __FILE__, __LINE__, ##__VA_ARGS__)

#define IARRAY_FAIL_IF(cond) do { if ((cond)) {IARRAY_TRACE2(iarray.error, "Tracing: "); goto fail;}} while(0)
#define IARRAY_FAIL_IF_ERROR(rc) IARRAY_FAIL_IF(INA_FAILED((rc)))

#define IARRAY_RETURN_IF_FAILED(rc) do { if (INA_FAILED(rc)) {IARRAY_TRACE2(iarray.error, "Tracing: "); return ina_err_get_rc(); } } while(0)
#define IARRAY_ERR_CATERVA(rc) do {if (rc != CATERVA_SUCCEED) {IARRAY_RETURN_IF_FAILED(INA_ERROR(IARRAY_ERR_CATERVA_FAILED));}} while(0)

#define IARRAY_ITER_FINISH() do { if (ina_err_get_rc() != INA_RC_PACK(IARRAY_ERR_END_ITER, 0)) { \
    return INA_ERROR(IARRAY_ERR_NOT_END_ITER); } else { ina_err_reset();}} while(0)
typedef struct iarray_context_s iarray_context_t;
typedef struct iarray_container_s iarray_container_t;

typedef struct iarray_iter_write_s iarray_iter_write_t;

typedef struct iarray_iter_read_s iarray_iter_read_t;
typedef struct iarray_iter_read_block_s iarray_iter_read_block_t;
typedef struct iarray_iter_write_block_s iarray_iter_write_block_t;

typedef enum iarray_random_rng_e {
    IARRAY_RANDOM_RNG_MRG32K3A,
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
    IARRAY_DATA_TYPE_DOUBLE = 0,
    IARRAY_DATA_TYPE_FLOAT = 1,
    // IARRAY_DATA_TYPE_FLOAT16 = 2, reserve this values for a future support
    // IARRAY_DATA_TYPE_FLOAT8 = 3,
    IARRAY_DATA_TYPE_INT64 = 10,
    IARRAY_DATA_TYPE_INT32 = 11,
    IARRAY_DATA_TYPE_INT16 = 12,
    IARRAY_DATA_TYPE_INT8 = 13,
    IARRAY_DATA_TYPE_UINT64 = 16,
    IARRAY_DATA_TYPE_UINT32 = 17,
    IARRAY_DATA_TYPE_UINT16 = 18,
    IARRAY_DATA_TYPE_UINT8 = 19,
    IARRAY_DATA_TYPE_BOOL = 24,
    IARRAY_DATA_TYPE_MAX  // marker; must be the last entry
} iarray_data_type_t;

typedef enum iarray_storage_format_e {
    IARRAY_STORAGE_ROW_WISE = 0,
    IARRAY_STORAGE_COL_WISE
} iarray_storage_format_t;

// The first 3 bits (0, 1, 2) of eval_method are reserved for the eval method
typedef enum iarray_eval_method_e {
    IARRAY_EVAL_METHOD_AUTO = 0u,
    IARRAY_EVAL_METHOD_ITERCHUNK = 1u,
    IARRAY_EVAL_METHOD_ITERBLOSC = 2u,
} iarray_eval_method_t;


typedef enum iarray_filter_flags_e {
    IARRAY_COMP_SHUFFLE    = 0x1,
    IARRAY_COMP_BITSHUFFLE = 0x2,
    IARRAY_COMP_DELTA      = 0x4,
    IARRAY_COMP_TRUNC_PREC = 0x8,
} iarray_filter_flags_t;

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
    IARRAY_COMPRESSION_ZFP_FIXED_ACCURACY,
    IARRAY_COMPRESSION_ZFP_FIXED_PRECISION,
    IARRAY_COMPRESSION_ZFP_FIXED_RATE
} iarray_compression_codec_t;

typedef enum iarray_compression_favor_e {
    IARRAY_COMPRESSION_FAVOR_BALANCE = 0,
    IARRAY_COMPRESSION_FAVOR_SPEED,
    IARRAY_COMPRESSION_FAVOR_CRATIO,
} iarray_compression_favor_t;

typedef enum iarray_split_mode_e {
    IARRAY_ALWAYS_SPLIT = 1,
    IARRAY_NEVER_SPLIT = 2,
    IARRAY_AUTO_SPLIT = 3,
    IARRAY_FORWARD_COMPAT_SPLIT = 4,
} iarray_split_mode_t;

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
    iarray_compression_favor_t compression_favor;
    int use_dict;
    int splitmode;
    int filter_flags;
    unsigned int eval_method;
    int max_num_threads; /* Maximum number of threads to use */
    uint8_t fp_mantissa_bits; /* Only useful together with flag: IARRAY_COMP_TRUNC_PREC */
    bool btune;  /* Enable btune */
    uint8_t compression_meta; /* Only useful together with compression codecs: IARRAY_COMPRESSION_ZFP */
} iarray_config_t;

typedef struct iarray_dtshape_s {
    iarray_data_type_t dtype;
    int32_t dtype_size;
    int8_t ndim;     /* if ndim = 0 it is a scalar */
    int64_t shape[IARRAY_DIMENSION_MAX];
} iarray_dtshape_t;

typedef struct iarray_storage_s {
    char *urlpath;
    bool contiguous;
    int64_t chunkshape[IARRAY_DIMENSION_MAX];
    int64_t blockshape[IARRAY_DIMENSION_MAX];
} iarray_storage_t;

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

typedef struct iarray_random_ctx_s iarray_random_ctx_t;

static const iarray_config_t IARRAY_CONFIG_DEFAULTS = {
    .compression_codec = IARRAY_COMPRESSION_LZ4,
    .compression_level = 5,
    .compression_favor = IARRAY_COMPRESSION_FAVOR_BALANCE,
    .use_dict = 0,
    .splitmode = IARRAY_AUTO_SPLIT,
    .filter_flags = IARRAY_COMP_SHUFFLE,
    .eval_method = IARRAY_EVAL_METHOD_ITERBLOSC,
    .max_num_threads = 1,
    .fp_mantissa_bits = 0,
    .btune = true,
    .compression_meta = 0,
};

static const iarray_config_t IARRAY_CONFIG_NO_COMPRESSION = {
    .compression_codec = IARRAY_COMPRESSION_LZ4,
    .compression_level = 0,
    .compression_favor = IARRAY_COMPRESSION_FAVOR_BALANCE,
    .use_dict = 0,
    .splitmode = IARRAY_AUTO_SPLIT,
    .filter_flags = 0,
    .eval_method = 0,
    .max_num_threads = 1,
    .fp_mantissa_bits = 0
};

typedef struct _iarray_jug_var_s {
    const char *var;
    iarray_container_t *c;
} _iarray_jug_var_t;

typedef struct jug_expression_s jug_expression_t;

// Struct to be used as user parameter
typedef union {
    float f32;
    double f64;
    int32_t i32;
    int64_t i64;
    bool b;
} iarray_user_param_t;

typedef struct iarray_expression_s {
    iarray_context_t *ctx;
    ina_str_t expr;
    int32_t typesize;
    int64_t nbytes;
    int nvars;
    int32_t max_out_len;
    jug_expression_t *jug_expr;
    uint64_t jug_expr_func;
    iarray_dtshape_t *out_dtshape;
    iarray_storage_t *out_store_properties;
    iarray_container_t *out;
    _iarray_jug_var_t vars[IARRAY_EXPR_OPERANDS_MAX];
    iarray_user_param_t user_params[IARRAY_EXPR_USER_PARAMS_MAX];  // the input user parameters
    unsigned int nuser_params;
} iarray_expression_t;

typedef struct iarray_udf_registry_s iarray_udf_registry_t;
typedef struct iarray_udf_library_s iarray_udf_library_t;

INA_API(ina_rc_t) iarray_init(void);
INA_API(void) iarray_destroy(void);

INA_API(const char *) iarray_err_strerror(ina_rc_t error);

INA_API(ina_rc_t) iarray_context_new(iarray_config_t *cfg, iarray_context_t **ctx);
INA_API(void) iarray_context_free(iarray_context_t **ctx);

/*
 *  Get the number of (logical) cores (`ncores`) in the system.
 *
 *  `ncores` won't be larger than `max_ncores`.  If `max_ncores` is 0, there is not a maximum cap.
 *
 */
INA_API(ina_rc_t) iarray_get_ncores(int *ncores, int64_t max_ncores);

/*
 *  Get the L2 size in the system.
 */
INA_API(ina_rc_t) iarray_get_L2_size(uint64_t *L2_size);

/*
 *  Provide advice for the partition shape of a `dtshape`.
 *
 *  If success, storage->chunkshape and storage->blockshape will contain the advice.
 *
 *  `min_` and `max_` contain minimum and maximum values for chunksize and blocksize.
 *  If `min_` or `max_` are 0, they default to sensible values (fractions of CPU caches).
 *
 */
INA_API(ina_rc_t)
iarray_partition_advice(iarray_context_t *ctx, iarray_dtshape_t *dtshape, iarray_storage_t *storage,
                        int64_t min_chunksize, int64_t max_chunksize,
                        int64_t min_blocksize, int64_t max_blocksize);

/*
 * Provide advice for the block shapes for performing a matrix-matrix multiplication.
 *
 * `a` and `b` are supposed to have (M, K) and (K, N) dimensions respectively
 * `c` is supposed to have a partition size of (m, n)
 * The hint for the block shapes are going to be (m, k) and (k, n) respectively
 *
 * The hints will be stored in `blockshape_a` and `blockshape_b`, which needs to be provided by the user.
 * The number of components for the block shapes is 2.
 *
 *  `low` and `high` contain low and high values for the partition size.  If `low` is 0, it defaults
 *  to a fraction of L2 cache size.  If `high` is 0, it defaults to a fraction of L3 cache size.
 *
 * Note: When performing matrix-*vector* operations, just pass the N dimension as 1.  The `k` hint
 * will be valid for this case too.  In this case, always pass `blockshape_a` and `blockshape_b` with
 * 2-components too (even if `blockshape_b` only has a dimension in this case).
 *
 */
INA_API(ina_rc_t) iarray_matmul_advice(iarray_context_t *ctx,
                                       iarray_container_t *a,
                                       iarray_container_t *b,
                                       iarray_container_t *c,
                                       int64_t *blockshape_a,
                                       int64_t *blockshape_b,
                                       int64_t low,
                                       int64_t high);

INA_API(ina_rc_t) iarray_random_ctx_new(iarray_context_t *ctx,
                                        uint32_t seed,
                                        iarray_random_rng_t rng,
                                        iarray_random_ctx_t **rng_ctx);

INA_API(void) iarray_random_ctx_free(iarray_context_t *ctx,
                                     iarray_random_ctx_t **rng_ctx);

INA_API(ina_rc_t) iarray_random_dist_set_param(iarray_random_ctx_t *ctx,
                                               iarray_random_dist_parameter_t key,
                                               double value);

INA_API(ina_rc_t) iarray_uninit(iarray_context_t *ctx,
                         iarray_dtshape_t *dtshape,
                         iarray_storage_t *storage,
                         iarray_container_t **container);


INA_API(ina_rc_t) iarray_arange(iarray_context_t *ctx,
                                iarray_dtshape_t *dtshape,
                                double start,
                                double step,
                                iarray_storage_t *storage,
                                iarray_container_t **container);

INA_API(ina_rc_t) iarray_linspace(iarray_context_t *ctx,
                                  iarray_dtshape_t *dtshape,
                                  double start,
                                  double stop,
                                  iarray_storage_t *storage,
                                  iarray_container_t **container);

INA_API(ina_rc_t) iarray_logspace(iarray_context_t *ctx,
                                  iarray_dtshape_t *dtshape,
                                  double start,
                                  double stop,
                                  double base,
                                  iarray_storage_t *storage,
                                  iarray_container_t **container);

INA_API(ina_rc_t) iarray_tri(iarray_context_t *ctx,
                             iarray_dtshape_t *dtshape,
                             int64_t k,
                             iarray_storage_t *storage,
                             iarray_container_t **container);

INA_API(ina_rc_t) iarray_eye(iarray_context_t *ctx,
                             iarray_dtshape_t *dtshape,
                             iarray_storage_t *storage,
                             iarray_container_t **container);


INA_API(ina_rc_t) iarray_empty(iarray_context_t *ctx,
                               iarray_dtshape_t *dtshape,
                               iarray_storage_t *storage,
                               iarray_container_t **container);

INA_API(ina_rc_t) iarray_zeros(iarray_context_t *ctx,
                               iarray_dtshape_t *dtshape,
                               iarray_storage_t *storage,
                               iarray_container_t **container);

INA_API(ina_rc_t) iarray_ones(iarray_context_t *ctx,
                              iarray_dtshape_t *dtshape,
                              iarray_storage_t *storage,
                              iarray_container_t **container);

INA_API(ina_rc_t) iarray_fill(iarray_context_t *ctx,
                                iarray_dtshape_t *dtshape,
                                void *value,
                                iarray_storage_t *storage,
                                iarray_container_t **container);

INA_API(ina_rc_t) iarray_copy(iarray_context_t *ctx,
                              iarray_container_t *src,
                              bool view,
                              iarray_storage_t *storage,
                              iarray_container_t **dest);

INA_API(ina_rc_t) iarray_random_uniform(iarray_context_t *ctx,
                                        iarray_dtshape_t *dtshape,
                                        iarray_random_ctx_t *random_ctx,
                                        iarray_storage_t *storage,
                                        iarray_container_t **container);

INA_API(ina_rc_t) iarray_random_rand(iarray_context_t *ctx,
                                     iarray_dtshape_t *dtshape,
                                     iarray_random_ctx_t *rand_ctx,
                                     iarray_storage_t *storage,
                                     iarray_container_t **container);

INA_API(ina_rc_t) iarray_random_randn(iarray_context_t *ctx,
                                      iarray_dtshape_t *dtshape,
                                      iarray_random_ctx_t *rand_ctx,
                                      iarray_storage_t *storage,
                                      iarray_container_t **container);

INA_API(ina_rc_t) iarray_random_beta(iarray_context_t *ctx,
                                     iarray_dtshape_t *dtshape,
                                     iarray_random_ctx_t *rand_ctx,
                                     iarray_storage_t *storage,
                                     iarray_container_t **container);

INA_API(ina_rc_t) iarray_random_lognormal(iarray_context_t *ctx,
                                          iarray_dtshape_t *dtshape,
                                          iarray_random_ctx_t *rand_ctx,
                                          iarray_storage_t *storage,
                                          iarray_container_t **container);

INA_API(ina_rc_t) iarray_random_exponential(iarray_context_t *ctx,
                                            iarray_dtshape_t *dtshape,
                                            iarray_random_ctx_t *random_ctx,
                                            iarray_storage_t *storage,
                                            iarray_container_t **container);

INA_API(ina_rc_t) iarray_random_normal(iarray_context_t *ctx,
                                       iarray_dtshape_t *dtshape,
                                       iarray_random_ctx_t *random_ctx,
                                       iarray_storage_t *storage,
                                       iarray_container_t **container);

INA_API(ina_rc_t) iarray_random_bernoulli(iarray_context_t *ctx,
                                          iarray_dtshape_t *dtshape,
                                          iarray_random_ctx_t *random_ctx,
                                          iarray_storage_t *storage,
                                          iarray_container_t **container);

INA_API(ina_rc_t) iarray_random_binomial(iarray_context_t *ctx,
                                         iarray_dtshape_t *dtshape,
                                         iarray_random_ctx_t *random_ctx,
                                         iarray_storage_t *storage,
                                         iarray_container_t **container);

INA_API(ina_rc_t) iarray_random_poisson(iarray_context_t *ctx,
                                        iarray_dtshape_t *dtshape,
                                        iarray_random_ctx_t *random_ctx,
                                        iarray_storage_t *storage,
                                        iarray_container_t **container);

INA_API(ina_rc_t) iarray_random_kstest(iarray_context_t *ctx,
                                       iarray_container_t *container1,
                                       iarray_container_t *container2,
                                       bool *res);

INA_API(ina_rc_t) iarray_get_slice(iarray_context_t *ctx,
                                   iarray_container_t *src,
                                   const int64_t *start,
                                   const int64_t *stop,
                                   bool view,
                                   iarray_storage_t *storage,
                                   iarray_container_t **container);

INA_API(ina_rc_t) iarray_set_slice(iarray_context_t *ctx,
                                   iarray_container_t *container,
                                   const int64_t *start,
                                   const int64_t *stop,
                                   iarray_container_t *slice);

INA_API(ina_rc_t) iarray_get_slice_buffer(iarray_context_t *ctx,
                                          iarray_container_t *container,
                                          const int64_t *start,
                                          const int64_t *stop,
                                          void *buffer,
                                          int64_t buflen);

INA_API(ina_rc_t) iarray_set_slice_buffer(iarray_context_t *ctx,
                                          iarray_container_t *container,
                                          const int64_t *start,
                                          const int64_t *stop,
                                          void *buffer,
                                          int64_t buflen);

INA_API(ina_rc_t) iarray_container_load(iarray_context_t *ctx,
                                        char *urlpath,
                                        iarray_container_t **container);

INA_API(ina_rc_t) iarray_container_open(iarray_context_t *ctx,
                                        char *urlpath,
                                        iarray_container_t **container);

INA_API(ina_rc_t) iarray_container_save(iarray_context_t *ctx,
                                        iarray_container_t *container,
                                        char *urlpath);

INA_API(ina_rc_t) iarray_container_remove(char *urlpath);

INA_API(ina_rc_t) iarray_container_resize(iarray_context_t *ctx,
                                          iarray_container_t *container,
                                          int64_t *new_shape,
                                          int64_t *start);
INA_API(ina_rc_t) iarray_container_insert(iarray_context_t *ctx,
                                          iarray_container_t *container,
                                          void *buffer,
                                          int64_t buffersize,
                                          const int8_t axis,
                                          int64_t insert_start);
INA_API(ina_rc_t) iarray_container_append(iarray_context_t *ctx,
                                          iarray_container_t *container,
                                          void *buffer,
                                          int64_t buffersize,
                                          const int8_t axis);
INA_API(ina_rc_t) iarray_container_delete(iarray_context_t *ctx,
                                          iarray_container_t *container,
                                          const int8_t axis,
                                          int64_t delete_start,
                                          int64_t delete_len);


INA_API(ina_rc_t) iarray_squeeze_index(iarray_context_t *ctx,
                                       iarray_container_t *container,
                                       bool *index);

INA_API(ina_rc_t) iarray_squeeze(iarray_context_t *ctx,
                                 iarray_container_t *container);

INA_API(ina_rc_t) iarray_get_dtshape(iarray_context_t *ctx,
                                     iarray_container_t *c,
                                     iarray_dtshape_t *dtshape);

INA_API(ina_rc_t) iarray_get_storage(iarray_context_t *ctx,
                                     iarray_container_t *c,
                                     iarray_storage_t *storage);

INA_API(ina_rc_t) iarray_get_cfg(iarray_context_t *ctx,
                                 iarray_container_t *c,
                                 iarray_config_t *cfg);

INA_API(ina_rc_t) iarray_is_view(iarray_context_t *ctx,
                                 iarray_container_t *c,
                                 bool *view);

INA_API(ina_rc_t) iarray_from_buffer(iarray_context_t *ctx,
                                     iarray_dtshape_t *dtshape,
                                     void *buffer,
                                     int64_t buflen,
                                     iarray_storage_t *storage,
                                     iarray_container_t **container);

INA_API(ina_rc_t) iarray_to_buffer(iarray_context_t *ctx,
                                   iarray_container_t *container,
                                   void *buffer,
                                   int64_t buflen);

INA_API(bool) iarray_is_empty(iarray_container_t *container);

INA_API(ina_rc_t) iarray_container_dtshape_equal(iarray_dtshape_t *a, iarray_dtshape_t *b);
INA_API(ina_rc_t) iarray_container_info(iarray_container_t *container, int64_t *nbytes, int64_t *cbytes);

INA_API(void) iarray_container_free(iarray_context_t *ctx, iarray_container_t **container);

/* Comparison operators -> not supported yet as we only support float and double and return would be int8 */
INA_API(ina_rc_t) iarray_container_gt(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result);
INA_API(ina_rc_t) iarray_container_lt(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result);
INA_API(ina_rc_t) iarray_container_gte(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result);
INA_API(ina_rc_t) iarray_container_lte(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result);
INA_API(ina_rc_t) iarray_container_eq(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result);

INA_API(ina_rc_t) iarray_container_almost_equal(iarray_container_t *a, iarray_container_t *b, double tol);
INA_API(ina_rc_t) iarray_container_equal(iarray_container_t *a, iarray_container_t *b);

INA_API(ina_rc_t) iarray_container_is_symmetric(iarray_container_t *a);
INA_API(ina_rc_t) iarray_container_is_triangular(iarray_container_t *a);

/* Metalayers */
typedef struct {
    char *name;
    //!< The name of the metalayer
    uint8_t *sdata;
    //!< The serialized data to store
    int32_t size;
    //!< The size of the serialized data
} iarray_metalayer_t;

INA_API(ina_rc_t) iarray_vlmeta_exists(iarray_context_t *ctx, iarray_container_t *c, const char *name, bool *exists);
INA_API(ina_rc_t) iarray_vlmeta_add(iarray_context_t *ctx, iarray_container_t *c, iarray_metalayer_t *meta);
INA_API(ina_rc_t) iarray_vlmeta_update(iarray_context_t *ctx, iarray_container_t *c, iarray_metalayer_t *meta);
INA_API(ina_rc_t) iarray_vlmeta_get(iarray_context_t *ctx, iarray_container_t *c, const char *name, iarray_metalayer_t *meta);
INA_API(ina_rc_t) iarray_vlmeta_delete(iarray_context_t *ctx, iarray_container_t *c, const char *name);
INA_API(ina_rc_t) iarray_vlmeta_nitems(iarray_context_t *ctx, iarray_container_t *c, int16_t *nitems);
INA_API(ina_rc_t) iarray_vlmeta_get_names(iarray_context_t *ctx, iarray_container_t *c, char **names);

/* Reductions */
typedef enum iarray_reduce_fun_e {
    IARRAY_REDUCE_MAX,
    IARRAY_REDUCE_MIN,
    IARRAY_REDUCE_SUM,
    IARRAY_REDUCE_PROD,
    IARRAY_REDUCE_MEAN,
} iarray_reduce_func_t;

typedef struct iarray_reduce_function_s iarray_reduce_function_t;


INA_API(ina_rc_t) iarray_reduce(iarray_context_t *ctx,
                                iarray_container_t *a,
                                iarray_reduce_func_t func,
                                int8_t axis,
                                iarray_storage_t *storage,
                                iarray_container_t **b);

INA_API(ina_rc_t) iarray_reduce_multi(iarray_context_t *ctx,
                                      iarray_container_t *a,
                                      iarray_reduce_func_t func,
                                      int8_t naxis,
                                      const int8_t *axis,
                                      iarray_storage_t *storage,
                                      iarray_container_t **b);

/* linear algebra */
INA_API(ina_rc_t) iarray_linalg_matmul(iarray_context_t *ctx,
                                       iarray_container_t *a,
                                       iarray_container_t *b,
                                       iarray_storage_t *storage,
                                       iarray_container_t **c);

INA_API(ina_rc_t) iarray_linalg_transpose(iarray_context_t *ctx,
                                          iarray_container_t *a,
                                          iarray_container_t **b);

INA_API(ina_rc_t) iarray_linalg_inverse(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result);
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
                                             const int64_t *iter_blockshape,
                                             iarray_iter_read_block_value_t *value,
                                             bool external_buffer);

INA_API(void) iarray_iter_read_block_free(iarray_iter_read_block_t **itr);
INA_API(ina_rc_t) iarray_iter_read_block_next(iarray_iter_read_block_t *itr, void *buffer, int32_t bufsize);
INA_API(ina_rc_t) iarray_iter_read_block_has_next(iarray_iter_read_block_t *itr);

INA_API(ina_rc_t) iarray_iter_write_block_new(iarray_context_t *ctx,
                                              iarray_iter_write_block_t **itr,
                                              iarray_container_t *cont,
                                              const int64_t *iter_blockshape,
                                              iarray_iter_write_block_value_t *value,
                                              bool external_buffer);

INA_API(void) iarray_iter_write_block_free(iarray_iter_write_block_t **itr);
INA_API(ina_rc_t) iarray_iter_write_block_next(iarray_iter_write_block_t *itr, void *buffer, int32_t bufsize);
INA_API(ina_rc_t) iarray_iter_write_block_has_next(iarray_iter_write_block_t *itr);

/* Expressions */
INA_API(ina_rc_t) iarray_expr_new(iarray_context_t *ctx, iarray_data_type_t dtype, iarray_expression_t **e);
INA_API(void) iarray_expr_free(iarray_context_t *ctx, iarray_expression_t **e);

INA_API(ina_rc_t) iarray_expr_bind(iarray_expression_t *e, const char *var, iarray_container_t *val);
INA_API(ina_rc_t) iarray_expr_bind_out_properties(iarray_expression_t *e, iarray_dtshape_t *dtshape, iarray_storage_t *store);
INA_API(ina_rc_t) iarray_expr_bind_param(iarray_expression_t *e, iarray_user_param_t val);

INA_API(ina_rc_t) iarray_expr_bind_scalar_float(iarray_expression_t *e, const char *var, float val);
INA_API(ina_rc_t) iarray_expr_bind_scalar_double(iarray_expression_t *e, const char *var, double val);

INA_API(ina_rc_t) iarray_expr_compile(iarray_expression_t *e, const char *expr);
INA_API(ina_rc_t) iarray_expr_compile_udf(iarray_expression_t *e,
                                          int llvm_bc_len,
                                          const char *llvm_bc,
                                          const char *name);

INA_API(ina_rc_t) iarray_eval(iarray_expression_t *e, iarray_container_t **container);

//FIXME: remove
INA_API(ina_rc_t) iarray_expr_get_mp(iarray_expression_t *e, ina_mempool_t **mp);
INA_API(ina_rc_t) iarray_expr_get_nthreads(iarray_expression_t *e, int *nthreads);


/* Zarr proxy */

typedef void (*zhandler_ptr) (char *zarr_urlpath, int64_t *slice_start, int64_t *slice_stop,
                                       uint8_t *dest);

INA_API(ina_rc_t) iarray_add_zproxy_postfilter(iarray_container_t *src, char *zarr_urlpath, zhandler_ptr zhandler);

INA_API(ina_rc_t) iarray_opt_gemv(iarray_context_t *ctx,
                                  iarray_container_t *a,
                                  iarray_container_t *b,
                                  iarray_storage_t *storage,
                                  iarray_container_t **c);

INA_API(ina_rc_t) iarray_opt_gemm(iarray_context_t *ctx,
                                  iarray_container_t *a,
                                  iarray_container_t *b,
                                  iarray_storage_t *storage,
                                  iarray_container_t **c);

INA_API(ina_rc_t) iarray_opt_gemm_b(iarray_context_t *ctx,
                                    iarray_container_t *a,
                                    iarray_container_t *b,
                                    iarray_storage_t *storage,
                                    iarray_container_t **c);

INA_API(ina_rc_t) iarray_opt_gemm_a(iarray_context_t *ctx,
                                    iarray_container_t *a,
                                    iarray_container_t *b,
                                    iarray_storage_t *storage,
                                    iarray_container_t **c);

/* UDF (User defined functions) registry and library functionality */

INA_API(ina_rc_t) iarray_udf_registry_new(iarray_udf_registry_t **udf_registry);

INA_API(void) iarray_udf_registry_free(iarray_udf_registry_t **udf_registry);

INA_API(ina_rc_t) iarray_udf_library_new(const char *name, iarray_udf_library_t **lib);
INA_API(void) iarray_udf_library_free(iarray_udf_library_t **lib);

INA_API(ina_rc_t) iarray_udf_func_register(iarray_udf_library_t *lib,
                                           int llvm_bc_len,
                                           const char *llvm_bc,
                                           iarray_data_type_t return_type,
                                           int num_args,
                                           iarray_data_type_t *arg_types,
                                           const char *name);

INA_API(ina_rc_t) iarray_udf_func_lookup(const char *full_name, uint64_t *function_ptr);


/* Indexing */

INA_API(ina_rc_t) iarray_set_orthogonal_selection(iarray_context_t *ctx,
                                                  iarray_container_t *c,
                                                  int64_t **selection, int64_t *selection_size,
                                                  void *buffer,
                                                  int64_t *buffer_shape,
                                                  int64_t buffer_size);

INA_API(ina_rc_t) iarray_get_orthogonal_selection(iarray_context_t *ctx,
                                                  iarray_container_t *c,
                                                  int64_t **selection, int64_t *selection_size,
                                                  void *buffer,
                                                  int64_t *buffer_shape,
                                                  int64_t buffer_size);
#endif

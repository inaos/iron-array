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

#ifndef _IARRAY_PRIVATE_H_
#define _IARRAY_PRIVATE_H_

#include <libiarray/iarray.h>

/* Dependencies */
#include <blosc2.h>
#include <caterva.h>
#include <mkl.h>

 /* Sizes */
#define _IARRAY_SIZE_KB  (1024)
#define _IARRAY_SIZE_MB  (1024*_IARRAY_SIZE_KB)
#define _IARRAY_SIZE_GB  (1024*_IARRAY_SIZE_MB)

/* Mempools */
/* FIXME: do some serious benchmarking for finding the optimal values below
 * (decide this at runtime maybe?) */
#define _IARRAY_MEMPOOL_OP_CHUNKS (1024*1024)
#define _IARRAY_MEMPOOL_EVAL (1024*1024)
#define _IARRAY_MEMPOOL_EVAL_TMP (1024*1024)

typedef enum iarray_functype_e {
    IARRAY_FUNC_ABS,
    IARRAY_FUNC_ACOS,
    IARRAY_FUNC_ASIN,
    IARRAY_FUNC_ATAN,
    IARRAY_FUNC_ATAN2,
    IARRAY_FUNC_CEIL,
    IARRAY_FUNC_COS,
    IARRAY_FUNC_COSH,
    IARRAY_FUNC_E,
    IARRAY_FUNC_EXP,
    IARRAY_FUNC_FAC,
    IARRAY_FUNC_FLOOR,
    IARRAY_FUNC_LN,
    IARRAY_FUNC_LOG,
    IARRAY_FUNC_LOG10,
    IARRAY_FUNC_NCR,
    IARRAY_FUNC_NEGATE,
    IARRAY_FUNC_NPR,
    IARRAY_FUNC_PI,
    IARRAY_FUNC_POW,
    IARRAY_FUNC_SIN,
    IARRAY_FUNC_SINH,
    IARRAY_FUNC_SQRT,
    IARRAY_FUNC_TAN,
    IARRAY_FUNC_TANH,
} iarray_functype_t;

typedef enum iarray_optype_e {
    IARRAY_OPERATION_TYPE_ADD,
    IARRAY_OPERATION_TYPE_SUB,
    IARRAY_OPERATION_TYPE_MUL,
    IARRAY_OPERATION_TYPE_DIVIDE,
    IARRAY_OPERATION_TYPE_NEGATE,
} iarray_optype_t;

typedef enum iarray_blas_type_e {
    IARRAY_OPERATION_TYPE_BLAS1,
    IARRAY_OPERATION_TYPE_BLAS2,
    IARRAY_OPERATION_TYPE_BLAS3
} iarray_blas_type_t;

struct iarray_context_s {
    iarray_config_t *cfg;
    ina_mempool_t *mp;
    ina_mempool_t *mp_chunk_cache;
    ina_mempool_t *mp_op;
    ina_mempool_t *mp_tmp_out;
    blosc2_prefilter_fn prefilter_fn;
    blosc2_prefilter_params *prefilter_params;
    /* FIXME: track expressions -> list */
};

typedef struct _iarray_container_store_s {
    ina_str_t id;
} _iarray_container_store_t;

typedef struct iarray_auxshape_s {
    int64_t offset[IARRAY_DIMENSION_MAX];
    int64_t shape_wos[IARRAY_DIMENSION_MAX];
    int64_t chunkshape_wos[IARRAY_DIMENSION_MAX];
    int64_t blockshape_wos[IARRAY_DIMENSION_MAX];
    int8_t index[IARRAY_DIMENSION_MAX];
} iarray_auxshape_t;

struct iarray_container_s {
    iarray_dtshape_t *dtshape;
    iarray_auxshape_t *auxshape;
    caterva_array_t *catarr;
    iarray_storage_t *storage;
    bool view;
    bool transposed;
    union {
        float f;
        double d;
    } scalar_value;
};

typedef struct iarray_iter_write_s {
    iarray_context_t *ctx;
    iarray_container_t *container;
    iarray_iter_write_value_t *val;
    uint8_t *chunk;
    void *pointer;

    int64_t nelem;
    int64_t nblock;
    int64_t nelem_block;

    int64_t *cur_block_shape;
    int64_t cur_block_size;
    int64_t *cur_block_index;

    int64_t *elem_index; // The elem index in coord
    int64_t elem_flat_index; // The elem index if the container will be flatten

    caterva_context_t *cat_ctx;
} iarray_iter_write_t;

static const iarray_iter_write_t IARRAY_ITER_WRITE_EMPTY = {0};

typedef struct iarray_iter_read_s {
    iarray_context_t *ctx;
    iarray_container_t *cont;
    iarray_iter_read_value_t *val;
    uint8_t *chunk;
    void *pointer;

    int64_t nelem; // The element counter in container
    int64_t nelem_block; // The element counter in a block
    int64_t nblock; // The block counter

    int64_t *cur_block_index; // The current block index
    int64_t cur_block_size; // The current block size
    int64_t *cur_block_shape; // The current block shape

    int64_t *block_shape; // The desired block shape (it will be the shape or the chunkshape)
    int64_t cont_size; // The container size

    int64_t *elem_index; // The elem index in coord
    int64_t elem_flat_index; // The elem index if the container will be flatten
} iarray_iter_read_t;


static const iarray_iter_read_t IARRAY_ITER_READ_EMPTY = {0};

typedef struct iarray_iter_write_block_s {
    iarray_context_t *ctx;
    iarray_container_t *cont;
    iarray_iter_write_block_value_t *val;
    uint8_t *block; // Pointer to a buffer of data
    void **block_pointer; // Pointer to a buffer pointer
    int64_t total_blocks; // Total number of blocks
    int64_t *block_shape; // The desired block shape
    int64_t block_shape_size; //The block shape size (number of elements)
    int64_t *cur_block_shape; // The shape of the current block (can be diff to the block shape passed)
    int64_t cur_block_size; // The size of the current block
    int64_t *cur_block_index; // The position of the block in the container
    int64_t *cur_elem_index; // The position of the first element of the block in the container
    int64_t *cont_eshape; // The extended shape of the container
    int64_t cont_esize; // The size of the extended shape
    int64_t nblock; // The block counter
    bool contiguous; // Flag to avoid copies using plainbuffer
    bool compressed_chunk_buffer;  // Flag to append an already compressed buffer
    bool external_buffer; // Flag to indicate if a external chunk is passed

    caterva_context_t *cat_ctx;
} iarray_iter_write_block_t;

static const iarray_iter_write_block_t IARRAY_ITER_WRITE_BLOCK_EMPTY = {0};

typedef struct iarray_iter_read_block_s {
    iarray_context_t *ctx;
    iarray_container_t *cont;
    iarray_iter_read_block_value_t *val;
    uint8_t *block; // Pointer to a buffer of data
    void **block_pointer; // Pointer to a buffer pointer
    int64_t total_blocks; // Total number of blocks
    int64_t *aux; // Aux variable used
    int64_t *block_shape; // The blockshape to be iterated
    int64_t block_shape_size; // The size of the blockshape (number of elements)
    int64_t *cur_block_shape; // The shape of the current block (can be diff to the block shape passed)
    int64_t cur_block_size; // The size of the current block
    int64_t *cur_block_index; // The position of the block in the container
    int64_t *cur_elem_index; // The position of the first element of the block in the container
    int64_t nblock; // The block counter
    bool contiguous; // Flag to avoid copies using plainbuffer
    bool external_buffer; // Flag to indicate if a external chunk is passed
    bool padding; // Iterate using padding or not
} iarray_iter_read_block_t;

static const iarray_iter_read_block_t IARRAY_ITER_READ_BLOCK_EMPTY = {0};

typedef struct iarray_iter_matmul_s {
    iarray_context_t *ctx;
    iarray_container_t *container1;
    iarray_container_t *container2;
    int64_t B0;
    int64_t B1;
    int64_t B2;
    int64_t M;
    int64_t K;
    int64_t N;
    int64_t nchunk1;
    int64_t nchunk2;
    int64_t cont;
} iarray_iter_matmul_t;

typedef struct iarray_variable_s {
    const char *name;
    const void *address;
    iarray_dtshape_t dtshape;
    void *context;
} iarray_variable_t;


typedef struct iarray_temporary_s {
    iarray_dtshape_t *dtshape;
    size_t size;
    void *data;
    union {
        float f;
        double d;
    } scalar_value;
} iarray_temporary_t;

typedef void(*_iarray_vml_fun_d_ab)(const MKL_INT n, const double a[], const double b[], double r[]);
typedef void(*_iarray_vml_fun_s_ab)(const MKL_INT n, const float a[], const float b[], float r[]);

typedef void(*_iarray_vml_fun_d_a)(const MKL_INT n, const double a[], double r[]);
typedef void(*_iarray_vml_fun_s_a)(const MKL_INT n, const float a[], float r[]);

ina_rc_t iarray_temporary_new(iarray_expression_t *expr, iarray_container_t *c, iarray_dtshape_t *dtshape, iarray_temporary_t **temp);

ina_rc_t iarray_shape_size(iarray_dtshape_t *dtshape, size_t *size);

/* FIXME: since we want to keep the changes to tinyexpr as little as possible we deviate from our usual function decls */
iarray_temporary_t* _iarray_func(iarray_expression_t *expr, iarray_temporary_t *operand1,
                                 iarray_temporary_t *operand2, iarray_functype_t func);

//static iarray_temporary_t* _iarray_op(iarray_temporary_t *lhs, iarray_temporary_t *rhs, iarray_optype_t op);
iarray_temporary_t* _iarray_op_add(iarray_expression_t *expr, iarray_temporary_t *lhs, iarray_temporary_t *rhs);
iarray_temporary_t* _iarray_op_sub(iarray_expression_t *expr, iarray_temporary_t *lhs, iarray_temporary_t *rhs);
iarray_temporary_t* _iarray_op_mul(iarray_expression_t *expr, iarray_temporary_t *lhs, iarray_temporary_t *rhs);
iarray_temporary_t* _iarray_op_divide(iarray_expression_t *expr, iarray_temporary_t *lhs, iarray_temporary_t *rhs);


// Iterators
ina_rc_t _iarray_iter_matmul_new(iarray_context_t *ctx, iarray_container_t *container1,
                                 iarray_container_t *container2, int64_t *ishape_a,
                                 int64_t *ishape_b, iarray_iter_matmul_t **itr);
void _iarray_iter_matmul_free(iarray_iter_matmul_t **itr);
void _iarray_iter_matmul_init(iarray_iter_matmul_t *itr);
void _iarray_iter_matmul_next(iarray_iter_matmul_t *itr);
int _iarray_iter_matmul_finished(iarray_iter_matmul_t *itr);

// Utilities
bool _iarray_file_exists(const char *filename);

ina_rc_t _iarray_get_slice_buffer(iarray_context_t *ctx,
                                  iarray_container_t *container,
                                  int64_t *start,
                                  int64_t *stop,
                                  int64_t *chunkshape,
                                  void *buffer,
                                  int64_t buflen);

INA_API(ina_rc_t) _iarray_get_slice_buffer_no_copy(iarray_context_t *ctx,
                                                   iarray_container_t *container,
                                                   int64_t *start,
                                                   int64_t *stop,
                                                   void **buffer,
                                                   int64_t buflen);


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

/* Blosc private functions */
ina_rc_t iarray_create_blosc_cparams(blosc2_cparams *cparams, iarray_context_t *ctx, int8_t typesize, int32_t blocksize);

/* Caterva private functions */
ina_rc_t iarray_create_caterva_cfg(iarray_config_t *cfg, void *(*alloc)(size_t), void (*free)(void *), caterva_config_t *cat_cfg);
ina_rc_t iarray_create_caterva_params(iarray_dtshape_t *dtshape, caterva_params_t *cat_params);
ina_rc_t iarray_create_caterva_storage(iarray_dtshape_t *dtshape, iarray_storage_t *storage, caterva_storage_t *cat_storage);

#endif

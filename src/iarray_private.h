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
#include "btune/iabtune.h"

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
    void *expr_vars;
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

/* RANDOM */
struct iarray_random_ctx_s {
    iarray_random_rng_t rng;
    uint32_t seed;
    VSLStreamStatePtr stream;
    double dparams[IARRAY_RANDOM_DIST_PARAM_SENTINEL];
    float fparams[IARRAY_RANDOM_DIST_PARAM_SENTINEL];
};

typedef int (* iarray_random_method_fn) (iarray_random_ctx_t *ctx,
                                         VSLStreamStatePtr stream,
                                         uint8_t itemsize,
                                         int32_t blocksize,
                                         uint8_t *buffer);


/* ITERATORS */
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

    caterva_ctx_t *cat_ctx;
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
    bool compressed_chunk_buffer;  // Flag to append an already compressed buffer
    bool external_buffer; // Flag to indicate if a external chunk is passed

    caterva_ctx_t *cat_ctx;
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
bool _iarray_path_exists(const char *urlpath);

ina_rc_t _iarray_get_slice_buffer(iarray_context_t *ctx,
                                  iarray_container_t *container,
                                  const int64_t *start,
                                  const int64_t *stop,
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

ina_rc_t iarray_set_dtype_size(iarray_dtshape_t *dtshape);

static int32_t _iarray_serialize_meta(iarray_data_type_t dtype, uint8_t **smeta)
{
    if (smeta == NULL) {
        return -1;
    }
    if (dtype > IARRAY_DATA_TYPE_MAX) {
        return -1;
    }
    int32_t smeta_len = 4;  // the dtype should take less than 7-bit, so 1 byte is enough to store it
    *smeta = malloc((size_t)smeta_len);

    uint8_t *pmeta = *smeta;

    *(*smeta + 0) = 0x93;  // [msgpack] fixarray of 3 elements

    // version
    *(*smeta + 1) = 0;

    // dtype entry
    *(*smeta + 2) = (uint8_t) dtype;  // positive fixnum (7-bit positive integer)

    // flags (initialising all the entries to 0)
    *(*smeta + 3) = 0;  // positive fixnum (7-bit for flags)

    return smeta_len;
}


static ina_rc_t _iarray_deserialize_meta(uint8_t *smeta, uint32_t smeta_len, iarray_data_type_t *dtype) {
    INA_UNUSED(smeta_len);
    INA_VERIFY_NOT_NULL(smeta);
    INA_VERIFY_NOT_NULL(dtype);

    uint8_t *pmeta = smeta;
    INA_ASSERT_EQUAL(*pmeta, 0x93);
    pmeta += 1;

    //version
    uint8_t version = *pmeta;
    INA_USED_BY_ASSERT(version);
    pmeta +=1;

    // We only have an entry with the datatype (enumerated < 128)
    *dtype = *pmeta;
    pmeta += 1;

    // Transpose byte
    pmeta += 1;

    assert(pmeta - smeta == smeta_len);

    if (*dtype >= IARRAY_DATA_TYPE_MAX) {
        IARRAY_TRACE1(iarray.error, "The data type is invalid");
        return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
    }

    return INA_SUCCESS;
}

ina_rc_t iarray_container_new(iarray_context_t *ctx, iarray_dtshape_t *dtshape,
                              iarray_storage_t *storage, iarray_container_t **container);

/* Constructor machinery */
typedef struct {
    iarray_container_t *a;  //!< The container to be built
    uint8_t ndim;  //!< The number of dimensions
    uint8_t itemsize;  //!< The size (in bytes) of each item
    int64_t strides[IARRAY_DIMENSION_MAX];  //!< The strides (for an item) inside the array
    int64_t chunk_strides[IARRAY_DIMENSION_MAX];  //!< The strides (for an item) inside a chunk
    int64_t block_strides[IARRAY_DIMENSION_MAX];  //!< The strides (for an item) inside a block
} iarray_constructor_array_info_t;


typedef int (* iarray_constructor_array_init_fn)(iarray_constructor_array_info_t *array_info,
                                                 void **custom_info);

typedef int (* iarray_constructor_array_destroy_fn)(iarray_constructor_array_info_t *array_info,
                                                    void **custom_info);

typedef struct {
    int64_t chunk_strides_block[IARRAY_DIMENSION_MAX];  //!< The strides (for a block) inside a chunk
    int64_t index[IARRAY_DIMENSION_MAX];  //!< The index of the chunk inside the array
    int64_t index_flat;  //!< The index of the chunk (flattened) inside the array
    int64_t start[IARRAY_DIMENSION_MAX];  //!< The index of the first chunk item
    int64_t stop[IARRAY_DIMENSION_MAX];  //!< The index of the last chunk item
    int64_t shape[IARRAY_DIMENSION_MAX];  //!< The chunk shape without padding
} iarray_constructor_chunk_info_t;


typedef int (* iarray_constructor_chunk_init_fn)(iarray_constructor_array_info_t *array_info,
                                                 iarray_constructor_chunk_info_t *chunk_info,
                                                 void *custom_info,
                                                 void **custom_chunk_info);

typedef int (* iarray_constructor_chunk_destroy_fn)(iarray_constructor_array_info_t *array_info,
                                                    iarray_constructor_chunk_info_t *chunk_info,
                                                    void *custom_info,
                                                    void **custom_chunk_info);

typedef struct {
    int64_t index_in_chunk[IARRAY_DIMENSION_MAX];  //!< The index of the block inside its chunk
    int64_t index_in_chunk_flat;  //!< The index of the block (flattened) inside its chunk
    int64_t block_strides[IARRAY_DIMENSION_MAX];  //!< The strides (for an item) inside the block without padding
    int64_t start[IARRAY_DIMENSION_MAX];  //!< The index of the first block item
    int64_t stop[IARRAY_DIMENSION_MAX];  //!< The index of the last block item
    int64_t shape[IARRAY_DIMENSION_MAX];  //!< The block shape without padding
    int64_t size;  //!< The block size without padding
    uint32_t tid;  //!< The thread id that is processing the block
} iarray_constructor_block_info_t;

typedef int (* iarray_constructor_block_init_fn)(iarray_constructor_array_info_t *array_info,
                                                 iarray_constructor_chunk_info_t *chunk_info,
                                                 iarray_constructor_block_info_t *block_info,
                                                 void *custom_info,
                                                 void *custom_chunk_info,
                                                 void **custom_block_info);

typedef int (* iarray_constructor_block_destroy_fn)(iarray_constructor_array_info_t *array_info,
                                                    iarray_constructor_chunk_info_t *chunk_info,
                                                    iarray_constructor_block_info_t *block_info,
                                                    void *custom_info,
                                                    void *custom_chunk_info,
                                                    void **custom_block_info);

typedef struct {
    int64_t index_in_block[IARRAY_DIMENSION_MAX];  //!< The index of the item inside the block
    int64_t index_in_block_flat;  //!< The index of the item (flattened) inside the block
    int64_t index_in_block2_flat;  //!< The index of the item (flattened) inside the block with padding
    int64_t index[IARRAY_DIMENSION_MAX];  //!< The index of the item inside the array
    int64_t index_flat;  //!< The index of the item (flattened) inside the array
} iarray_constructor_item_info_t;


typedef ina_rc_t (* iarray_constructor_item_fn)(iarray_constructor_array_info_t *array_params,
                                              iarray_constructor_chunk_info_t *chunk_params,
                                              iarray_constructor_block_info_t *block_params,
                                              iarray_constructor_item_info_t *item_params,
                                              void *custom_params,
                                              void *custom_chunk_params,
                                              void *custom_block_params,
                                              uint8_t *item);


typedef int (* iarray_constructor_generator_fn)(uint8_t *dest,
                                                iarray_constructor_array_info_t *array_params,
                                                iarray_constructor_chunk_info_t *chunk_params,
                                                iarray_constructor_block_info_t *block_params,
                                                void *custom_params,
                                                void *custom_chunk_params,
                                                void *custom_block_params,
                                                iarray_constructor_item_fn item_fn);

typedef struct {
    void *constructor_info;
    iarray_constructor_array_init_fn array_init_fn;
    iarray_constructor_array_destroy_fn array_destroy_fn;
    iarray_constructor_chunk_init_fn chunk_init_fn;
    iarray_constructor_chunk_destroy_fn chunk_destroy_fn;
    iarray_constructor_block_init_fn block_init_fn;
    iarray_constructor_block_destroy_fn block_destroy_fn;
    iarray_constructor_item_fn item_fn;
    iarray_constructor_generator_fn generator_fn;
} iarray_constructor_block_params_t;

static iarray_constructor_block_params_t IARRAY_CONSTRUCTOR_BLOCK_PARAMS_DEFAULT = {0};

typedef struct {
    void *constructor_info;
    iarray_constructor_array_init_fn array_init_fn;
    iarray_constructor_array_destroy_fn array_destroy_fn;
    iarray_constructor_chunk_init_fn chunk_init_fn;
    iarray_constructor_chunk_destroy_fn chunk_destroy_fn;
    iarray_constructor_block_init_fn block_init_fn;
    iarray_constructor_block_destroy_fn block_destroy_fn;
    iarray_constructor_item_fn item_fn;
} iarray_constructor_element_params_t;

static iarray_constructor_element_params_t IARRAY_CONSTRUCTOR_ELEMENT_PARAMS_DEFAULT = {0};

ina_rc_t iarray_constructor_block(iarray_context_t *ctx,
                                  iarray_dtshape_t *dtshape,
                                  iarray_constructor_block_params_t *const_params,
                                  iarray_storage_t *storage,
                                  iarray_container_t **container);

ina_rc_t iarray_constructor_element(iarray_context_t *ctx,
                                    iarray_dtshape_t *dtshape,
                                    iarray_constructor_element_params_t *element_params,
                                    iarray_storage_t *storage,
                                    iarray_container_t **container);

INA_API(ina_rc_t) iarray_random_prefilter(iarray_context_t *ctx,
                                          iarray_dtshape_t *dtshape,
                                          iarray_random_ctx_t *random_ctx,
                                          iarray_random_method_fn random_method_fn,
                                          iarray_storage_t *storage,
                                          iarray_container_t **container);

// Inline functions
static inline void compute_strides(uint8_t ndim, int64_t *shape, int64_t *strides) {
    if (ndim == 0) {
        return;
    }
    strides[ndim - 1] = 1;
    for (int j = ndim - 2; j >= 0; --j) {
        strides[j] = shape[j + 1] * strides[j + 1];
    }
}

static inline void iarray_index_multidim_to_unidim(uint8_t ndim, int64_t *strides, int64_t *index, int64_t *i) {
    *i = 0;
    for (int j = 0; j < ndim; ++j) {
        *i += index[j] * strides[j];
    }
}

static inline void iarray_index_unidim_to_multidim(uint8_t ndim, int64_t *strides, int64_t i, int64_t *index) {
    if (ndim == 0) {
        return;
    }

    index[0] = i / strides[0];
    for (int j = 1; j < ndim; ++j) {
        index[j] = (i % strides[j - 1]) / strides[j];
    }
}

static inline void iarray_index_unidim_to_multidim_shape(int8_t ndim, int64_t *shape, int64_t i, int64_t *index) {
    if (ndim == 0) {
        return;
    }

    int64_t strides[CATERVA_MAX_DIM];
    compute_strides(ndim, shape, strides);

    index[0] = i / strides[0];
    for (int j = 1; j < ndim; ++j) {
        index[j] = (i % strides[j - 1]) / strides[j];
    }
}

#endif

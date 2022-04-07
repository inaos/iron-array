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

#ifndef _MINJUGG_H_
#define _MINJUGG_H_

#include <libinac/lib.h>
#include <libiarray/iarray.h>

typedef struct jug_context_s jug_context_t;
typedef struct jug_udf_s jug_udf_t;
typedef struct jug_udf_registry_s jug_udf_registry_t;
typedef struct jug_udf_library_s jug_udf_library_t;

typedef enum jug_expression_dtype_e {
    JUG_EXPRESSION_DTYPE_DOUBLE = 1,
    JUG_EXPRESSION_DTYPE_FLOAT = 2,
    JUG_EXPRESSION_DTYPE_SINT8 = 3,
    JUG_EXPRESSION_DTYPE_SINT16 = 4,
    JUG_EXPRESSION_DTYPE_SINT32 = 5,
    JUG_EXPRESSION_DTYPE_SINT64 = 6,
    JUG_EXPRESSION_DTYPE_UINT8 = 7,
    JUG_EXPRESSION_DTYPE_UINT16 = 8,
    JUG_EXPRESSION_DTYPE_UINT32 = 9,
    JUG_EXPRESSION_DTYPE_UINT64 = 10,
} jug_expression_dtype_t;

INA_API(ina_rc_t) jug_init(void);
INA_API(void) jug_destroy(void);

INA_API(ina_rc_t) jug_expression_new(jug_expression_t **expr, jug_expression_dtype_t dtype);
INA_API(void) jug_expression_free(jug_expression_t **expr);

INA_API(ina_rc_t) jug_expression_operands_parse(jug_expression_t *e, 
                                                const char *expr, 
                                                int *num_operands, 
                                                ina_str_t *operands);
INA_API(void) jug_exression_operands_free(jug_expression_t *e, ina_str_t *operands);
    
INA_API(ina_rc_t) jug_expression_compile(jug_expression_t *e,
                                         const char *expr, 
                                         int num_vars, 
                                         void *vars, 
                                         uint64_t *function_addr);

INA_API(ina_rc_t) jug_udf_compile(jug_expression_t *e,
                                  int llvm_bc_len,
                                  const char *llvm_bc,
                                  const char *name,
                                  uint64_t *function_addr);

INA_API(ina_rc_t) jug_udf_registry_new(jug_udf_registry_t **udf_registry);
INA_API(void) jug_udf_registry_free(jug_udf_registry_t **udf_registry);

INA_API(ina_rc_t) jug_udf_library_new(jug_udf_registry_t *registry, const char *name, jug_udf_library_t **udf_lib);
INA_API(void) jug_udf_library_free(jug_udf_registry_t *registry, jug_udf_library_t **jug_lib);

INA_API(ina_rc_t) jug_udf_library_compile(jug_udf_library_t *lib,
                                          const char *name,
                                          int llvm_bc_len,
                                          const char *llvm_bc);

/* FIXME the below declarations actually do not belong here */
typedef enum te_expr_type_e {
    EXPR_TYPE_ADD,
    EXPR_TYPE_SUB,
    EXPR_TYPE_MUL,
    EXPR_TYPE_DIVIDE,
    EXPR_TYPE_NEGATE,
    EXPR_TYPE_COMMA,
    EXPR_TYPE_ABS,
    EXPR_TYPE_ACOS,
    EXPR_TYPE_ASIN,
    EXPR_TYPE_ATAN,
    EXPR_TYPE_ATAN2,
    EXPR_TYPE_CEIL,
    EXPR_TYPE_COS,
    EXPR_TYPE_COSH,
    EXPR_TYPE_E,
    EXPR_TYPE_EXP,
    EXPR_TYPE_FAC,
    EXPR_TYPE_FLOOR,
    EXPR_TYPE_LOG,
    EXPR_TYPE_LOG10,
    EXPR_TYPE_NCR,
    EXPR_TYPE_NPR,
    EXPR_TYPE_PI,
    EXPR_TYPE_POW,
    EXPR_TYPE_SIN,
    EXPR_TYPE_SINH,
    EXPR_TYPE_SQRT,
    EXPR_TYPE_TAN,
    EXPR_TYPE_TANH,
    EXPR_TYPE_FMOD,
    EXPR_TYPE_MIN,
    EXPR_TYPE_MAX
} te_expr_type_t;

typedef struct jug_te_variable {
    const char *name;
    te_expr_type_t address;
    int type;
    void *context;
} jug_te_variable;

#endif

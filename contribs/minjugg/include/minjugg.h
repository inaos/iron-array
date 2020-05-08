/*
 * Copyright INAOS GmbH, Thalwil, 2019.
 * Copyright Francesc Alted, 2019.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of INAOS GmbH
 * and Francesc Alted ("Confidential Information"). You shall not disclose such Confidential
 * Information and shall use it only in accordance with the terms of the license agreement.
 *
 */

#ifndef _MINJUGG_H_
#define _MINJUGG_H_

#include <libinac/lib.h>
#include <libiarray/iarray.h>

typedef struct jug_context_s jug_context_t;
typedef struct jug_expression_s jug_expression_t;
typedef struct jug_udf_s jug_udf_t;

INA_API(ina_rc_t) jug_init(void);
INA_API(void) jug_destroy(void);

INA_API(ina_rc_t) jug_expression_new(jug_expression_t **expr);
INA_API(void) jug_expression_free(jug_expression_t **expr);
INA_API(ina_rc_t) jug_expression_compile(jug_expression_t *e,
    const char *expr, int num_vars, void *vars, int32_t typesize, uint64_t *function_addr);

INA_API(ina_rc_t) jug_udf_compile(jug_expression_t *e,
                                  int llvm_bc_len,
                                  const char *llvm_bc,
                                  const char *name,
                                  uint64_t *function_addr);

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
    EXPR_TYPE_LN,
    EXPR_TYPE_LOG,
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
    EXPR_TYPE_CUSTOM
} te_expr_type_t;
typedef struct jug_te_variable {
    const char *name;
    te_expr_type_t address;
    int type;
    void *context;
} jug_te_variable;

#endif

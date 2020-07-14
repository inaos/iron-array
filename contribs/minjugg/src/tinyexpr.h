/*
 * TINYEXPR - Tiny recursive descent parser and evaluation engine in C
 *
 * Copyright (c) 2015-2018 Lewis Van Winkle
 *
 * http://CodePlea.com
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 * claim that you wrote the original software. If you use this software
 * in a product, an acknowledgement in the product documentation would be
 * appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 * misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

#ifndef __JUG_TINYEXPR_H__
#define __JUG_TINYEXPR_H__


#ifdef __cplusplus
extern "C" {
#endif

typedef struct jug_te_expr {
    int type;
    union {double value; const char *bound; te_expr_type_t function;};
    void *parameters[1];
} jug_te_expr;


enum {
    TE_VARIABLE = 0,

    TE_FUNCTION0 = 8, TE_FUNCTION1, TE_FUNCTION2, TE_FUNCTION3,
    TE_FUNCTION4, TE_FUNCTION5, TE_FUNCTION6, TE_FUNCTION7,

    TE_CLOSURE0 = 16, TE_CLOSURE1, TE_CLOSURE2, TE_CLOSURE3,
    TE_CLOSURE4, TE_CLOSURE5, TE_CLOSURE6, TE_CLOSURE7,

    TE_FLAG_PURE = 32
};

enum { TE_CONSTANT = 1 };

static const char te_function_map_str[][32] = {
    "EXPR_TYPE_ADD",
    "EXPR_TYPE_SUB",
    "EXPR_TYPE_MUL",
    "EXPR_TYPE_DIVIDE",
    "EXPR_TYPE_NEGATE",
    "EXPR_TYPE_COMMA",
    "EXPR_TYPE_ABS",
    "EXPR_TYPE_ACOS",
    "EXPR_TYPE_ASIN",
    "EXPR_TYPE_ATAN",
    "EXPR_TYPE_ATAN2",
    "EXPR_TYPE_CEIL",
    "EXPR_TYPE_COS",
    "EXPR_TYPE_COSH",
    "EXPR_TYPE_E",
    "EXPR_TYPE_EXP",
    "EXPR_TYPE_FAC",
    "EXPR_TYPE_FLOOR",
    "EXPR_TYPE_LOG",
    "EXPR_TYPE_LOG10",
    "EXPR_TYPE_NCR",
    "EXPR_TYPE_NPR",
    "EXPR_TYPE_PI",
    "EXPR_TYPE_POW",
    "EXPR_TYPE_SIN",
    "EXPR_TYPE_SINH",
    "EXPR_TYPE_SQRT",
    "EXPR_TYPE_TAN",
    "EXPR_TYPE_TANH",
    "EXPR_TYPE_FMOD"
};

/* Parses the input expression and binds variables. */
/* Returns NULL on error. */
jug_te_expr *jug_te_compile(const char *expression, const jug_te_variable *variables, int var_count, int *error);


/* Frees the expression. */
/* This is safe to call on NULL pointers. */
void jug_te_free(jug_te_expr *n);


#ifdef __cplusplus
}
#endif

#endif /*__TINYEXPR_H__*/

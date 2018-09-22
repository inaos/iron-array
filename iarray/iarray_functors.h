#ifndef IARRAY_FUNCTORS_H_
#define IARRAY_FUNCTORS_H_

#include "iarray.h"
#include "iarray_private.h"

typedef ina_rc_t (*iarray_op_add)(iarray_expression_t *expr, iarray_temporary_t *lhs, iarray_temporary_t *rhs, iarray_temporary_t **out);
typedef ina_rc_t (*iarray_op_sub)(iarray_expression_t *ctx, iarray_temporary_t *lhs, iarray_temporary_t *rhs, iarray_temporary_t **out);
typedef ina_rc_t (*iarray_op_mul)(iarray_expression_t *ctx, iarray_temporary_t *lhs, iarray_temporary_t *rhs, iarray_temporary_t **out);
typedef ina_rc_t (*iarray_op_div)(iarray_expression_t *ctx, iarray_temporary_t *lhs, iarray_temporary_t *rhs, iarray_temporary_t **out);



typedef ina_rc_t (*iarray_op_negate)(iarray_expression_t *ctx, iarray_temporary_t *x, iarray_temporary_t **out);
typedef ina_rc_t (*iarray_op_comma)(iarray_expression_t *ctx, iarray_temporary_t *lhs, iarray_temporary_t *rhs, iarray_temporary_t **out);

typedef ina_rc_t (*iarray_op_abs)(iarray_expression_t *ctx, iarray_temporary_t *x, iarray_temporary_t **out);
typedef ina_rc_t (*iarray_op_acos)(iarray_expression_t *ctx, iarray_temporary_t *x, iarray_temporary_t **out);
typedef ina_rc_t (*iarray_op_asin)(iarray_expression_t *ctx, iarray_temporary_t *x, iarray_temporary_t **out);
typedef ina_rc_t (*iarray_op_atan)(iarray_expression_t *ctx, iarray_temporary_t *x, iarray_temporary_t **out);
typedef ina_rc_t (*iarray_op_atan2)(iarray_expression_t *ctx, iarray_temporary_t *x, iarray_temporary_t **out);
typedef ina_rc_t (*iarray_op_ceil)(iarray_expression_t *ctx, iarray_temporary_t *x, iarray_temporary_t **out);
typedef ina_rc_t (*iarray_op_cos)(iarray_expression_t *ctx, iarray_temporary_t *x, iarray_temporary_t **out);
typedef ina_rc_t (*iarray_op_cosh)(iarray_expression_t *ctx, iarray_temporary_t *x, iarray_temporary_t **out);
typedef ina_rc_t (*iarray_op_e)(iarray_expression_t *ctx, iarray_temporary_t *x, iarray_temporary_t **out);
typedef ina_rc_t (*iarray_op_exp)(iarray_expression_t *ctx, iarray_temporary_t *x, iarray_temporary_t **out);
typedef ina_rc_t (*iarray_op_fac)(iarray_expression_t *ctx, iarray_temporary_t *x, iarray_temporary_t **out);
typedef ina_rc_t (*iarray_op_floor)(iarray_expression_t *ctx, iarray_temporary_t *x, iarray_temporary_t **out);
typedef ina_rc_t (*iarray_op_ln)(iarray_expression_t *ctx, iarray_temporary_t *x, iarray_temporary_t **out);
typedef ina_rc_t (*iarray_op_log)(iarray_expression_t *ctx, iarray_temporary_t *x, iarray_temporary_t **out);
typedef ina_rc_t (*iarray_op_log10)(iarray_expression_t *ctx, iarray_temporary_t *x, iarray_temporary_t **out);
typedef ina_rc_t (*iarray_op_pi)(iarray_expression_t *ctx, iarray_temporary_t *x, iarray_temporary_t **out);
typedef ina_rc_t (*iarray_op_pow)(iarray_expression_t *ctx, iarray_temporary_t *b, iarray_temporary_t *e);
typedef ina_rc_t (*iarray_op_sin)(iarray_expression_t *ctx, iarray_temporary_t *x, iarray_temporary_t **out);
typedef ina_rc_t (*iarray_op_sinh)(iarray_expression_t *ctx, iarray_temporary_t *x, iarray_temporary_t **out);
typedef ina_rc_t (*iarray_op_sqrt)(iarray_expression_t *ctx, iarray_temporary_t *x, iarray_temporary_t **out);
typedef ina_rc_t (*iarray_op_tan)(iarray_expression_t *ctx, iarray_temporary_t *x, iarray_temporary_t **out);
typedef ina_rc_t (*iarray_op_tanh)(iarray_expression_t *ctx, iarray_temporary_t *x, iarray_temporary_t **out);

typedef ina_rc_t (*iarray_op_erf)(iarray_expression_t *ctx, iarray_temporary_t *x, iarray_temporary_t **out);
typedef ina_rc_t (*iarray_op_erfc)(iarray_expression_t *ctx, iarray_temporary_t *x, iarray_temporary_t **out);

typedef ina_rc_t(*iarray_op_matmul)(iarray_expression_t *ctx, iarray_temporary_t *lhs, iarray_temporary_t *rhs, iarray_temporary_t **out);
typedef ina_rc_t(*iarray_op_hemm)(iarray_expression_t *ctx, iarray_temporary_t *lhs, iarray_temporary_t *rhs, iarray_temporary_t **out);
typedef ina_rc_t(*iarray_op_symm)(iarray_expression_t *ctx, iarray_temporary_t *lhs, iarray_temporary_t *rhs, iarray_temporary_t **out);
typedef ina_rc_t(*iarray_op_trmm)(iarray_expression_t *ctx, iarray_temporary_t *lhs, iarray_temporary_t *rhs, iarray_temporary_t **out);

/* reductions */
typedef ina_rc_t (*iarray_op_sum)(iarray_expression_t *ctx, iarray_temporary_t *x, iarray_temporary_t **out);

#endif
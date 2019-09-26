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

/* COMPILE TIME OPTIONS */

/* Exponentiation associativity:
For a^b^c = (a^b)^c and -a^b = (-a)^b do nothing.
For a^b^c = a^(b^c) and -a^b = -(a^b) uncomment the next line.*/
/* #define TE_POW_FROM_RIGHT */

/* Logarithms
For log = base 10 log do nothing
For log = natural log uncomment the next line. */
/* #define TE_NAT_LOG */

#include <libiarray/iarray.h>
#include "tinyexpr.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#if defined(_OPENMP)
#include <omp.h>
#endif


#ifndef NAN
#define NAN (0.0/0.0)
#endif

#ifndef INFINITY
#define INFINITY (1.0/0.0)
#endif


typedef double (*te_fun2)(double, double);

enum {
    TOK_NULL = TE_CLOSURE7+1, TOK_ERROR, TOK_END, TOK_SEP,
    TOK_OPEN, TOK_CLOSE, TOK_NUMBER, TOK_VARIABLE, TOK_INFIX
};


enum {TE_CONSTANT = 1};


INA_DISABLE_WARNING_MSVC(4201); /* Since this is TinyExpr code we do not properly fix it */
typedef struct state {
    const char *start;
    const char *next;
    int type;
    int nthreads;
	double scalar;
    union {iarray_temporary_t *value; const void **bound; const void *function;};
    void *context;
	iarray_expression_t *expr;
    const te_variable *lookup;
    int lookup_len;
} state;
INA_ENABLE_WARNING_MSVC(4201);

#define TYPE_MASK(TYPE) ((TYPE)&0x0000001F)

#define IS_PURE(TYPE) (((TYPE) & TE_FLAG_PURE) != 0)
#define IS_FUNCTION(TYPE) (((TYPE) & TE_FUNCTION0) != 0)
#define IS_CLOSURE(TYPE) (((TYPE) & TE_CLOSURE0) != 0)
#define ARITY(TYPE) ( ((TYPE) & (TE_FUNCTION0 | TE_CLOSURE0)) ? ((TYPE) & 0x00000007) : 0 )
#define NEW_EXPR(state, type, ...) new_expr((state), (type), (const te_expr*[]){__VA_ARGS__})

static te_expr *new_expr(state *state, const int type, const te_expr *parameters[]) {
    const int arity = ARITY(type);
    const int psize = sizeof(void*) * arity;
    const int size = (sizeof(te_expr) - sizeof(void*)) + psize + (IS_CLOSURE(type) ? sizeof(void*) : 0);
    //te_expr *ret = malloc(size);
	ina_mempool_t *mp;
	iarray_expr_get_mp(state->expr, &mp);
    iarray_expr_get_nthreads(state->expr, &state->nthreads);
    te_expr *ret = ina_mempool_dalloc(mp, (size_t)size);
    memset(ret, 0, size);
    if (arity && parameters) {
        memcpy(ret->parameters, parameters, psize);
    }
    ret->type = type;
    ret->bound = 0;
    return ret;
}


void te_free_parameters(te_expr *n) {
    if (!n) return;
    switch (TYPE_MASK(n->type)) {
        case TE_FUNCTION7: case TE_CLOSURE7: te_free(n->parameters[6]);     /* Falls through. */
        case TE_FUNCTION6: case TE_CLOSURE6: te_free(n->parameters[5]);     /* Falls through. */
        case TE_FUNCTION5: case TE_CLOSURE5: te_free(n->parameters[4]);     /* Falls through. */
        case TE_FUNCTION4: case TE_CLOSURE4: te_free(n->parameters[3]);     /* Falls through. */
        case TE_FUNCTION3: case TE_CLOSURE3: te_free(n->parameters[2]);     /* Falls through. */
        case TE_FUNCTION2: case TE_CLOSURE2: te_free(n->parameters[1]);     /* Falls through. */
        case TE_FUNCTION1: case TE_CLOSURE1: te_free(n->parameters[0]);
    }
}


void te_free(te_expr *n) {
    if (!n) return;
    te_free_parameters(n);
    //free(n);
}


static double pi(void) {return 3.14159265358979323846;}
static double e(void) {return 2.71828182845904523536;}
static double fac(double a) {/* simplest version of fac */
    if (a < 0.0)
        return NAN;
    if (a > UINT_MAX)
        return INFINITY;
    unsigned int ua = (unsigned int)(a);
    unsigned long int result = 1, i;
    for (i = 1; i <= ua; i++) {
        if (i > ULONG_MAX / result)
            return INFINITY;
        result *= i;
    }
    return (double)result;
}
static double ncr(double n, double r) {
    if (n < 0.0 || r < 0.0 || n < r) return NAN;
    if (n > UINT_MAX || r > UINT_MAX) return INFINITY;
    unsigned long int un = (unsigned int)(n), ur = (unsigned int)(r), i;
    unsigned long int result = 1;
    if (ur > un / 2) ur = un - ur;
    for (i = 1; i <= ur; i++) {
        if (result > ULONG_MAX / (un - ur + i))
            return INFINITY;
        result *= un - ur + i;
        result /= i;
    }
    return result;
}
static double npr(double n, double r) {return ncr(n, r) * fac(r);}

/* Functions */

iarray_temporary_t* func_cos(iarray_expression_t *expr, iarray_temporary_t *operand)
{
    return _iarray_func(expr, operand, IARRAY_FUNC_COS);
}

iarray_temporary_t* func_sin(iarray_expression_t *expr, iarray_temporary_t *operand)
{
    return _iarray_func(expr, operand, IARRAY_FUNC_SIN);
}

iarray_temporary_t* func_tan(iarray_expression_t *expr, iarray_temporary_t *operand)
{
    return _iarray_func(expr, operand, IARRAY_FUNC_TAN);
}

iarray_temporary_t* func_cosh(iarray_expression_t *expr, iarray_temporary_t *operand)
{
    return _iarray_func(expr, operand, IARRAY_FUNC_COSH);
}

iarray_temporary_t* func_sinh(iarray_expression_t *expr, iarray_temporary_t *operand)
{
    return _iarray_func(expr, operand, IARRAY_FUNC_SINH);
}

iarray_temporary_t* func_tanh(iarray_expression_t *expr, iarray_temporary_t *operand)
{
    return _iarray_func(expr, operand, IARRAY_FUNC_TANH);
}


INA_DISABLE_WARNING_MSVC(4152);
static const te_variable functions[] = {
    /* must be in alphabetical order */
    {"abs", NULL, NULL,     TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"acos", NULL, NULL,    TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"asin", NULL, NULL,    TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"atan", NULL, NULL,    TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"atan2", NULL, NULL,  TE_FUNCTION2 | TE_FLAG_PURE, 0},
//    {"ceil", NULL, ceil,    TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"cos", NULL, func_cos,      TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"cosh", NULL, func_cosh,    TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"e", NULL, e,          TE_FUNCTION0 | TE_FLAG_PURE, 0},
    {"exp", NULL, NULL,      TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"fac", NULL, NULL,      TE_FUNCTION1 | TE_FLAG_PURE, 0},
//    {"floor", floor,  TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"ln", NULL, NULL,       TE_FUNCTION1 | TE_FLAG_PURE, 0},
#ifdef TE_NAT_LOG
    {"log", NULL, NULL,      TE_FUNCTION1 | TE_FLAG_PURE, 0},
#else
    {"log", NULL, NULL,    TE_FUNCTION1 | TE_FLAG_PURE, 0},
#endif
    {"log10", NULL, log10,  TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"ncr", NULL, NULL,      TE_FUNCTION2 | TE_FLAG_PURE, 0},
    {"npr", NULL, NULL,      TE_FUNCTION2 | TE_FLAG_PURE, 0},
    {"pi", NULL, pi,        TE_FUNCTION0 | TE_FLAG_PURE, 0},
    {"pow", NULL, NULL,      TE_FUNCTION2 | TE_FLAG_PURE, 0},
    {"sin", NULL, func_sin,      TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"sinh", NULL, func_sinh,    TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"sqrt", NULL, NULL,    TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"tan", NULL, func_tan,      TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"tanh", NULL, func_tanh,    TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {0, 0, 0, 0, 0}
};
INA_ENABLE_WARNING_MSVC(4152);

static const te_variable *find_builtin(const char *name, size_t len) {
    int imin = 0;
    int imax = sizeof(functions) / sizeof(te_variable) - 2;

    /*Binary search.*/
    while (imax >= imin) {
        const int i = (imin + ((imax-imin)/2));
        int c = strncmp(name, functions[i].name, len);
        if (!c) c = '\0' - functions[i].name[len];
        if (c == 0) {
            return functions + i;
        } else if (c > 0) {
            imin = i + 1;
        } else {
            imax = i - 1;
        }
    }

    return 0;
}

static const te_variable *find_lookup(const state *s, const char *name, size_t len) {
    int iters;
    const te_variable *var;
    if (!s->lookup) return 0;

    for (var = s->lookup, iters = s->lookup_len; iters; ++var, --iters) {
        if (strncmp(name, var->name, len) == 0 && var->name[len] == '\0') {
            return var;
        }
    }
    return 0;
}


#define add _iarray_op_add
#define sub _iarray_op_sub
#define mul _iarray_op_mul
#define divide _iarray_op_divide
//static double add(double a, double b) {return a + b;}
//static double sub(double a, double b) {return a - b;}
//static double mul(double a, double b) {return a * b;}
//static double divide(double a, double b) {return a / b;}
static double negate(double a) {return -a;}
static double comma(double a, double b) {(void)a; return b;}


void next_token(state *s) {
    s->type = TOK_NULL;

    do {

        if (!*s->next){
            s->type = TOK_END;
            return;
        }

        /* Try reading a number. */
        if ((s->next[0] >= '0' && s->next[0] <= '9') || s->next[0] == '.') {
            s->scalar = strtod(s->next, (char**)&s->next);
            s->type = TOK_NUMBER;
        } else {
            /* Look for a variable or builtin function call. */
            if (s->next[0] >= 'a' && s->next[0] <= 'z') {
                const char *start;
                start = s->next;
                while ((s->next[0] >= 'a' && s->next[0] <= 'z') || (s->next[0] >= '0' && s->next[0] <= '9') || (s->next[0] == '_')) s->next++;

                const te_variable *var = find_lookup(s, start, s->next - start);
                if (!var) var = find_builtin(start, s->next - start);

                if (!var) {
                    s->type = TOK_ERROR;
                } else {
                    switch(TYPE_MASK(var->type))
                    {
                        case TE_VARIABLE:
                            s->type = TOK_VARIABLE;
                            s->bound = var->address;
                            break;

                        case TE_CLOSURE0: case TE_CLOSURE1: case TE_CLOSURE2: case TE_CLOSURE3:         /* Falls through. */
                        case TE_CLOSURE4: case TE_CLOSURE5: case TE_CLOSURE6: case TE_CLOSURE7:         /* Falls through. */
                            s->context = var->context;                                                  /* Falls through. */

                        case TE_FUNCTION0: case TE_FUNCTION1: case TE_FUNCTION2: case TE_FUNCTION3:     /* Falls through. */
                        case TE_FUNCTION4: case TE_FUNCTION5: case TE_FUNCTION6: case TE_FUNCTION7:     /* Falls through. */
                            s->type = var->type;
                            s->function = var->function;
                            if (s->function == NULL) {
                                printf("Undefined function: '%s'\n", var->name);
                            }
                            break;
                        default:
                            printf("Unknown type; cannot never happen.  If you see this, inform about it.\n");
                            break;
                    }
                }

            } else {
                /* Look for an operator or special character. */
                INA_DISABLE_WARNING_MSVC(4152);
                switch (s->next++[0]) {
                    case '+': s->type = TOK_INFIX; s->function = add; break;
                    case '-': s->type = TOK_INFIX; s->function = sub; break;
                    case '*': s->type = TOK_INFIX; s->function = mul; break;
                    case '/': s->type = TOK_INFIX; s->function = divide; break;
                    case '^': s->type = TOK_INFIX; s->function = pow; break;
                    case '%': s->type = TOK_INFIX; s->function = fmod; break;
                    case '(': s->type = TOK_OPEN; break;
                    case ')': s->type = TOK_CLOSE; break;
                    case ',': s->type = TOK_SEP; break;
                    case ' ': case '\t': case '\n': case '\r': break;
                    default: s->type = TOK_ERROR; break;
                }
                INA_ENABLE_WARNING_MSVC(4152);
            }
        }
    } while (s->type == TOK_NULL);
}


static te_expr *list(state *s);
static te_expr *expr(state *s);
static te_expr *power(state *s);

static te_expr *base(state *s) {
    /* <base>      =    <constant> | <variable> | <function-0> {"(" ")"} | <function-1> <power> | <function-X> "(" <expr> {"," <expr>} ")" | "(" <list> ")" */
    te_expr *ret;
    int arity;

    switch (TYPE_MASK(s->type)) {
        case TOK_NUMBER:
            ret = new_expr(s, TE_CONSTANT, 0);
			ina_mempool_t *mp;
			iarray_expr_get_mp(s->expr, &mp);
			ret->value = ina_mempool_dalloc(mp, sizeof(iarray_temporary_t));  /* FIXME: for now we have to allocate a scalar for every chunk */
			//ret->value = ina_mem_alloc(sizeof(iarray_temporary_t));
			memset(ret->value, 0, sizeof(iarray_temporary_t));
			// Make this an actual scalar
			iarray_dtshape_t sshape = {
					.ndim = 0,
					.dtype = IARRAY_DATA_TYPE_DOUBLE,
			};
			ret->value->dtshape = ina_mempool_dalloc(mp, sizeof(iarray_dtshape_t));
			//ret->value->dtshape = ina_mem_alloc(sizeof(iarray_dtshape_t));
			memcpy(ret->value->dtshape, &sshape, sizeof(iarray_dtshape_t));
            ret->value->scalar_value.d = s->scalar;
            //ret->value->scalar_value.f = (float)s->scalar;
            next_token(s);
            break;

        case TOK_VARIABLE:
            ret = new_expr(s, TE_VARIABLE, 0);
            ret->bound = s->bound;
            next_token(s);
            break;

        case TE_FUNCTION0:
        case TE_CLOSURE0:
            ret = new_expr(s, s->type, 0);
            ret->function = s->function;
            if (IS_CLOSURE(s->type)) ret->parameters[0] = s->context;
            next_token(s);
            if (s->type == TOK_OPEN) {
                next_token(s);
                if (s->type != TOK_CLOSE) {
                    s->type = TOK_ERROR;
                } else {
                    next_token(s);
                }
            }
            break;

        case TE_FUNCTION1:
        case TE_CLOSURE1:
            ret = new_expr(s, s->type, 0);
            ret->function = s->function;
            if (IS_CLOSURE(s->type)) ret->parameters[1] = s->context;
            next_token(s);
            ret->parameters[0] = power(s);
            break;

        case TE_FUNCTION2: case TE_FUNCTION3: case TE_FUNCTION4:
        case TE_FUNCTION5: case TE_FUNCTION6: case TE_FUNCTION7:
        case TE_CLOSURE2: case TE_CLOSURE3: case TE_CLOSURE4:
        case TE_CLOSURE5: case TE_CLOSURE6: case TE_CLOSURE7:
            arity = ARITY(s->type);

            ret = new_expr(s, s->type, 0);
            ret->function = s->function;
            if (IS_CLOSURE(s->type)) ret->parameters[arity] = s->context;
            next_token(s);

            if (s->type != TOK_OPEN) {
                s->type = TOK_ERROR;
            } else {
                int i;
                for(i = 0; i < arity; i++) {
                    next_token(s);
                    ret->parameters[i] = expr(s);
                    if(s->type != TOK_SEP) {
                        break;
                    }
                }
                if(s->type != TOK_CLOSE || i != arity - 1) {
                    s->type = TOK_ERROR;
                } else {
                    next_token(s);
                }
            }

            break;

        case TOK_OPEN:
            next_token(s);
            ret = list(s);
            if (s->type != TOK_CLOSE) {
                s->type = TOK_ERROR;
            } else {
                next_token(s);
            }
            break;

        default:
            ret = new_expr(s, 0, 0);
            s->type = TOK_ERROR;
            ret->value = NULL;
            break;
    }

    return ret;
}

INA_DISABLE_WARNING_MSVC(4152);
INA_DISABLE_WARNING_MSVC(4204);
static te_expr *power(state *s) {
    /* <power>     =    {("-" | "+")} <base> */
    int sign = 1;
    while (s->type == TOK_INFIX && (s->function == add || s->function == sub)) {
        if (s->function == sub) sign = -sign;
        next_token(s);
    }

    te_expr *ret;

    if (sign == 1) {
        ret = base(s);
    } else {
        ret = NEW_EXPR(s, TE_FUNCTION1 | TE_FLAG_PURE, base(s));
        ret->function = negate;
    }

    return ret;
}

#ifdef TE_POW_FROM_RIGHT
static te_expr *factor(state *s) {
    /* <factor>    =    <power> {"^" <power>} */
    te_expr *ret = power(s);

    int neg = 0;
    te_expr *insertion = 0;

    if (ret->type == (TE_FUNCTION1 | TE_FLAG_PURE) && ret->function == negate) {
        te_expr *se = ret->parameters[0];
        //free(ret);
        ret = se;
        neg = 1;
    }

    while (s->type == TOK_INFIX && (s->function == pow)) {
        te_fun2 t = s->function;
        next_token(s);

        if (insertion) {
            /* Make exponentiation go right-to-left. */
            te_expr *insert = NEW_EXPR(s, TE_FUNCTION2 | TE_FLAG_PURE, insertion->parameters[1], power(s));
            insert->function = t;
            insertion->parameters[1] = insert;
            insertion = insert;
        } else {
            ret = NEW_EXPR(s, TE_FUNCTION2 | TE_FLAG_PURE, ret, power(s));
            ret->function = t;
            insertion = ret;
        }
    }

    if (neg) {
        ret = NEW_EXPR(s, TE_FUNCTION1 | TE_FLAG_PURE, ret);
        ret->function = negate;
    }

    return ret;
}
#else
static te_expr *factor(state *s) {
    /* <factor>    =    <power> {"^" <power>} */
    te_expr *ret = power(s);

    while (s->type == TOK_INFIX && (s->function == pow)) {
        te_fun2 t = (te_fun2)s->function;
        next_token(s);
        ret = NEW_EXPR(s, TE_FUNCTION2 | TE_FLAG_PURE, ret, power(s));
        ret->function = t;
    }

    return ret;
}
#endif



static te_expr *term(state *s) {
    /* <term>      =    <factor> {("*" | "/" | "%") <factor>} */
    te_expr *ret = factor(s);

    while (s->type == TOK_INFIX && (s->function == mul || s->function == divide || s->function == fmod)) {
        te_fun2 t = (te_fun2)s->function;
        next_token(s);
        ret = NEW_EXPR(s, TE_FUNCTION2 | TE_FLAG_PURE, ret, factor(s));
        ret->function = t;
    }

    return ret;
}


static te_expr *expr(state *s) {
    /* <expr>      =    <term> {("+" | "-") <term>} */
    te_expr *ret = term(s);

    while (s->type == TOK_INFIX && (s->function == add || s->function == sub)) {
        te_fun2 t = (te_fun2)s->function;
        next_token(s);
        ret = NEW_EXPR(s, TE_FUNCTION2 | TE_FLAG_PURE, ret, term(s));
        ret->function = t;
    }

    return ret;
}


static te_expr *list(state *s) {
    /* <list>      =    <expr> {"," <expr>} */
    te_expr *ret = expr(s);

    while (s->type == TOK_SEP) {
        next_token(s);
        ret = NEW_EXPR(s, TE_FUNCTION2 | TE_FLAG_PURE, ret, expr(s));
        ret->function = comma;
    }

    return ret;
}
INA_ENABLE_WARNING_MSVC(4152);
INA_ENABLE_WARNING_MSVC(4204);

//#define TE_FUN(...) ((double(*)(__VA_ARGS__))n->function)
#define TE_FUN(...) ( (iarray_temporary_t*(*)(__VA_ARGS__))n->function )
#define M(e) te_eval(expr, n->parameters[e])


iarray_temporary_t *te_eval(iarray_expression_t *expr, const te_expr *n) {
    if (!n) return NULL;

    int nthread = 0;
#if defined(_OPENMP)
    nthread = omp_get_thread_num();
#endif
    //printf("nthread (te_eval): %d\n", nthread);

    switch(TYPE_MASK(n->type)) {
        case TE_CONSTANT: return n->value;
        case TE_VARIABLE: {
            iarray_temporary_t *iatemp = (iarray_temporary_t *) (n->bound[nthread]);
            return iatemp;
        }
        case TE_FUNCTION0: case TE_FUNCTION1: case TE_FUNCTION2: case TE_FUNCTION3:
        case TE_FUNCTION4: case TE_FUNCTION5: case TE_FUNCTION6: case TE_FUNCTION7:
            //printf("Arity: %d\n", ARITY(n->type));
            switch(ARITY(n->type)) {
                case 0: return TE_FUN(iarray_expression_t*)(expr);
                case 1: return TE_FUN(iarray_expression_t*, iarray_temporary_t*)(expr, M(0));
                case 2: return TE_FUN(iarray_expression_t*, iarray_temporary_t*, iarray_temporary_t*)(expr, M(0), M(1));
                case 3: return TE_FUN(iarray_expression_t*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*)(expr, M(0), M(1), M(2));
                case 4: return TE_FUN(iarray_expression_t*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*)(expr, M(0), M(1), M(2), M(3));
                case 5: return TE_FUN(iarray_expression_t*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*)(expr, M(0), M(1), M(2), M(3), M(4));
                case 6: return TE_FUN(iarray_expression_t*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*)(expr, M(0), M(1), M(2), M(3), M(4), M(5));
                case 7: return TE_FUN(iarray_expression_t*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*)(expr, M(0), M(1), M(2), M(3), M(4), M(5), M(6));
                default: return NULL;
            }

        case TE_CLOSURE0: case TE_CLOSURE1: case TE_CLOSURE2: case TE_CLOSURE3:
        case TE_CLOSURE4: case TE_CLOSURE5: case TE_CLOSURE6: case TE_CLOSURE7:
            switch(ARITY(n->type)) {
                case 0: return TE_FUN(void*)(n->parameters[0]);
                case 1: return TE_FUN(iarray_expression_t*, void*, iarray_temporary_t*)(expr, n->parameters[1], M(0));
                case 2: return TE_FUN(iarray_expression_t*, void*, iarray_temporary_t*, iarray_temporary_t*)(expr, n->parameters[2], M(0), M(1));
                case 3: return TE_FUN(iarray_expression_t*, void*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*)(expr, n->parameters[3], M(0), M(1), M(2));
                case 4: return TE_FUN(iarray_expression_t*, void*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*)(expr, n->parameters[4], M(0), M(1), M(2), M(3));
                case 5: return TE_FUN(iarray_expression_t*, void*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*)(expr, n->parameters[5], M(0), M(1), M(2), M(3), M(4));
                case 6: return TE_FUN(iarray_expression_t*, void*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*)(expr, n->parameters[6], M(0), M(1), M(2), M(3), M(4), M(5));
                case 7: return TE_FUN(iarray_expression_t*, void*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*, iarray_temporary_t*)(expr, n->parameters[7], M(0), M(1), M(2), M(3), M(4), M(5), M(6));
                default: return NULL;
            }

        default: return NULL;
    }

}

#undef TE_FUN
#undef M

static void optimize(iarray_expression_t *expr, te_expr *n) {
    /* Evaluates as much as possible. */
    if (n->type == TE_CONSTANT) return;
    if (n->type == TE_VARIABLE) return;

    /* Only optimize out functions flagged as pure. */
    if (IS_PURE(n->type)) {
        const int arity = ARITY(n->type);
        int known = 1;
        int i;
        for (i = 0; i < arity; ++i) {
            optimize(expr, n->parameters[i]);
            if (((te_expr*)(n->parameters[i]))->type != TE_CONSTANT) {
                known = 0;
            }
        }
        if (known) {
            const iarray_temporary_t *value = te_eval(expr, n);
            te_free_parameters(n);
            n->type = TE_CONSTANT;
            n->value = (iarray_temporary_t *)value;
        }
    }
}


te_expr *te_compile(iarray_expression_t *expr, const char *expression, const te_variable *variables, int var_count, int *error) {
    state s;
    s.start = s.next = expression;
    s.lookup = variables;
    s.lookup_len = var_count;
    s.expr = expr;

    next_token(&s);
    te_expr *root = list(&s);

    if (s.type != TOK_END) {
        te_free(root);
        if (error) {
            *error = (int)(s.next - s.start);
            if (*error == 0) *error = 1;
        }
        return 0;
    } else {
        optimize(expr, root);
        if (error) *error = 0;
        return root;
    }
}


iarray_temporary_t *te_interp(iarray_expression_t *expr, const char *expression, int *error) {
    te_expr *n = te_compile(expr, expression, 0, 0, error);
	iarray_temporary_t *ret;
    if (n) {
        ret = te_eval(expr, n);
        te_free(n);
    } else {
        ret = NULL;
    }
    return ret;
}

static void pn (const te_expr *n, int depth) {
    int i, arity;
    printf("%*s", depth, "");

    switch(TYPE_MASK(n->type)) {
    case TE_CONSTANT: printf("%f\n", n->value->scalar_value.d); break;
    case TE_VARIABLE: printf("bound %p\n", n->bound[0]); break;

    case TE_FUNCTION0: case TE_FUNCTION1: case TE_FUNCTION2: case TE_FUNCTION3:
    case TE_FUNCTION4: case TE_FUNCTION5: case TE_FUNCTION6: case TE_FUNCTION7:
    case TE_CLOSURE0: case TE_CLOSURE1: case TE_CLOSURE2: case TE_CLOSURE3:
    case TE_CLOSURE4: case TE_CLOSURE5: case TE_CLOSURE6: case TE_CLOSURE7:
         arity = ARITY(n->type);
         printf("f%d", arity);
         for(i = 0; i < arity; i++) {
             printf(" %p", n->parameters[i]);
         }
         printf("\n");
         for(i = 0; i < arity; i++) {
             pn(n->parameters[i], depth + 1);
         }
         break;
    default:
        printf("Unknown type; cannot never happen.  If you see this, inform about it.\n");
        break;
    }
}


void te_print(const te_expr *n) {
    pn(n, 0);
}

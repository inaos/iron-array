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

/*
 * MODIFICATIONS:
 *
 * Copyright ironArray SL 2021.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of ironArray SL
 * ("Confidential Information"). You shall not disclose such Confidential
 * Information and shall use it only in accordance with the terms of the license agreement.
 *
 */

#include <minjugg.h>
#include "tinyexpr.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>

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


typedef struct state {
    const char *start;
    const char *next;
    int type;
    union {double value; const char *bound; te_expr_type_t function;};
    void *context;

    const jug_te_variable *lookup;
    int lookup_len;

    jug_udf_registry_t *registry;
    ina_mempool_t *variable_mem_pool;
} state;


#define TYPE_MASK(TYPE) ((TYPE)&0x0000001F)

#define IS_PURE(TYPE) (((TYPE) & TE_FLAG_PURE) != 0)
#define IS_FUNCTION(TYPE) (((TYPE) & TE_FUNCTION0) != 0)
#define IS_CLOSURE(TYPE) (((TYPE) & TE_CLOSURE0) != 0)
#define ARITY(TYPE) ( ((TYPE) & (TE_FUNCTION0 | TE_CLOSURE0)) ? ((TYPE) & 0x00000007) : 0 )
#define NEW_EXPR(type, ...) new_expr((type), (const jug_te_expr*[]){__VA_ARGS__})

static jug_te_expr *new_expr(const int type, const jug_te_expr *parameters[]) {
    const int arity = ARITY(type);
    const int csize = sizeof(void*) * arity;
    const int size = (sizeof(jug_te_expr) - sizeof(void*)) + csize + (IS_CLOSURE(type) ? sizeof(void*) : 0);
    jug_te_expr *ret = malloc(size);
    memset(ret, 0, size);
    if (arity && parameters) {
        memcpy(ret->parameters, parameters, csize);
    }
    ret->type = type;
    ret->bound = 0;

    return ret;
}


static void te_free_parameters(jug_te_expr *n) {
    if (!n) return;
    switch (TYPE_MASK(n->type)) {
        case TE_FUNCTION7: case TE_CLOSURE7: jug_te_free(n->parameters[6]);     /* Falls through. */
        case TE_FUNCTION6: case TE_CLOSURE6: jug_te_free(n->parameters[5]);     /* Falls through. */
        case TE_FUNCTION5: case TE_CLOSURE5: jug_te_free(n->parameters[4]);     /* Falls through. */
        case TE_FUNCTION4: case TE_CLOSURE4: jug_te_free(n->parameters[3]);     /* Falls through. */
        case TE_FUNCTION3: case TE_CLOSURE3: jug_te_free(n->parameters[2]);     /* Falls through. */
        case TE_FUNCTION2: case TE_CLOSURE2: jug_te_free(n->parameters[1]);     /* Falls through. */
        case TE_FUNCTION1: case TE_CLOSURE1: jug_te_free(n->parameters[0]);
    }
}


void jug_te_free(jug_te_expr *n) {
    if (!n) return;
    te_free_parameters(n);
    free(n);
}

static const jug_te_variable functions[] = {
    /* must be in alphabetical order */
    {"abs", EXPR_TYPE_ABS,     TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"absolute", EXPR_TYPE_ABS,     TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"acos", EXPR_TYPE_ACOS,    TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"arccos", EXPR_TYPE_ACOS,    TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"arcsin", EXPR_TYPE_ASIN,    TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"arctan", EXPR_TYPE_ATAN,    TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"arctan2", EXPR_TYPE_ATAN2,  TE_FUNCTION2 | TE_FLAG_PURE, 0},
    {"asin", EXPR_TYPE_ASIN,    TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"atan", EXPR_TYPE_ATAN,    TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"atan2", EXPR_TYPE_ATAN2,  TE_FUNCTION2 | TE_FLAG_PURE, 0},
    {"ceil", EXPR_TYPE_CEIL,    TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"cos", EXPR_TYPE_COS,      TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"cosh", EXPR_TYPE_COSH,    TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"exp", EXPR_TYPE_EXP,      TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"floor", EXPR_TYPE_FLOOR,  TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"log", EXPR_TYPE_LOG,    TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"log10", EXPR_TYPE_LOG10,  TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"max", EXPR_TYPE_MAX,  TE_FUNCTION2 | TE_FLAG_PURE, 0},
    {"min", EXPR_TYPE_MIN,  TE_FUNCTION2 | TE_FLAG_PURE, 0},
    {"negate", EXPR_TYPE_NEGATE,    TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"negative", EXPR_TYPE_NEGATE,    TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"pow", EXPR_TYPE_POW,      TE_FUNCTION2 | TE_FLAG_PURE, 0},
    {"power", EXPR_TYPE_POW,      TE_FUNCTION2 | TE_FLAG_PURE, 0},
    {"sin", EXPR_TYPE_SIN,      TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"sinh", EXPR_TYPE_SINH,    TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"sqrt", EXPR_TYPE_SQRT,    TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"tan", EXPR_TYPE_TAN,      TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {"tanh", EXPR_TYPE_TANH,    TE_FUNCTION1 | TE_FLAG_PURE, 0},
    {0, 0, 0, 0}
};

static const jug_te_variable *find_builtin(const char *name, int len) {
    int imin = 0;
    int imax = sizeof(functions) / sizeof(jug_te_variable) - 2;

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

static const jug_te_variable *find_lookup(const state *s, const char *name, int len) {
    int iters;
    const jug_te_variable *var;
    if (!s->lookup) return 0;

    for (var = s->lookup, iters = s->lookup_len; iters; ++var, --iters) {
        if (strncmp(name, var->name, len) == 0 && var->name[len] == '\0') {
            return var;
        }
    }
    return 0;
}

static const jug_te_variable* find_custom(const state *s, const char *name, int len) {
    INA_UNUSED(len);

    jug_udf_function_t *cf = NULL;

    if (INA_FAILED(jug_udf_library_lookup_function(s->registry, name, &cf))) {
        return 0;
    }

    jug_te_variable *v = (jug_te_variable *) ina_mempool_dalloc(s->variable_mem_pool, sizeof(jug_te_variable));
    v->name = name;
    v->address = EXPR_TYPE_CUSTOM;
    v->type = TE_CUSTOM;
    v->context = cf;

    return v;
}

static void next_token(state *s) {
    s->type = TOK_NULL;

    do {

        if (!*s->next){
            s->type = TOK_END;
            return;
        }

        /* Try reading a number. */
        if ((s->next[0] >= '0' && s->next[0] <= '9') || s->next[0] == '.') {
            s->value = strtod(s->next, (char**)&s->next);
            s->type = TOK_NUMBER;
        } else {
            /* Look for a variable or builtin function call. */
            if (s->next[0] >= 'a' && s->next[0] <= 'z') {
                const char *start;
                start = s->next;
                while ((s->next[0] >= 'a' && s->next[0] <= 'z') || (s->next[0] >= '0' && s->next[0] <= '9') || (s->next[0] == '_') || (s->next[0] == '.')) s->next++;

                const jug_te_variable *var = find_lookup(s, start, (int) (s->next - start));
                if (!var) var = find_builtin(start, (int) (s->next - start));
                if (!var) var = find_custom(s, start, (int) (s->next - start));

                if (!var) {
                    s->type = TOK_ERROR;
                } else {
                    if (var->type == TE_CUSTOM) {
                        s->type = var->type;
                        s->function = var->address;
                        s->context = var->context;
                    } 
                    else {
                        switch (TYPE_MASK(var->type)) {
                            case TE_VARIABLE:
                                s->type = TOK_VARIABLE;
                                s->bound = var->name;
                                break;

                            case TE_CLOSURE0:
                            case TE_CLOSURE1:
                            case TE_CLOSURE2:
                            case TE_CLOSURE3:
                            case TE_CLOSURE4:
                            case TE_CLOSURE5:
                            case TE_CLOSURE6:
                            case TE_CLOSURE7:
                                s->context = var->context;

                            case TE_FUNCTION0:
                            case TE_FUNCTION1:
                            case TE_FUNCTION2:
                            case TE_FUNCTION3:
                            case TE_FUNCTION4:
                            case TE_FUNCTION5:
                            case TE_FUNCTION6:
                            case TE_FUNCTION7:
                                s->type = var->type;
                                s->function = var->address;
                                break;
                        }
                    }
                }

            } else {
                /* Look for an operator or special character. */
                switch (s->next++[0]) {
                    case '+': s->type = TOK_INFIX; s->function = EXPR_TYPE_ADD; break;
                    case '-': s->type = TOK_INFIX; s->function = EXPR_TYPE_SUB; break;
                    case '*': {
                        s->type = TOK_INFIX;
                        if (s->next[0] == '*') {
                            // pow can also be expressed as '**'
                            s->next++;
                            s->function = EXPR_TYPE_POW;
                            break;
                        }
                        s->function = EXPR_TYPE_MUL;
                        break;
                    }
                    case '/': s->type = TOK_INFIX; s->function = EXPR_TYPE_DIVIDE; break;
                    case '^': s->type = TOK_INFIX; s->function = EXPR_TYPE_POW; break;
                    case '%': s->type = TOK_INFIX; s->function = EXPR_TYPE_FMOD; break;
                    case '(': s->type = TOK_OPEN; break;
                    case ')': s->type = TOK_CLOSE; break;
                    case ',': s->type = TOK_SEP; break;
                    case ' ': case '\t': case '\n': case '\r': break;
                    default: s->type = TOK_ERROR; break;
                }
            }
        }
    } while (s->type == TOK_NULL);
}

static jug_te_expr *list(state *s);
static jug_te_expr *expr(state *s);
static jug_te_expr *power(state *s);

static jug_te_expr *base(state *s) {
    /* <base>      =    <constant> | <variable> | <function-0> {"(" ")"} | <function-1> <power> | <function-X> "(" <expr> {"," <expr>} ")" | "(" <list> ")" */
    jug_te_expr *ret;
    int arity;

    if (s->type == TE_CUSTOM) {
        if (s->context == NULL) {
            s->type = TOK_ERROR;
            return ret;
        }
        jug_udf_function_t *udf_fun = (jug_udf_function_t*) s->context;
        int cust_arity = jug_udf_function_get_arity(udf_fun);

        ret = new_expr(s->type, 0);
        ret->function = s->function;
        ret->parameters[0] = s->context;
        next_token(s);

        if (s->type != TOK_OPEN) {
            s->type = TOK_ERROR;
        } else {
            int i;
            for (i = 1; i < cust_arity + 1; i++) {
                next_token(s);
                ret->parameters[i] = expr(s);
                if (s->type != TOK_SEP) {
                    break;
                }
            }
            if (s->type != TOK_CLOSE || i != cust_arity) {
                s->type = TOK_ERROR;
            } else {
                next_token(s);
            }
        }
        return ret;
    }

    switch (TYPE_MASK(s->type)) {
        case TOK_NUMBER:
            ret = new_expr(TE_CONSTANT, 0);
            ret->value = s->value;
            next_token(s);
            break;

        case TOK_VARIABLE:
            ret = new_expr(TE_VARIABLE, 0);
            ret->bound = s->bound;
            next_token(s);
            break;

        case TE_FUNCTION0:
        case TE_CLOSURE0:
            ret = new_expr(s->type, 0);
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
            ret = new_expr(s->type, 0);
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

            ret = new_expr(s->type, 0);
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
            ret = new_expr(0, 0);
            s->type = TOK_ERROR;
            ret->value = NAN;
            break;
    }

    return ret;
}


static jug_te_expr *power(state *s) {
    /* <power>     =    {("-" | "+")} <base> */
    int sign = 1;
    while (s->type == TOK_INFIX && (s->function == EXPR_TYPE_ADD || s->function == EXPR_TYPE_SUB)) {
        if (s->function == EXPR_TYPE_SUB) sign = -sign;
        next_token(s);
    }

    jug_te_expr *ret;

    if (sign == 1) {
        ret = base(s);
    } else {
        ret = NEW_EXPR(TE_FUNCTION1 | TE_FLAG_PURE, base(s));
        ret->function = EXPR_TYPE_NEGATE;
    }

    return ret;
}

static jug_te_expr *factor(state *s) {
    /* <factor>    =    <power> {"^" <power>} */
    jug_te_expr *ret = power(s);

    while (s->type == TOK_INFIX && (s->function == EXPR_TYPE_POW)) {
        te_expr_type_t t = s->function;
        next_token(s);
        ret = NEW_EXPR(TE_FUNCTION2 | TE_FLAG_PURE, ret, power(s));
        ret->function = t;
    }

    return ret;
}

static jug_te_expr *term(state *s) {
    /* <term>      =    <factor> {("*" | "/" | "%") <factor>} */
    jug_te_expr *ret = factor(s);

    while (s->type == TOK_INFIX && (s->function == EXPR_TYPE_MUL || s->function == EXPR_TYPE_DIVIDE || s->function == EXPR_TYPE_FMOD)) {
        te_expr_type_t t = s->function;
        next_token(s);
        ret = NEW_EXPR(TE_FUNCTION2 | TE_FLAG_PURE, ret, factor(s));
        ret->function = t;
    }

    return ret;
}


static jug_te_expr *expr(state *s) {
    /* <expr>      =    <term> {("+" | "-") <term>} */
    jug_te_expr *ret = term(s);

    while (s->type == TOK_INFIX && (s->function == EXPR_TYPE_ADD || s->function == EXPR_TYPE_SUB)) {
        te_expr_type_t t = s->function;
        next_token(s);
        ret = NEW_EXPR(TE_FUNCTION2 | TE_FLAG_PURE, ret, term(s));
        ret->function = t;
    }

    return ret;
}


static jug_te_expr *list(state *s) {
    /* <list>      =    <expr> {"," <expr>} */
    jug_te_expr *ret = expr(s);

    while (s->type == TOK_SEP) {
        next_token(s);
        ret = NEW_EXPR(TE_FUNCTION2 | TE_FLAG_PURE, ret, expr(s));
        ret->function = EXPR_TYPE_COMMA;
    }

    return ret;
}

jug_te_expr *jug_te_compile(jug_udf_registry_t *registry, ina_mempool_t *variable_pool, const char *expression, const jug_te_variable *variables, int var_count, int *error) {
    state s;
    s.start = s.next = expression;
    s.lookup = variables;
    s.lookup_len = var_count;
    s.registry = registry;
    s.variable_mem_pool = variable_pool;

    next_token(&s);
    jug_te_expr *root = list(&s);

    if (s.type != TOK_END) {
        jug_te_free(root);
        if (error) {
            *error = (int) (s.next - s.start);
            if (*error == 0) *error = 1;
            int padding = *error + (int) strlen("Error at ");
            IARRAY_TRACE1(iarray.error, "Error at %s\n%*s^\n%*s%s%*s", expression, padding - 1, "",
                          padding - 1, "", "Error happens here", 10, "");
        }
        return 0;
    } else {
        if (error) *error = 0;
        return root;
    }
}


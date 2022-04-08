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

#include <minjugg.h>

#include <llvm-c/Core.h>
#include <llvm-c/ExecutionEngine.h>
#include <llvm-c/Target.h>
#include <llvm-c/Analysis.h>
#include <llvm-c/BitWriter.h>
#include <llvm-c/IRReader.h>

#include <llvm-c/Transforms/PassManagerBuilder.h>
#include <llvm-c/Transforms/Vectorize.h> // LLVMAddLoopVectorizePass

#include "minjuggutil.h"
#include "tinyexpr.h"

//#define _JUG_DEBUG_WRITE_BC_TO_FILE
//#define _JUG_DEBUG_WRITE_ERROR_TO_STDERR
//#define _JUG_DEBUG_DECLARE_PRINT_IN_IR

/* This is required to make sure Intel SVML is being linked and loaded properly */
extern double *__svml_sin2(double *input);

struct jug_expression_s {
    LLVMContextRef context;
    LLVMModuleRef mod;
    LLVMExecutionEngineRef engine;
    jug_expression_dtype_t dtype;
    ina_hashtable_t *fun_map;
    ina_hashtable_t *decl_cache;
    void **fun_map_te;
    LLVMBuilderRef builder;
    int32_t typesize;
    LLVMTypeRef expr_type;
};

static char *_jug_def_triple = NULL;
static LLVMTargetDataRef _jug_data_ref = NULL;
static LLVMTargetMachineRef _jug_tm_ref = NULL;

typedef LLVMValueRef(*_jug_llvm_fun_p_one_arg_t)(LLVMBuilderRef builder, LLVMValueRef arg, const char *name);
typedef LLVMValueRef(*_jug_llvm_fun_p_two_arg_t)(LLVMBuilderRef builder, LLVMValueRef lhs, LLVMValueRef rhs, const char *name);

typedef struct _jug_fun_type_s {
    char name[32];
    int require_decl;
    int nyi; /* not yet implemented */
    int arity;
    void* no_decl_ref_f32;
    void* no_decl_ref_f64;
    char decl_name_f32[32];
    char decl_name_f64[32];
    void *uint_ref;
    void *sint_ref;
    char decl_name_sint_pre[32];
    char decl_name_uint_pre[32];
} _jug_fun_type_t;

static const _jug_fun_type_t _jug_function_map[] = {
    {"EXPR_TYPE_ADD", 0, 0, 2, (void *) LLVMBuildFAdd, (void *) LLVMBuildFAdd, {0}, {0}, (void *) LLVMBuildAdd, (void *) LLVMBuildAdd, {0}, {0}},
    {"EXPR_TYPE_SUB", 0, 0, 2, (void *) LLVMBuildFSub, (void *) LLVMBuildFSub, {0}, {0}, (void *) LLVMBuildSub, (void *) LLVMBuildSub, {0}, {0}},
    {"EXPR_TYPE_MUL", 0, 0, 2, (void *) LLVMBuildFMul, (void *) LLVMBuildFMul, {0}, {0}, (void *) LLVMBuildMul, (void *) LLVMBuildMul, {0}, {0}},
    {"EXPR_TYPE_DIVIDE", 0, 0, 2, (void *) LLVMBuildFDiv, (void *) LLVMBuildFDiv, {0}, {0}, (void *) LLVMBuildUDiv, (void *) LLVMBuildSDiv, {0}, {0}},
    {"EXPR_TYPE_NEGATE", 0, 0, 1, (void *) LLVMBuildFNeg, (void *) LLVMBuildFNeg, {0}, {0}, (void *) LLVMBuildNeg, (void *) LLVMBuildNeg, {0}, {0}},
    {"EXPR_TYPE_COMMA", 1, 1, 1, NULL, NULL, {0}, {0}, 0, 0, {0}, {0}},
    {"EXPR_TYPE_ABS", 1, 0, 1, NULL, NULL, "llvm.fabs.f32", "llvm.fabs.f64", 0, 0, "llvm.abs.", "llvm.abs."},
    {"EXPR_TYPE_ACOS", 1, 0, 1, NULL, NULL, "acosf", "acos", 0, 0, {0}, {0}},
    {"EXPR_TYPE_ASIN", 1, 0, 1, NULL, NULL, "asinf", "asin", 0, 0, {0}, {0}},
    {"EXPR_TYPE_ATAN", 1, 0, 1, NULL, NULL, "atanf", "atan", 0, 0, {0}, {0}},
    {"EXPR_TYPE_ATAN2", 1, 0, 2, NULL, NULL, "atan2f", "atan2", 0, 0, {0}, {0}},
    {"EXPR_TYPE_CEIL", 1, 0, 1, NULL, NULL, "llvm.ceil.f32", "llvm.ceil.f64", 0, 0, {0}, {0}},
    {"EXPR_TYPE_COS", 1, 0, 1, NULL, NULL, "llvm.cos.f32", "llvm.cos.f64", 0, 0, {0}, {0}},
    {"EXPR_TYPE_COSH", 1, 0, 1, NULL, NULL, "coshf", "cosh", 0, 0, {0}, {0}},
    {"EXPR_TYPE_E", 1, 1, 1, NULL, NULL, {0}, {0}, 0, 0, {0}, {0}},
    {"EXPR_TYPE_EXP", 1, 0, 1, NULL, NULL, "llvm.exp.f32", "llvm.exp.f64", 0, 0, {0}, {0}},
    {"EXPR_TYPE_FAC", 1, 1, 1, NULL, NULL, {0}, {0}, 0, 0, {0}, {0}},
    {"EXPR_TYPE_FLOOR", 1, 0, 1, NULL, NULL, "llvm.floor.f32", "llvm.floor.f64", 0, 0, {0}, {0}},
    {"EXPR_TYPE_LOG", 1, 0, 1, NULL, NULL, "llvm.log.f32", "llvm.log.f64", 0, 0, {0}, {0}},
    {"EXPR_TYPE_LOG10", 1, 0, 1, NULL, NULL, "llvm.log10.f32", "llvm.log10.f64", 0, 0, {0}, {0}},
    {"EXPR_TYPE_NCR", 1, 1, 1, NULL, NULL, {0}, {0}, 0, 0, {0}, {0}},
    {"EXPR_TYPE_NPR", 1, 1, 1, NULL, NULL, {0}, {0}, 0, 0, {0}, {0}},
    {"EXPR_TYPE_PI", 1, 1, 1, NULL, NULL, {0}, {0}, 0, 0, {0}, {0}},
    {"EXPR_TYPE_POW", 1, 0, 2, NULL, NULL, "llvm.pow.f32", "llvm.pow.f64", 0, 0, {0}, {0}},
    {"EXPR_TYPE_SIN", 1, 0, 1, NULL, NULL, "llvm.sin.f32", "llvm.sin.f64", 0, 0, {0}, {0}},
    {"EXPR_TYPE_SINH", 1, 0, 1, NULL, NULL, "sinhf", "sinh", 0, 0, {0}, {0}},
    {"EXPR_TYPE_SQRT", 1, 0, 1, NULL, NULL, "llvm.sqrt.f32", "llvm.sqrt.f64", 0, 0, {0}, {0}},
    {"EXPR_TYPE_TAN", 1, 0, 1, NULL, NULL, "tanf", "tan", 0, 0, {0}, {0}},
    {"EXPR_TYPE_TANH", 1, 0, 1, NULL, NULL, "tanhf", "tanh", 0, 0, {0}, {0}},
    {"EXPR_TYPE_FMOD", 1, 0, 2, NULL, NULL, "fmodf", "fmod", 0, 0, {0}, {0}},
    {"EXPR_TYPE_MIN", 1, 0, 2, NULL, NULL, "fminf", "fmin", 0, 0, "llvm.smin.", "llvm.umin."},
    {"EXPR_TYPE_MAX", 1, 0, 2, NULL, NULL, "fmaxf", "fmax", 0, 0, "llvm.smax.", "llvm.umax."},
    NULL, // MSVC does not allow the {} form, so express the sentinel as NULL
};

static LLVMValueRef _jug_build_fun_call(jug_expression_t *e, const char *name, int num_args, LLVMValueRef *args)
{
    ina_str_t fname = NULL;

    /* lookup function */
    const _jug_fun_type_t *f = NULL;
    ina_hashtable_get_str(e->fun_map, name, (void**)&f);
    INA_ASSERT_NOT_NULL(f);

    if (f->nyi) {
        INA_ASSERT_TRUE(0);
    }

    INA_ASSERT_EQUAL(num_args, f->arity);

    /* declare function - if required */
    LLVMTypeRef *param_types = NULL;
    LLVMValueRef fun_decl = NULL;
    if (f->require_decl) {
        const char *fun_name;
        if (e->dtype == JUG_EXPRESSION_DTYPE_FLOAT) {
            fun_name = f->decl_name_f32;
        }
        else if (e->dtype == JUG_EXPRESSION_DTYPE_DOUBLE) {
            fun_name = f->decl_name_f64;
        }
        else {
            switch (e->dtype) {
                case JUG_EXPRESSION_DTYPE_SINT8:
                    fname = ina_str_new_fromcstr(f->decl_name_sint_pre);
                    fname = ina_str_catcstr(fname, "i8");
                    break;
                case JUG_EXPRESSION_DTYPE_SINT16:
                    fname = ina_str_new_fromcstr(f->decl_name_sint_pre);
                    fname = ina_str_catcstr(fname, "i16");
                    break;
                case JUG_EXPRESSION_DTYPE_SINT32:
                    fname = ina_str_new_fromcstr(f->decl_name_sint_pre);
                    fname = ina_str_catcstr(fname, "i32");
                    break;
                case JUG_EXPRESSION_DTYPE_SINT64:
                    fname = ina_str_new_fromcstr(f->decl_name_sint_pre);
                    fname = ina_str_catcstr(fname, "i64");
                    break;
                case JUG_EXPRESSION_DTYPE_UINT8:
                    fname = ina_str_new_fromcstr(f->decl_name_uint_pre);
                    fname = ina_str_catcstr(fname, "u8");
                    break;
                case JUG_EXPRESSION_DTYPE_UINT16:
                    fname = ina_str_new_fromcstr(f->decl_name_uint_pre);
                    fname = ina_str_catcstr(fname, "u16");
                    break;
                case JUG_EXPRESSION_DTYPE_UINT32:
                    fname = ina_str_new_fromcstr(f->decl_name_uint_pre);
                    fname = ina_str_catcstr(fname, "u32");
                    break;
                case JUG_EXPRESSION_DTYPE_UINT64:
                    fname = ina_str_new_fromcstr(f->decl_name_uint_pre);
                    fname = ina_str_catcstr(fname, "u64");
                    break;
                default:
                    IARRAY_TRACE1(iarray.error, "Invalid data type");
                    INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
            }
            fun_name = ina_str_cstr(fname);
        }
        ina_hashtable_get_str(e->decl_cache, fun_name, (void**)&fun_decl);
        if (fun_decl == NULL) {
            // We need special handling for ABS(INTEGER) because it requires a second argument
            LLVMTypeRef fn_type;
            if (strcmp("EXPR_TYPE_ABS", name) == 0 && !(e->dtype == JUG_EXPRESSION_DTYPE_DOUBLE || e->dtype == JUG_EXPRESSION_DTYPE_FLOAT)) {
                param_types = (LLVMTypeRef*)ina_mem_alloc(sizeof(LLVMTypeRef)*2);
                param_types[0] = e->expr_type;
                param_types[1] = LLVMInt1Type();
                fn_type = LLVMFunctionType(e->expr_type, param_types, 2, 0);
            }
            else {
                param_types = (LLVMTypeRef*)ina_mem_alloc(sizeof(LLVMTypeRef)*num_args);
                for (int i = 0; i < num_args; ++i) {
                    param_types[i] = e->expr_type;
                }
                fn_type = LLVMFunctionType(e->expr_type, param_types, num_args, 0);
            }
            fun_decl = LLVMAddFunction(e->mod, fun_name, fn_type);
            ina_hashtable_set_str(e->decl_cache, fun_name, fun_decl);
        }
    }
    else {
        /* if not required, build IR instruction and return (ADD, SUB, MUL, DIV etc.) */
        void *fun_ref = NULL;
        switch (e->dtype) {
            case JUG_EXPRESSION_DTYPE_FLOAT:
                fun_ref = f->no_decl_ref_f32;
                break;
            case JUG_EXPRESSION_DTYPE_DOUBLE:
                fun_ref = f->no_decl_ref_f64;
                break;
            case JUG_EXPRESSION_DTYPE_SINT8:
            case JUG_EXPRESSION_DTYPE_SINT16:
            case JUG_EXPRESSION_DTYPE_SINT32:
            case JUG_EXPRESSION_DTYPE_SINT64:
                fun_ref = f->sint_ref;
                break;
            case JUG_EXPRESSION_DTYPE_UINT8:
            case JUG_EXPRESSION_DTYPE_UINT16:
            case JUG_EXPRESSION_DTYPE_UINT32:
            case JUG_EXPRESSION_DTYPE_UINT64:
                fun_ref = f->uint_ref;
                break;
        }
        _jug_llvm_fun_p_one_arg_t oa;
        _jug_llvm_fun_p_two_arg_t ta;
        switch (f->arity) {
            case 1:
                oa = (_jug_llvm_fun_p_one_arg_t)fun_ref;
                return oa(e->builder, args[0], f->name);
            case 2:
                ta = (_jug_llvm_fun_p_two_arg_t)fun_ref;
                return ta(e->builder, args[0], args[1], f->name);
            default:
                INA_ASSERT_TRUE(0);
        }
    }

    /* build call */
    LLVMValueRef ret;
    INA_ASSERT_NOT_NULL(fun_decl);
    // We need special handling for ABS(INTEGER) because it requires a second argument
    if (strcmp("EXPR_TYPE_ABS", name) == 0 && !(e->dtype == JUG_EXPRESSION_DTYPE_DOUBLE || e->dtype == JUG_EXPRESSION_DTYPE_FLOAT)) {
        LLVMValueRef c_zero = LLVMConstInt(LLVMInt1Type(), 0, 1);
        LLVMValueRef largs[] = { args[0],  c_zero};
        ret = LLVMBuildCall(e->builder, fun_decl, largs, 2, name);
    }
    else {
        ret = LLVMBuildCall(e->builder, fun_decl, args, num_args, name);
    }

    /* cleanup - if required */
    if (param_types != NULL) {
        ina_mem_free(param_types);
    }

    if (fname != NULL) {
        ina_str_free(fname);
    }

    return ret;
}

static LLVMValueRef _jug_expr_build_proxy_one_args(jug_expression_t *e, LLVMValueRef arg, const char *name)
{
    LLVMValueRef args[] = { arg };
    return _jug_build_fun_call(e, name, 1, args);
}

static LLVMValueRef _jug_expr_build_proxy_two_args(jug_expression_t *e, LLVMValueRef lhs, LLVMValueRef rhs, const char *name)
{
    LLVMValueRef args[] = { lhs, rhs };
    return _jug_build_fun_call(e, name, 2, args);
}

static ina_rc_t _jug_register_functions(jug_expression_t *e)
{
    ina_hashtable_new(INA_HASHTABLE_STR_KEY,
        INA_HASH32_LOOKUP3,
        INA_HASHTABLE_TYPE_DEFAULT,
        INA_HASHTABLE_GROW_DEFAULT,
        INA_HASHTABLE_SHRINK_DEFAULT,
        INA_HASHTABLE_DEFAULT_CAPACITY,
        INA_HASHTABLE_CF_DEFAULT, &e->fun_map);

    ina_hashtable_new(INA_HASHTABLE_STR_KEY,
        INA_HASH32_LOOKUP3,
        INA_HASHTABLE_TYPE_DEFAULT,
        INA_HASHTABLE_GROW_DEFAULT,
        INA_HASHTABLE_SHRINK_DEFAULT,
        INA_HASHTABLE_DEFAULT_CAPACITY,
        INA_HASHTABLE_CF_DEFAULT, &e->decl_cache);

    int size = (sizeof(_jug_function_map) / sizeof(_jug_fun_type_t)) - 1; /* do not count the sentinel */

    e->fun_map_te = ina_mem_alloc(sizeof(void*) * size);

    for (int i = 0; i < size; ++i) {
        const _jug_fun_type_t *f = &(_jug_function_map[i]);
        switch (f->arity) {
            case 1:
                e->fun_map_te[i] = (void**)_jug_expr_build_proxy_one_args;
                break;
            case 2:
                e->fun_map_te[i] = (void**)_jug_expr_build_proxy_two_args;
                break;
            default:
                INA_ASSERT_TRUE(0);
        }
        ina_hashtable_set_str(e->fun_map, f->name, f);
    }

    return INA_SUCCESS;
}

typedef jug_expression_t* jug_expression_ptr_t;
#define TE_FUN(...) ((LLVMValueRef(*)(__VA_ARGS__))e->fun_map_te[n->function])
#define M(p) _jug_expr_compile_expression(e, n->parameters[p], params)
#define TYPE_MASK(TYPE) ((TYPE)&0x0000001F)
#define ARITY(TYPE) ( ((TYPE) & (TE_FUNCTION0 | TE_CLOSURE0)) ? ((TYPE) & 0x00000007) : 0 )
static LLVMValueRef _jug_expr_compile_expression(jug_expression_t *e, jug_te_expr *n, ina_hashtable_t *params)
{
    switch (TYPE_MASK(n->type)) {
        case TE_CONSTANT: {
            LLVMValueRef constant;
            switch (e->dtype) {
                case JUG_EXPRESSION_DTYPE_DOUBLE:
                case JUG_EXPRESSION_DTYPE_FLOAT:
                    constant = LLVMConstReal(e->expr_type, n->value);
                    break;
                case JUG_EXPRESSION_DTYPE_UINT8:
                    constant = LLVMConstInt(e->expr_type, (uint8_t)n->value, 0);
                    break;
                case JUG_EXPRESSION_DTYPE_UINT16:
                    constant = LLVMConstInt(e->expr_type, (uint16_t)n->value, 0);
                    break;
                case JUG_EXPRESSION_DTYPE_UINT32:
                    constant = LLVMConstInt(e->expr_type, (uint32_t)n->value, 0);
                    break;
                case JUG_EXPRESSION_DTYPE_UINT64:
                    constant = LLVMConstInt(e->expr_type, (uint64_t)n->value, 0);
                    break;
                case JUG_EXPRESSION_DTYPE_SINT8:
                    constant = LLVMConstInt(e->expr_type, (int8_t)n->value, 1);
                    break;
                case JUG_EXPRESSION_DTYPE_SINT16:
                    constant = LLVMConstInt(e->expr_type, (int16_t)n->value, 1);
                    break;
                case JUG_EXPRESSION_DTYPE_SINT32:
                    constant = LLVMConstInt(e->expr_type, (int32_t)n->value, 1);
                    break;
                case JUG_EXPRESSION_DTYPE_SINT64:
                    constant = LLVMConstInt(e->expr_type, (int64_t)n->value, 1);
                    break;
            }
            return constant;
        }
        case TE_VARIABLE: {
            LLVMValueRef param;
            ina_hashtable_get_str(params, n->bound, (void**)&param);
            return param;
        }
        case TE_FUNCTION0: case TE_FUNCTION1: case TE_FUNCTION2: case TE_FUNCTION3:
        case TE_FUNCTION4: case TE_FUNCTION5: case TE_FUNCTION6: case TE_FUNCTION7:
            switch (ARITY(n->type)) {
            case 0: return TE_FUN(jug_expression_ptr_t, const char*)(e, te_function_map_str[n->function]);
            case 1: return TE_FUN(jug_expression_ptr_t, LLVMValueRef, const char*)(e, M(0), te_function_map_str[n->function]);
            case 2: return TE_FUN(jug_expression_ptr_t, LLVMValueRef, LLVMValueRef, const char*)(e, M(0), M(1), te_function_map_str[n->function]);
            case 3: return TE_FUN(jug_expression_ptr_t, LLVMValueRef, LLVMValueRef, LLVMValueRef, const char*)(e, M(0), M(1), M(2), te_function_map_str[n->function]);
            case 4: return TE_FUN(jug_expression_ptr_t, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, const char*)(e, M(0), M(1), M(2), M(3), te_function_map_str[n->function]);
            case 5: return TE_FUN(jug_expression_ptr_t, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, const char*)(e, M(0), M(1), M(2), M(3), M(4), te_function_map_str[n->function]);
            case 6: return TE_FUN(jug_expression_ptr_t, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, const char*)(e, M(0), M(1), M(2), M(3), M(4), M(5), te_function_map_str[n->function]);
            case 7: return TE_FUN(jug_expression_ptr_t, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, const char*)(e, M(0), M(1), M(2), M(3), M(4), M(5), M(6), te_function_map_str[n->function]);
            default: return NULL;
            }

        case TE_CLOSURE0: case TE_CLOSURE1: case TE_CLOSURE2: case TE_CLOSURE3:
        case TE_CLOSURE4: case TE_CLOSURE5: case TE_CLOSURE6: case TE_CLOSURE7:
            switch (ARITY(n->type)) {
            case 0: return TE_FUN(void*, jug_expression_ptr_t, const char*)(n->parameters[0], e, te_function_map_str[n->function]);
            case 1: return TE_FUN(void*, jug_expression_ptr_t, LLVMValueRef, const char*)(n->parameters[1], e, M(0), te_function_map_str[n->function]);
            case 2: return TE_FUN(void*, jug_expression_ptr_t, LLVMValueRef, LLVMValueRef, const char*)(n->parameters[2], e, M(0), M(1), te_function_map_str[n->function]);
            case 3: return TE_FUN(void*, jug_expression_ptr_t, LLVMValueRef, LLVMValueRef, LLVMValueRef, const char*)(n->parameters[3], e, M(0), M(1), M(2), te_function_map_str[n->function]);
            case 4: return TE_FUN(void*, jug_expression_ptr_t, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, const char*)(n->parameters[4], e, M(0), M(1), M(2), M(3), te_function_map_str[n->function]);
            case 5: return TE_FUN(void*, jug_expression_ptr_t, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, const char*)(n->parameters[5], e, M(0), M(1), M(2), M(3), M(4), te_function_map_str[n->function]);
            case 6: return TE_FUN(void*, jug_expression_ptr_t, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, const char*)(n->parameters[6], e, M(0), M(1), M(2), M(3), M(4), M(5), te_function_map_str[n->function]);
            case 7: return TE_FUN(void*, jug_expression_ptr_t, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, const char*)(n->parameters[7], e, M(0), M(1), M(2), M(3), M(4), M(5), M(6), te_function_map_str[n->function]);
            default: return NULL;
            }

        default: return NULL;
    }
}
#undef TE_FUN
#undef M
#undef TYPE_MASK
#undef ARITY

#ifdef _JUG_DEBUG_DECLARE_PRINT_IN_IR
static void debug_print(LLVMBuilderRef builder, LLVMModuleRef module, const char *fmt, LLVMValueRef value)
{
    LLVMValueRef format = LLVMBuildGlobalStringPtr(builder, fmt, "format");
    LLVMValueRef printf_function = LLVMGetNamedFunction(module, "printf");
    LLVMValueRef printf_args[] = { format, value };
    LLVMBuildCall(builder, printf_function, printf_args, 2, "printf");
}
#endif

static LLVMValueRef _jug_expr_compile_function(
    jug_expression_t *e,
    const char *name,
    jug_te_expr *expression,
    int var_len,
    jug_te_variable *vars)
{
    ina_hashtable_t *param_values = NULL;

    ina_hashtable_new(INA_HASHTABLE_STR_KEY,
        INA_HASH32_LOOKUP3,
        INA_HASHTABLE_TYPE_DEFAULT,
        INA_HASHTABLE_GROW_DEFAULT,
        INA_HASHTABLE_SHRINK_DEFAULT,
        INA_HASHTABLE_DEFAULT_CAPACITY,
        INA_HASHTABLE_CF_DEFAULT, &param_values);

    LLVMTypeRef int32Type = LLVMInt32Type();

    e->context = LLVMContextCreate();

    LLVMValueRef constant_zero = LLVMConstInt(int32Type, 0, 1);
    LLVMValueRef constant_one = LLVMConstInt(int32Type, 1, 1);
    switch (e->dtype) {
        case JUG_EXPRESSION_DTYPE_FLOAT:
            e->expr_type = LLVMFloatType();
            break;
        case JUG_EXPRESSION_DTYPE_DOUBLE:
            e->expr_type = LLVMDoubleType();
            break;
        case JUG_EXPRESSION_DTYPE_SINT8:
        case JUG_EXPRESSION_DTYPE_UINT8:
            e->expr_type = LLVMInt8Type();
            break;
        case JUG_EXPRESSION_DTYPE_SINT16:
        case JUG_EXPRESSION_DTYPE_UINT16:
            e->expr_type = LLVMInt16Type();
            break;
        case JUG_EXPRESSION_DTYPE_SINT32:
        case JUG_EXPRESSION_DTYPE_UINT32:
            e->expr_type = LLVMInt32Type();
            break;
        case JUG_EXPRESSION_DTYPE_SINT64:
        case JUG_EXPRESSION_DTYPE_UINT64:
            e->expr_type = LLVMInt64Type();
            break;
        default:
            return NULL;
    }

    /* define the parameter structure for prefilter */
#define JUG_EVAL_PPARAMS_STRUCT_NUM_FIELDS 7
    LLVMTypeRef params_struct = LLVMStructCreateNamed(e->context, "struct.iarray_eval_pparams_t");
    LLVMTypeRef *params_struct_types = ina_mem_alloc(sizeof(LLVMTypeRef) * JUG_EVAL_PPARAMS_STRUCT_NUM_FIELDS);
    params_struct_types[0] = LLVMInt32Type();  /* ninputs */
    params_struct_types[1] = LLVMArrayType(LLVMPointerType(LLVMInt8Type(), 0), IARRAY_EXPR_OPERANDS_MAX);  /* inputs */
    params_struct_types[2] = LLVMArrayType(LLVMInt32Type(), IARRAY_EXPR_OPERANDS_MAX);  /* inputs typesizes */
    params_struct_types[3] = LLVMPointerType(LLVMInt8Type(), 0);  /* userdata */
    params_struct_types[4] = LLVMPointerType(LLVMInt8Type(), 0);  /* out */
    params_struct_types[5] = LLVMInt32Type();  /* out_size */
    params_struct_types[6] = LLVMInt32Type();  /* out typesize */

    LLVMStructSetBody(params_struct, params_struct_types, JUG_EVAL_PPARAMS_STRUCT_NUM_FIELDS, 0);

    LLVMTypeRef param_types[1] = {
        LLVMPointerType(params_struct, 0)
    };
    LLVMTypeRef prototype = LLVMFunctionType(LLVMInt32Type(), param_types, 1, 0);
    LLVMValueRef f = LLVMAddFunction(e->mod, name, prototype);

    LLVMBasicBlockRef stackvar_sec = LLVMAppendBasicBlock(f, "stack_vars");
    LLVMBasicBlockRef loop_len = LLVMAppendBasicBlock(f, "loop_len");
    LLVMBasicBlockRef entry = LLVMAppendBasicBlock(f, "entry");
    LLVMBasicBlockRef condition = LLVMAppendBasicBlock(f, "condition");
    LLVMBasicBlockRef body = LLVMAppendBasicBlock(f, "body");
    LLVMBasicBlockRef increment = LLVMAppendBasicBlock(f, "increment");
    LLVMBasicBlockRef end = LLVMAppendBasicBlock(f, "end");

    e->builder = LLVMCreateBuilder(); // FIXME, probably better to build it from context, mem-leak?

    LLVMValueRef param_ptr = LLVMGetParam(f, 0);

    LLVMValueRef local_output;
    LLVMValueRef *local_inputs;
    ina_str_t *local_input_labels;
    LLVMPositionBuilderAtEnd(e->builder, stackvar_sec);
    {
        local_output = LLVMBuildAlloca(e->builder, LLVMPointerType(e->expr_type, 0), "local_output");
        local_inputs = ina_mem_alloc(sizeof(LLVMValueRef*)*var_len); // leaking memory for now
        local_input_labels = ina_mem_alloc(sizeof(ina_str_t)*var_len); // leaking memory for now

        LLVMValueRef ninputs = LLVMBuildStructGEP(e->builder, param_ptr, 0, "ninputs");
        INA_UNUSED(ninputs); // TODO: compare arg_count with ninputs, return error (constant_one) if different

        LLVMValueRef inputs_ptr = LLVMBuildStructGEP(e->builder, param_ptr, 1, "inputs_ptr");
        LLVMValueRef inputs = LLVMBuildLoad(e->builder, inputs_ptr, "inputs");
        for (int i = 0; i < var_len; ++i) {
            local_inputs[i] = ina_mem_alloc(sizeof(LLVMValueRef));

            local_input_labels[i] = ina_str_sprintf("input[%d]", i); // leaking memory for now
            local_inputs[i] = LLVMBuildAlloca(e->builder, LLVMPointerType(e->expr_type, 0), ina_str_cstr(local_input_labels[i]));

            /* Load array of inputs */
            LLVMValueRef in_addr = LLVMBuildExtractValue(e->builder, inputs, i, "inputs[index]");

            /* Cast to value type */
            LLVMTypeRef type_cast = LLVMPointerType(e->expr_type, 0);
            LLVMValueRef cast_in = LLVMBuildCast(e->builder, LLVMBitCast, in_addr, type_cast, "cast[double*]");

            /* Store pointer in stack var */
            LLVMBuildStore(e->builder, cast_in, local_inputs[i]);
        }

        LLVMValueRef out_ptr = LLVMBuildStructGEP(e->builder, param_ptr, 4, "out_ptr");
        LLVMValueRef out = LLVMBuildLoad(e->builder, out_ptr, "out");
        LLVMValueRef out_cast = LLVMBuildCast(e->builder, LLVMBitCast, out, LLVMPointerType(e->expr_type, 0), "out_cast");
        LLVMBuildStore(e->builder, out_cast, local_output);

        LLVMBuildBr(e->builder, loop_len);
    }


    LLVMValueRef len;
    LLVMPositionBuilderAtEnd(e->builder, loop_len);
    {
        LLVMValueRef out_size_ptr = LLVMBuildStructGEP(e->builder, param_ptr, 5, "out_size_ptr");
        LLVMValueRef out_size = LLVMBuildLoad(e->builder, out_size_ptr, "out_size");
        LLVMValueRef out_size_val = LLVMBuildPtrToInt(e->builder, out_size, LLVMInt32Type(), "out_size_val");

        LLVMValueRef out_typesize_ptr = LLVMBuildStructGEP(e->builder, param_ptr, 6, "out_typesize_ptr");
        LLVMValueRef out_typesize = LLVMBuildLoad(e->builder, out_typesize_ptr, "out_typesize");
        LLVMValueRef out_typesize_val = LLVMBuildPtrToInt(e->builder, out_typesize, LLVMInt32Type(), "out_typesize_val");

        len = LLVMBuildExactSDiv(e->builder, out_size_val, out_typesize_val, "calculate_len");
        LLVMBuildBr(e->builder, entry);
    }

    LLVMValueRef index_addr;
    LLVMPositionBuilderAtEnd(e->builder, entry);
    {
        index_addr = LLVMBuildAlloca(e->builder, int32Type, "index");
        LLVMBuildStore(e->builder, constant_zero, index_addr);
        LLVMBuildBr(e->builder, condition);
    }

    LLVMPositionBuilderAtEnd(e->builder, condition);
    {
        LLVMValueRef index = LLVMBuildLoad(e->builder, index_addr, "[index]");
        LLVMValueRef cond = LLVMBuildICmp(e->builder, LLVMIntSLT, index, len, "index < len");
        LLVMBuildCondBr(e->builder, cond, body, end);
    }
    LLVMPositionBuilderAtEnd(e->builder, body);
    {
        LLVMValueRef md_values_access[] = { LLVMMDString("llvm.access.group",
            (unsigned int)strlen("llvm.access.group")) };
        LLVMValueRef md_access = LLVMMDNode(md_values_access, 1);

        LLVMValueRef md_values[] = { LLVMMDString("llvm.loop.parallel_accesses",
            (unsigned int)strlen("llvm.loop.parallel_accesses")), md_access };
        LLVMValueRef md_node = LLVMMDNode(md_values, 2);

        LLVMValueRef index = LLVMBuildLoad(e->builder, index_addr, "[index]");

        /* Load the scalar values from the inputs */
        for (int i = 0; i < var_len; ++i) {
            LLVMValueRef stack_var = LLVMBuildLoad(e->builder, local_inputs[i], "load_stackvar");
            LLVMValueRef addr = LLVMBuildGEP(e->builder, stack_var, &index, 1, "buffer[index]");

            /* Load scalar value */
            LLVMValueRef val = LLVMBuildLoad(e->builder, addr, "value");
            LLVMSetMetadata(val, LLVMInstructionValueKind, md_access);
            const char *key = vars[i].name;
            ina_hashtable_set_str(param_values, key, val);
        }

        /* compute the expression */
        LLVMValueRef result = _jug_expr_compile_expression(e, expression, param_values);

        /* store the result */
        LLVMValueRef local_out_ref = LLVMBuildLoad(e->builder, local_output, "local_output");
        LLVMValueRef out_addr = LLVMBuildGEP(e->builder, local_out_ref, &index, 1, "out_addr");
        LLVMValueRef store = LLVMBuildStore(e->builder, result, out_addr);
        LLVMSetMetadata(store, LLVMInstructionValueKind, md_access);

        LLVMValueRef loop_latch = LLVMBuildBr(e->builder, increment);
        LLVMSetMetadata(loop_latch, LLVMInstructionValueKind, md_node);
    }
    LLVMPositionBuilderAtEnd(e->builder, increment);
    {
        LLVMValueRef index = LLVMBuildLoad(e->builder, index_addr, "[index]");
        LLVMValueRef indexpp = LLVMBuildAdd(e->builder, index, constant_one, "index++");
        LLVMBuildStore(e->builder, indexpp, index_addr);
        LLVMBuildBr(e->builder, condition);
    }
    LLVMPositionBuilderAtEnd(e->builder, end);

    LLVMBuildRet(e->builder, constant_zero);

    ina_hashtable_free(&param_values);

    return f;
}

static void _jug_apply_optimisation_passes(jug_expression_t *e)
{
    LLVMPassManagerBuilderRef pmb = LLVMPassManagerBuilderCreate();
    jug_utils_enable_loop_vectorize(pmb);
    LLVMPassManagerBuilderSetOptLevel(pmb, 2); // Opt level 0-3

    // Module pass manager
    LLVMPassManagerRef pm = LLVMCreatePassManager();
    LLVMAddAnalysisPasses(_jug_tm_ref, pm);
    LLVMPassManagerBuilderPopulateModulePassManager(pmb, pm);

    LLVMAddLoopVectorizePass(pm);
    LLVMAddSLPVectorizePass(pm);

    // Run
    LLVMRunPassManager(pm, e->mod);

    // Dispose
    LLVMDisposePassManager(pm);
    LLVMPassManagerBuilderDispose(pmb);
}

#ifdef _JUG_DEBUG_DECLARE_PRINT_IN_IR
static void _jug_declare_printf(LLVMModuleRef mod)
{
    LLVMTypeRef printf_args_ty_list[] = { LLVMPointerType(LLVMInt8Type(), 0) };
    LLVMTypeRef printf_ty =
        LLVMFunctionType(LLVMInt64Type(), printf_args_ty_list, 0, 1);
    LLVMAddFunction(mod, "printf", printf_ty);
}
#endif

/*
 * Code common to jug_expression_compile and jug_udf_compile functions:
 * verifies module, optimizes, creates execution engine
 */
static LLVMBool _jug_prepare_module(jug_expression_t *e, bool reload)
{
    LLVMBool error;
    char *message = NULL;

    // Verify the module
    error = LLVMVerifyModule(e->mod, LLVMAbortProcessAction, &message);
    if (error)
    {
        fprintf(stderr, "LLVM module verification error: '%s'\n", message);
        goto exit;
    }

    LLVMSetModuleDataLayout(e->mod, _jug_data_ref);
    LLVMSetTarget(e->mod, _jug_def_triple);

    // Debug: write bitcode before otimization
#ifdef _JUG_DEBUG_WRITE_BC_TO_FILE
    if (LLVMWriteBitcodeToFile(e->mod, "expression.bc") != 0) {
        fprintf(stderr, "error writing bitcode to file, skipping\n");
    }
#endif

    // Workaround
    if (reload) {
        LLVMMemoryBufferRef buffer = LLVMWriteBitcodeToMemoryBuffer(e->mod);
        error = LLVMParseIRInContext(e->context, buffer, &e->mod, &message);
        if (error) {
            fprintf(stderr, "LLVM module parse error: '%s'\n", message);
            goto exit;
        }
    }

    // Optimze
    _jug_apply_optimisation_passes(e);
#ifdef _JUG_DEBUG_WRITE_BC_TO_FILE
    if (LLVMWriteBitcodeToFile(e->mod, "expression_opt.bc") != 0) {
        fprintf(stderr, "error writing bitcode to file, skipping\n");
    }
#endif

    // Create execution engine
    error = jug_utils_create_execution_engine(e->mod, &e->engine);
    //error = LLVMCreateExecutionEngineForModule(&e->engine, e->mod, &message);
    if (error) {
        fprintf(stderr, "LLVM execution engine creation error: '%s'\n", message);
        goto exit;
    }

exit:
    LLVMDisposeMessage(message);
    return error;
}


INA_API(ina_rc_t) jug_init()
{
    char *error = NULL;
    jug_util_set_svml_vector_library();

    LLVMBool llvm_error;
    llvm_error = LLVMInitializeNativeTarget();
    llvm_error = LLVMInitializeNativeAsmPrinter();
    INA_UNUSED(llvm_error);
    LLVMLinkInMCJIT();

    _jug_def_triple = LLVMGetDefaultTargetTriple();
    LLVMTargetRef target_ref;
    if (LLVMGetTargetFromTriple(_jug_def_triple, &target_ref, &error)) {
#ifdef _JUG_DEBUG_WRITE_ERROR_TO_STDERR
        fprintf(stderr, "%s", error);
#endif
        LLVMDisposeMessage(error);
        return INA_ERR_FATAL;
    }

    if (!LLVMTargetHasJIT(target_ref)) {
#ifdef _JUG_DEBUG_WRITE_ERROR_TO_STDERR
        fprintf(stderr, "Fatal error: Cannot do JIT on this platform");
#endif
        LLVMDisposeMessage(error);
        return INA_ERR_FATAL;
    }

    const char *cpu = jug_utils_get_cpu_string();
    _jug_tm_ref =
        LLVMCreateTargetMachine(target_ref, _jug_def_triple, cpu, "",
            LLVMCodeGenLevelDefault,
            LLVMRelocDefault,
            LLVMCodeModelJITDefault);
    _jug_data_ref = LLVMCreateTargetDataLayout(_jug_tm_ref);

    /* This is required to make sure Intel SVML is being linked and loaded properly */
    double wrkarnd[2] = { 0.1, 0.2 };
    __svml_sin2(wrkarnd);

    return INA_SUCCESS;
}

INA_API(void) jug_destroy()
{
// FIX: the code below makes some tests to fail.  Commenting this out for the time being.
//    if (_jug_tm_ref != NULL) {
//        LLVMDisposeTargetMachine(_jug_tm_ref);
//        _jug_tm_ref = NULL;
//    }
}

INA_API(ina_rc_t) jug_expression_new(jug_expression_t **expr, jug_expression_dtype_t dtype) {
    LLVMModuleRef m;
    *expr = (jug_expression_t*)ina_mem_alloc(sizeof(jug_expression_t));
    memset(*expr, 0, sizeof(jug_expression_t));
    (*expr)->dtype = dtype;
    (*expr)->mod = LLVMModuleCreateWithName("expr_engine");
    m = (*expr)->mod;
    INA_UNUSED(m);
    _jug_register_functions(*expr);

#ifdef _JUG_DEBUG_DECLARE_PRINT_IN_IR
    _jug_declare_printf(m);
#endif

    return INA_SUCCESS;
}

INA_API(void) jug_expression_free(jug_expression_t **expr)
{
    INA_VERIFY_FREE(expr);
    if ((*expr)->fun_map != NULL) {
        ina_hashtable_free(&(*expr)->fun_map);
    }
    if ((*expr)->decl_cache != NULL) {
        ina_hashtable_free(&(*expr)->decl_cache);
    }
    if ((*expr)->fun_map_te != NULL) {
        ina_mem_free((*expr)->fun_map_te);
    }
    if ((*expr)->engine != NULL) {
        LLVMDisposeExecutionEngine((*expr)->engine);
    }
    /*if ((*expr)->mod != NULL) {
        LLVMDisposeModule((*expr)->mod);
    }*/
    INA_MEM_FREE_SAFE(*expr);
}

INA_API(ina_rc_t) jug_udf_compile(
    jug_expression_t *e,
    int llvm_bc_len,
    const char *llvm_bc,
    const char *name,
    uint64_t *function_addr)
{
    char *message = NULL;
    LLVMMemoryBufferRef buffer;
    LLVMBool error;
    ina_rc_t rc = INA_SUCCESS;

    // Read the IR file into a buffer
    buffer = LLVMCreateMemoryBufferWithMemoryRange(llvm_bc, llvm_bc_len, "udf", 0);

    // now create our module
    e->context = LLVMContextCreate();
    error = LLVMParseIRInContext(e->context, buffer, &e->mod, &message);
    if (error) {
#ifdef _JUG_DEBUG_WRITE_ERROR_TO_STDERR
        fprintf(stderr, "Invalid IR detected! message: '%s'\n", message);
#endif
        rc = INA_ERR_FAILED;
        goto exit;
    }

    if (_jug_prepare_module(e, false)) {
        rc = INA_ERR_FAILED;
        goto exit;
    }

    *function_addr = LLVMGetFunctionAddress(e->engine, name);

exit:
    LLVMDisposeMessage(message);
    // for some strange reason, this does a "pointer being freed was not allocated"
    //LLVMDisposeMemoryBuffer(memoryBuffer);
    return rc;
}

INA_API(ina_rc_t) jug_expression_compile(
    jug_expression_t *e,
    const char *expr_str,
    int num_vars,
    void *vars,
    uint64_t *function_addr)
{
    int parse_error = 0;

    jug_te_variable *te_vars = (jug_te_variable*)vars;
    jug_te_expr *expression = jug_te_compile(expr_str, te_vars, num_vars, &parse_error);
    if (parse_error) {
        IARRAY_TRACE1(iarray.error, "Error parsing the expression with juggernaut");
        IARRAY_RETURN_IF_FAILED(INA_ERR_INVALID_ARGUMENT);
    }
    _jug_expr_compile_function(e, "expr_func", expression, num_vars, te_vars);
    jug_te_free(expression);

    if (_jug_prepare_module(e, true)) {
        IARRAY_TRACE1(iarray.error, "Error preparing LLVM module");
        return INA_ERROR(INA_ERR_FAILED);
    }

    *function_addr = LLVMGetFunctionAddress(e->engine, "expr_func");

    return INA_SUCCESS;
}

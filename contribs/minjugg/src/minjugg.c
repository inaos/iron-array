#include <minjugg.h>

#include <llvm-c/Core.h>
#include <llvm-c/ExecutionEngine.h>
#include <llvm-c/Target.h>
#include <llvm-c/Analysis.h>
#include <llvm-c/BitWriter.h>
#include <llvm-c/IRReader.h>

#include <llvm-c/Transforms/PassManagerBuilder.h>
#include <llvm-c/Transforms/Vectorize.h> // LLVMAddLoopVectorizePass

#include <blosc2.h>

#include "minjuggutil.h"
#include "tinyexpr.h"

#define _JUG_DEBUG_WRITE_BC_TO_FILE
#define _JUG_DEBUG_WRITE_ERROR_TO_STDERR

struct jug_expression_s {
    LLVMContextRef context;
    LLVMModuleRef mod;
    LLVMExecutionEngineRef engine;
};

static LLVMValueRef _jug_builtin_cos_f64;
static LLVMValueRef _jug_builtin_abs_f64;
static LLVMValueRef _jug_builtin_acos_f64;
static LLVMValueRef _jug_builtin_asin_f64;
static LLVMValueRef _jug_builtin_atan_f64;
static LLVMValueRef _jug_builtin_atan2_f64;
static LLVMValueRef _jug_builtin_ceil_f64;
static LLVMValueRef _jug_builtin_cosh_f64;
static LLVMValueRef _jug_builtin_exp_f64;
static LLVMValueRef _jug_builtin_floor_f64;
static LLVMValueRef _jug_builtin_ln_f64;
static LLVMValueRef _jug_builtin_log_f64;
static LLVMValueRef _jug_builtin_pow_f64;
static LLVMValueRef _jug_builtin_sin_f64;
static LLVMValueRef _jug_builtin_sinh_f64;
static LLVMValueRef _jug_builtin_sqrt_f64;
static LLVMValueRef _jug_builtin_tan_f64;
static LLVMValueRef _jug_builtin_tanh_f64;
static LLVMValueRef _jug_builtin_fmod_f64;

static char *_jug_def_triple = NULL;
static LLVMTargetDataRef _jug_data_ref = NULL;
static LLVMTargetMachineRef tm_ref = NULL;

static void _jug_declare_cos_f64(LLVMModuleRef mod)
{
    LLVMTypeRef param_types[] = { LLVMDoubleType() };
    LLVMTypeRef fn_type = LLVMFunctionType(LLVMDoubleType(), param_types, 1, 0);
    _jug_builtin_cos_f64 = LLVMAddFunction(mod, "llvm.cos.f64", fn_type);
}

static void _jug_declare_abs_f64(LLVMModuleRef mod)
{
    LLVMTypeRef param_types[] = { LLVMDoubleType() };
    LLVMTypeRef fn_type = LLVMFunctionType(LLVMDoubleType(), param_types, 1, 0);
    _jug_builtin_abs_f64 = LLVMAddFunction(mod, "llvm.fabs.f64", fn_type);
}

static void _jug_declare_acos_f64(LLVMModuleRef mod)
{
    LLVMTypeRef param_types[] = { LLVMDoubleType() };
    LLVMTypeRef fn_type = LLVMFunctionType(LLVMDoubleType(), param_types, 1, 0);
    _jug_builtin_acos_f64 = LLVMAddFunction(mod, "acos", fn_type);
}

static void _jug_declare_asin_f64(LLVMModuleRef mod)
{
    LLVMTypeRef param_types[] = { LLVMDoubleType() };
    LLVMTypeRef fn_type = LLVMFunctionType(LLVMDoubleType(), param_types, 1, 0);
    _jug_builtin_asin_f64 = LLVMAddFunction(mod, "asin", fn_type);
}

static void _jug_declare_atan_f64(LLVMModuleRef mod)
{
    LLVMTypeRef param_types[] = { LLVMDoubleType() };
    LLVMTypeRef fn_type = LLVMFunctionType(LLVMDoubleType(), param_types, 1, 0);
    _jug_builtin_atan_f64 = LLVMAddFunction(mod, "atan", fn_type);
}

static void _jug_declare_atan2_f64(LLVMModuleRef mod)
{
    LLVMTypeRef param_types[] = { LLVMDoubleType(), LLVMDoubleType() };
    LLVMTypeRef fn_type = LLVMFunctionType(LLVMDoubleType(), param_types, 2, 0);
    _jug_builtin_atan2_f64 = LLVMAddFunction(mod, "atan2", fn_type);
}

static void _jug_declare_ceil_f64(LLVMModuleRef mod)
{
    LLVMTypeRef param_types[] = { LLVMDoubleType() };
    LLVMTypeRef fn_type = LLVMFunctionType(LLVMDoubleType(), param_types, 1, 0);
    _jug_builtin_ceil_f64 = LLVMAddFunction(mod, "llvm.ceil.f64", fn_type);
}

static void _jug_declare_cosh_f64(LLVMModuleRef mod)
{
    LLVMTypeRef param_types[] = { LLVMDoubleType() };
    LLVMTypeRef fn_type = LLVMFunctionType(LLVMDoubleType(), param_types, 1, 0);
    _jug_builtin_cosh_f64 = LLVMAddFunction(mod, "cosh", fn_type);
}

static void _jug_declare_exp_f64(LLVMModuleRef mod)
{
    LLVMTypeRef param_types[] = { LLVMDoubleType() };
    LLVMTypeRef fn_type = LLVMFunctionType(LLVMDoubleType(), param_types, 1, 0);
    _jug_builtin_exp_f64 = LLVMAddFunction(mod, "llvm.exp.f64", fn_type);
}

static void _jug_declare_floor_f64(LLVMModuleRef mod)
{
    LLVMTypeRef param_types[] = { LLVMDoubleType() };
    LLVMTypeRef fn_type = LLVMFunctionType(LLVMDoubleType(), param_types, 1, 0);
    _jug_builtin_floor_f64 = LLVMAddFunction(mod, "llvm.floor.f64", fn_type);
}

static void _jug_declare_ln_f64(LLVMModuleRef mod)
{
    LLVMTypeRef param_types[] = { LLVMDoubleType() };
    LLVMTypeRef fn_type = LLVMFunctionType(LLVMDoubleType(), param_types, 1, 0);
    _jug_builtin_ln_f64 = LLVMAddFunction(mod, "llvm.log.f64", fn_type);
}

static void _jug_declare_log_f64(LLVMModuleRef mod)
{
    LLVMTypeRef param_types[] = { LLVMDoubleType() };
    LLVMTypeRef fn_type = LLVMFunctionType(LLVMDoubleType(), param_types, 1, 0);
    _jug_builtin_log_f64 = LLVMAddFunction(mod, "llvm.log10.f64", fn_type);
}

static void _jug_declare_pow_f64(LLVMModuleRef mod)
{
    LLVMTypeRef param_types[] = { LLVMDoubleType(), LLVMDoubleType() };
    LLVMTypeRef fn_type = LLVMFunctionType(LLVMDoubleType(), param_types, 2, 0);
    _jug_builtin_pow_f64 = LLVMAddFunction(mod, "llvm.pow.f64", fn_type);
}

static void _jug_declare_sin_f64(LLVMModuleRef mod)
{
    LLVMTypeRef param_types[] = { LLVMDoubleType() };
    LLVMTypeRef fn_type = LLVMFunctionType(LLVMDoubleType(), param_types, 1, 0);
    _jug_builtin_sin_f64 = LLVMAddFunction(mod, "sin", fn_type);
}

static void _jug_declare_sinh_f64(LLVMModuleRef mod)
{
    LLVMTypeRef param_types[] = { LLVMDoubleType() };
    LLVMTypeRef fn_type = LLVMFunctionType(LLVMDoubleType(), param_types, 1, 0);
    _jug_builtin_sinh_f64 = LLVMAddFunction(mod, "sinh", fn_type);
}

static void _jug_declare_sqrt_f64(LLVMModuleRef mod)
{
    LLVMTypeRef param_types[] = { LLVMDoubleType() };
    LLVMTypeRef fn_type = LLVMFunctionType(LLVMDoubleType(), param_types, 1, 0);
    _jug_builtin_sqrt_f64 = LLVMAddFunction(mod, "llvm.sqrt.f64", fn_type);
}

static void _jug_declare_tan_f64(LLVMModuleRef mod)
{
    LLVMTypeRef param_types[] = { LLVMDoubleType() };
    LLVMTypeRef fn_type = LLVMFunctionType(LLVMDoubleType(), param_types, 1, 0);
    _jug_builtin_tan_f64 = LLVMAddFunction(mod, "tan", fn_type);
}

static void _jug_declare_tanh_f64(LLVMModuleRef mod)
{
    LLVMTypeRef param_types[] = { LLVMDoubleType() };
    LLVMTypeRef fn_type = LLVMFunctionType(LLVMDoubleType(), param_types, 1, 0);
    _jug_builtin_tanh_f64 = LLVMAddFunction(mod, "tanh", fn_type);
}

static void _jug_declare_fmod_f64(LLVMModuleRef mod)
{
    LLVMTypeRef param_types[] = { LLVMDoubleType(), LLVMDoubleType() };
    LLVMTypeRef fn_type = LLVMFunctionType(LLVMDoubleType(), param_types, 2, 0);
    _jug_builtin_fmod_f64 = LLVMAddFunction(mod, "fmod", fn_type);
}

static LLVMValueRef _jug_build_comma(LLVMBuilderRef builder, LLVMValueRef lhs, LLVMValueRef rhs, const char *name)
{
    INA_UNUSED(builder);
    INA_UNUSED(lhs);
    INA_UNUSED(name);
    return rhs;
}

static LLVMValueRef _jug_build_cos_f64(LLVMBuilderRef builder, LLVMValueRef arg, const char *name)
{
    LLVMValueRef args[] = { arg };
    return LLVMBuildCall(builder, _jug_builtin_cos_f64, args, 1, name);
}

static LLVMValueRef _jug_build_abs_f64(LLVMBuilderRef builder, LLVMValueRef arg, const char *name)
{
    LLVMValueRef args[] = { arg };
    return LLVMBuildCall(builder, _jug_builtin_abs_f64, args, 1, name);
}

static LLVMValueRef _jug_build_acos_f64(LLVMBuilderRef builder, LLVMValueRef arg, const char *name)
{
    LLVMValueRef args[] = { arg };
    return LLVMBuildCall(builder, _jug_builtin_acos_f64, args, 1, name);
}

static LLVMValueRef _jug_build_asin_f64(LLVMBuilderRef builder, LLVMValueRef arg, const char *name)
{
    LLVMValueRef args[] = { arg };
    return LLVMBuildCall(builder, _jug_builtin_asin_f64, args, 1, name);
}

static LLVMValueRef _jug_build_atan_f64(LLVMBuilderRef builder, LLVMValueRef arg, const char *name)
{
    LLVMValueRef args[] = { arg };
    return LLVMBuildCall(builder, _jug_builtin_atan_f64, args, 1, name);
}

static LLVMValueRef _jug_build_atan2_f64(LLVMBuilderRef builder, LLVMValueRef lhs, LLVMValueRef rhs, const char *name)
{
    LLVMValueRef args[] = { lhs, rhs };
    return LLVMBuildCall(builder, _jug_builtin_atan2_f64, args, 2, name);
}

static LLVMValueRef _jug_build_ceil_f64(LLVMBuilderRef builder, LLVMValueRef arg, const char *name)
{
    LLVMValueRef args[] = { arg };
    return LLVMBuildCall(builder, _jug_builtin_ceil_f64, args, 1, name);
}

static LLVMValueRef _jug_build_cosh_f64(LLVMBuilderRef builder, LLVMValueRef arg, const char *name)
{
    LLVMValueRef args[] = { arg };
    return LLVMBuildCall(builder, _jug_builtin_cosh_f64, args, 1, name);
}

static LLVMValueRef _jug_build_exp_f64(LLVMBuilderRef builder, LLVMValueRef arg, const char *name)
{
    LLVMValueRef args[] = { arg };
    return LLVMBuildCall(builder, _jug_builtin_exp_f64, args, 1, name);
}

static LLVMValueRef _jug_build_ln_f64(LLVMBuilderRef builder, LLVMValueRef arg, const char *name)
{
    LLVMValueRef args[] = { arg };
    return LLVMBuildCall(builder, _jug_builtin_ln_f64, args, 1, name);
}

static LLVMValueRef _jug_build_log_f64(LLVMBuilderRef builder, LLVMValueRef arg, const char *name)
{
    LLVMValueRef args[] = { arg };
    return LLVMBuildCall(builder, _jug_builtin_log_f64, args, 1, name);
}

static LLVMValueRef _jug_build_floor_f64(LLVMBuilderRef builder, LLVMValueRef arg, const char *name)
{
    LLVMValueRef args[] = { arg };
    return LLVMBuildCall(builder, _jug_builtin_floor_f64, args, 1, name);
}

static LLVMValueRef _jug_build_pow_f64(LLVMBuilderRef builder, LLVMValueRef lhs, LLVMValueRef rhs, const char *name)
{
    LLVMValueRef args[] = { lhs, rhs };
    return LLVMBuildCall(builder, _jug_builtin_pow_f64, args, 2, name);
}

static LLVMValueRef _jug_build_sin_f64(LLVMBuilderRef builder, LLVMValueRef arg, const char *name)
{
    LLVMValueRef args[] = { arg };
    return LLVMBuildCall(builder, _jug_builtin_sin_f64, args, 1, name);
}

static LLVMValueRef _jug_build_sinh_f64(LLVMBuilderRef builder, LLVMValueRef arg, const char *name)
{
    LLVMValueRef args[] = { arg };
    return LLVMBuildCall(builder, _jug_builtin_sinh_f64, args, 1, name);
}

static LLVMValueRef _jug_build_sqrt_f64(LLVMBuilderRef builder, LLVMValueRef arg, const char *name)
{
    LLVMValueRef args[] = { arg };
    return LLVMBuildCall(builder, _jug_builtin_sqrt_f64, args, 1, name);
}

static LLVMValueRef _jug_build_tan_f64(LLVMBuilderRef builder, LLVMValueRef arg, const char *name)
{
    LLVMValueRef args[] = { arg };
    return LLVMBuildCall(builder, _jug_builtin_tan_f64, args, 1, name);
}

static LLVMValueRef _jug_build_tanh_f64(LLVMBuilderRef builder, LLVMValueRef arg, const char *name)
{
    LLVMValueRef args[] = { arg };
    return LLVMBuildCall(builder, _jug_builtin_tanh_f64, args, 1, name);
}

static LLVMValueRef _jug_build_fmod_f64(LLVMBuilderRef builder, LLVMValueRef lhs, LLVMValueRef rhs, const char *name)
{
    LLVMValueRef args[] = { lhs, rhs };
    return LLVMBuildCall(builder, _jug_builtin_fmod_f64, args, 2, name);
}

static void* _jug_function_map[] = {
    LLVMBuildFAdd,
    LLVMBuildFSub,
    LLVMBuildFMul,
    LLVMBuildFDiv,
    LLVMBuildFNeg,
    _jug_build_comma,
    _jug_build_abs_f64,
    _jug_build_acos_f64,
    _jug_build_asin_f64,
    _jug_build_atan_f64,
    _jug_build_atan2_f64,
    _jug_build_ceil_f64,
    _jug_build_cos_f64,
    _jug_build_cosh_f64,
    NULL,//"EXPR_TYPE_E",
    _jug_build_exp_f64,
    NULL,// EXPR_TYPE_FAC,
    _jug_build_floor_f64,
    _jug_build_ln_f64,
    _jug_build_log_f64,
    NULL,//"EXPR_TYPE_NCR",
    NULL,//"EXPR_TYPE_NPR",
    NULL,//"EXPR_TYPE_PI",
    _jug_build_pow_f64,
    _jug_build_sin_f64,
    _jug_build_sinh_f64,
    _jug_build_sqrt_f64,
    _jug_build_tan_f64,
    _jug_build_tanh_f64,
    _jug_build_fmod_f64,
    0,
};
#define TE_FUN(...) ((LLVMValueRef(*)(__VA_ARGS__))_jug_function_map[n->function])
#define M(e) _jug_expr_compile_expression(builder, n->parameters[e], params)
#define TYPE_MASK(TYPE) ((TYPE)&0x0000001F)
#define ARITY(TYPE) ( ((TYPE) & (TE_FUNCTION0 | TE_CLOSURE0)) ? ((TYPE) & 0x00000007) : 0 )
static LLVMValueRef _jug_expr_compile_expression(LLVMBuilderRef builder, jug_te_expr *n, ina_hashtable_t *params)
{
    switch (TYPE_MASK(n->type)) {
        case TE_CONSTANT: return LLVMConstReal(LLVMDoubleType(), n->value);
        case TE_VARIABLE: {
            LLVMValueRef param;
            ina_hashtable_get_str(params, n->bound, (void**)&param);
            return param;
        }
        case TE_FUNCTION0: case TE_FUNCTION1: case TE_FUNCTION2: case TE_FUNCTION3:
        case TE_FUNCTION4: case TE_FUNCTION5: case TE_FUNCTION6: case TE_FUNCTION7:
            switch (ARITY(n->type)) {
            case 0: return TE_FUN(LLVMBuilderRef, const char*)(builder, te_function_map_str[n->function]);
            case 1: return TE_FUN(LLVMBuilderRef, LLVMValueRef, const char*)(builder, M(0), te_function_map_str[n->function]);
            case 2: return TE_FUN(LLVMBuilderRef, LLVMValueRef, LLVMValueRef, const char*)(builder, M(0), M(1), te_function_map_str[n->function]);
            case 3: return TE_FUN(LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, const char*)(builder, M(0), M(1), M(2), te_function_map_str[n->function]);
            case 4: return TE_FUN(LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, const char*)(builder, M(0), M(1), M(2), M(3), te_function_map_str[n->function]);
            case 5: return TE_FUN(LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, const char*)(builder, M(0), M(1), M(2), M(3), M(4), te_function_map_str[n->function]);
            case 6: return TE_FUN(LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, const char*)(builder, M(0), M(1), M(2), M(3), M(4), M(5), te_function_map_str[n->function]);
            case 7: return TE_FUN(LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, const char*)(builder, M(0), M(1), M(2), M(3), M(4), M(5), M(6), te_function_map_str[n->function]);
            default: return NULL;
            }

        case TE_CLOSURE0: case TE_CLOSURE1: case TE_CLOSURE2: case TE_CLOSURE3:
        case TE_CLOSURE4: case TE_CLOSURE5: case TE_CLOSURE6: case TE_CLOSURE7:
            switch (ARITY(n->type)) {
            case 0: return TE_FUN(void*, LLVMBuilderRef, const char*)(n->parameters[0], builder, te_function_map_str[n->function]);
            case 1: return TE_FUN(void*, LLVMBuilderRef, LLVMValueRef, const char*)(n->parameters[1], builder, M(0), te_function_map_str[n->function]);
            case 2: return TE_FUN(void*, LLVMBuilderRef, LLVMValueRef, LLVMValueRef, const char*)(n->parameters[2], builder, M(0), M(1), te_function_map_str[n->function]);
            case 3: return TE_FUN(void*, LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, const char*)(n->parameters[3], builder, M(0), M(1), M(2), te_function_map_str[n->function]);
            case 4: return TE_FUN(void*, LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, const char*)(n->parameters[4], builder, M(0), M(1), M(2), M(3), te_function_map_str[n->function]);
            case 5: return TE_FUN(void*, LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, const char*)(n->parameters[5], builder, M(0), M(1), M(2), M(3), M(4), te_function_map_str[n->function]);
            case 6: return TE_FUN(void*, LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, const char*)(n->parameters[6], builder, M(0), M(1), M(2), M(3), M(4), M(5), te_function_map_str[n->function]);
            case 7: return TE_FUN(void*, LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, const char*)(n->parameters[7], builder, M(0), M(1), M(2), M(3), M(4), M(5), M(6), te_function_map_str[n->function]);
            default: return NULL;
            }

        default: return NULL;
    }
}
#undef TE_FUN
#undef M
#undef TYPE_MASK
#undef ARITY

static void debug_print(LLVMBuilderRef builder, LLVMModuleRef module, const char *fmt, LLVMValueRef value)
{
    LLVMValueRef format = LLVMBuildGlobalStringPtr(builder, fmt, "format");
    LLVMValueRef printf_function = LLVMGetNamedFunction(module, "printf");
    LLVMValueRef printf_args[] = { format, value };
    LLVMBuildCall(builder, printf_function, printf_args, 2, "printf");
}

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

    LLVMValueRef constant_zero = LLVMConstInt(int32Type, 0, 1);
    LLVMValueRef constant_one = LLVMConstInt(int32Type, 1, 1);

    e->context = LLVMContextCreate();

    /* define the parameter structure for prefilter */
    LLVMTypeRef params_struct = LLVMStructCreateNamed(e->context, "struct.blosc2_prefilter_params");
    LLVMTypeRef *params_struct_types = ina_mem_alloc(sizeof(LLVMTypeRef) * 7);
    params_struct_types[0] = LLVMInt32Type();
    params_struct_types[1] = LLVMArrayType(LLVMPointerType(LLVMInt8Type(), 0), BLOSC2_PREFILTER_INPUTS_MAX);
    params_struct_types[2] = LLVMArrayType(LLVMInt32Type(), BLOSC2_PREFILTER_INPUTS_MAX);
    params_struct_types[3] = LLVMPointerType(LLVMInt8Type(), 0); /* userdata */
    params_struct_types[4] = LLVMPointerType(LLVMInt8Type(), 0); /* out */
    params_struct_types[5] = LLVMInt32Type();
    params_struct_types[6] = LLVMInt32Type();

    LLVMStructSetBody(params_struct, params_struct_types, 7, 0);

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

    LLVMBuilderRef builder = LLVMCreateBuilder(); // FIXME, probably better to build it from context, mem-leak?

    LLVMValueRef param_ptr = LLVMGetParam(f, 0);

    LLVMValueRef local_output;
    LLVMValueRef *local_inputs;
    ina_str_t *local_input_labels;
    LLVMPositionBuilderAtEnd(builder, stackvar_sec);
    {
        local_output = LLVMBuildAlloca(builder, LLVMPointerType(LLVMDoubleType(), 0), "local_output");
        local_inputs = ina_mem_alloc(sizeof(LLVMValueRef*)*var_len); // leaking memory for now
        local_input_labels = ina_mem_alloc(sizeof(ina_str_t)*var_len); // leaking memory for now

        LLVMValueRef ninputs = LLVMBuildStructGEP(builder, param_ptr, 0, "ninputs");
        INA_UNUSED(ninputs); // TODO: compare arg_count with ninputs, return error (constant_one) if different

        LLVMValueRef inputs_ptr = LLVMBuildStructGEP(builder, param_ptr, 1, "inputs_ptr");
        LLVMValueRef inputs = LLVMBuildLoad(builder, inputs_ptr, "inputs");
        for (int i = 0; i < var_len; ++i) {
            local_inputs[i] = ina_mem_alloc(sizeof(LLVMValueRef));

            local_input_labels[i] = ina_str_sprintf("input[%d]", i); // leaking memory for now
            local_inputs[i] = LLVMBuildAlloca(builder, LLVMPointerType(LLVMDoubleType(), 0), ina_str_cstr(local_input_labels[i]));

            /* Load array of inputs */
            LLVMValueRef in_addr = LLVMBuildExtractValue(builder, inputs, i, "inputs[index]");

            /* Cast to value type */
            LLVMTypeRef type_cast = LLVMPointerType(LLVMDoubleType(), 0);
            LLVMValueRef cast_in = LLVMBuildCast(builder, LLVMBitCast, in_addr, type_cast, "cast[double*]");

            /* Store pointer in stack var */
            LLVMBuildStore(builder, cast_in, local_inputs[i]);

            /* Load data array */
            //LLVMValueRef addr = LLVMBuildGEP(builder, cast_in, &index, 1, "buffer[index]");
            //LLVMValueRef cast_addr = LLVMBuildCast(builder, LLVMBitCast, addr, LLVMPointerType(LLVMDoubleType(), 0), "cast[double]");

            /* Load scalar value
            LLVMValueRef val = LLVMBuildLoad(builder, cast_addr, "value");
            LLVMSetMetadata(val, LLVMInstructionValueKind, md_access);
            const char *key = vars[i].name;
            ina_hashtable_set_str(param_values, key, val);*/
        }

        LLVMValueRef out_ptr = LLVMBuildStructGEP(builder, param_ptr, 4, "out_ptr");
        LLVMValueRef out = LLVMBuildLoad(builder, out_ptr, "out");
        LLVMValueRef out_cast = LLVMBuildCast(builder, LLVMBitCast, out, LLVMPointerType(LLVMDoubleType(), 0), "out_cast");
        LLVMBuildStore(builder, out_cast, local_output);

        LLVMBuildBr(builder, loop_len);
    }


    LLVMValueRef len;
    LLVMPositionBuilderAtEnd(builder, loop_len);
    {
        LLVMValueRef out_size_ptr = LLVMBuildStructGEP(builder, param_ptr, 5, "out_size_ptr");
        LLVMValueRef out_size = LLVMBuildLoad(builder, out_size_ptr, "out_size");
        LLVMValueRef out_size_val = LLVMBuildPtrToInt(builder, out_size, LLVMInt32Type(), "out_size_val");

        LLVMValueRef out_typesize_ptr = LLVMBuildStructGEP(builder, param_ptr, 6, "out_typesize_ptr");
        LLVMValueRef out_typesize = LLVMBuildLoad(builder, out_typesize_ptr, "out_typesize");
        LLVMValueRef out_typesize_val = LLVMBuildPtrToInt(builder, out_typesize, LLVMInt32Type(), "out_size_val");

        len = LLVMBuildExactSDiv(builder, out_size_val, out_typesize_val, "calculate_len");
        LLVMBuildBr(builder, entry);
    }

    LLVMValueRef index_addr;
    LLVMPositionBuilderAtEnd(builder, entry);
    {
        index_addr = LLVMBuildAlloca(builder, int32Type, "index");
        LLVMBuildStore(builder, constant_zero, index_addr);
        LLVMBuildBr(builder, condition);
    }

    LLVMPositionBuilderAtEnd(builder, condition);
    {
        LLVMValueRef index = LLVMBuildLoad(builder, index_addr, "[index]");
        LLVMValueRef cond = LLVMBuildICmp(builder, LLVMIntSLT, index, len, "index < len");
        LLVMBuildCondBr(builder, cond, body, end);
    }
    LLVMPositionBuilderAtEnd(builder, body);
    {
        LLVMValueRef md_values_access[] = { LLVMMDString("llvm.access.group",
            (unsigned int)strlen("llvm.access.group")) };
        LLVMValueRef md_access = LLVMMDNode(md_values_access, 1);

        LLVMValueRef md_values[] = { LLVMMDString("llvm.loop.parallel_accesses",
            (unsigned int)strlen("llvm.loop.parallel_accesses")), md_access };
        LLVMValueRef md_node = LLVMMDNode(md_values, 2);

        /*LLVMValueRef md_values_vec[] = { LLVMMDString("llvm.loop.vectorize.enable",
            (unsigned int)strlen("llvm.loop.vectorize.enable")),
            LLVMConstInt(LLVMInt1Type(), 1, 1)
        };
        LLVMValueRef md_node_vec = LLVMMDNode(md_values_vec, 2);*/

        LLVMValueRef index = LLVMBuildLoad(builder, index_addr, "[index]");

        /* Load the scalar values from the inputs */
        for (int i = 0; i < var_len; ++i) {
            LLVMValueRef stack_var = LLVMBuildLoad(builder, local_inputs[i], "load_stackvar");
            LLVMValueRef addr = LLVMBuildGEP(builder, stack_var, &index, 1, "buffer[index]");
            //LLVMValueRef cast_addr = LLVMBuildCast(builder, LLVMBitCast, addr, LLVMPointerType(LLVMDoubleType(), 0), "cast[double]");

            /* Load scalar value */
            LLVMValueRef val = LLVMBuildLoad(builder, addr, "value");
            LLVMSetMetadata(val, LLVMInstructionValueKind, md_access);
            const char *key = vars[i].name;
            ina_hashtable_set_str(param_values, key, val);
        }

        /* compute the expression */
        LLVMValueRef result = _jug_expr_compile_expression(builder, expression, param_values);

        /* store the result */
        LLVMValueRef local_out_ref = LLVMBuildLoad(builder, local_output, "local_output");
        LLVMValueRef out_addr = LLVMBuildGEP(builder, local_out_ref, &index, 1, "out_addr");
        LLVMValueRef store = LLVMBuildStore(builder, result, out_addr);
        LLVMSetMetadata(store, LLVMInstructionValueKind, md_access);

        LLVMValueRef loop_latch = LLVMBuildBr(builder, increment);
        LLVMSetMetadata(loop_latch, LLVMInstructionValueKind, md_node);
    }
    LLVMPositionBuilderAtEnd(builder, increment);
    {
        LLVMValueRef index = LLVMBuildLoad(builder, index_addr, "[index]");
        LLVMValueRef indexpp = LLVMBuildAdd(builder, index, constant_one, "index++");
        LLVMBuildStore(builder, indexpp, index_addr);
        LLVMBuildBr(builder, condition);
    }
    LLVMPositionBuilderAtEnd(builder, end);

    LLVMBuildRet(builder, constant_zero);

    ina_hashtable_free(&param_values);

    return f;
}

static void _jug_apply_optimisation_passes(jug_expression_t *e)
{
    /*
     * FIXME With OptLevel > 0 or LLVMAddInstructionCombiningPass the call to
     * LLVMRunPassManager gets stuck.
     * Other passes, such as LLVMAddScalarReplAggregatesPassSSA, make the code
     * fail with "SCEVAddExpr operand types don't match!"
     */

    jug_util_set_svml_vector_library();

    LLVMPassManagerBuilderRef pmb = LLVMPassManagerBuilderCreate();
    jug_utils_enable_loop_vectorize(pmb);
    LLVMPassManagerBuilderSetOptLevel(pmb, 2); // Opt level 0-3

    // Module pass manager
    LLVMPassManagerRef pm = LLVMCreatePassManager();
    LLVMAddAnalysisPasses(tm_ref, pm);
    LLVMPassManagerBuilderPopulateModulePassManager(pmb, pm);

    LLVMAddLoopVectorizePass(pm);
    LLVMAddSLPVectorizePass(pm);

    // Run
    LLVMRunPassManager(pm, e->mod);

    // Dispose
    LLVMDisposePassManager(pm);
    LLVMPassManagerBuilderDispose(pmb);
}

static void _jug_declare_printf(LLVMModuleRef mod)
{
    LLVMTypeRef printf_args_ty_list[] = { LLVMPointerType(LLVMInt8Type(), 0) };
    LLVMTypeRef printf_ty =
        LLVMFunctionType(LLVMInt64Type(), printf_args_ty_list, 0, 1);
    LLVMAddFunction(mod, "printf", printf_ty);
}

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
    error = LLVMCreateExecutionEngineForModule(&e->engine, e->mod, &message);
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

    LLVMBool llvm_error;
    llvm_error = LLVMInitializeNativeTarget();
    llvm_error = LLVMInitializeNativeAsmPrinter();
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

    tm_ref =
        LLVMCreateTargetMachine(target_ref, _jug_def_triple, "", "",
            LLVMCodeGenLevelDefault,
            LLVMRelocDefault,
            LLVMCodeModelJITDefault);
    _jug_data_ref = LLVMCreateTargetDataLayout(tm_ref);

    return INA_SUCCESS;
}

INA_API(void) jug_destroy()
{
    /* FIXME: do proper cleanup of LLVM stuff */
}

INA_API(ina_rc_t) jug_expression_new(jug_expression_t **expr)
{
    LLVMModuleRef m;
    *expr = (jug_expression_t*)ina_mem_alloc(sizeof(jug_expression_t));
    (*expr)->mod = LLVMModuleCreateWithName("expr_engine");
    m = (*expr)->mod;

    _jug_declare_printf(m);
    _jug_declare_abs_f64(m);
    _jug_declare_acos_f64(m);
    _jug_declare_asin_f64(m);
    _jug_declare_atan_f64(m);
    _jug_declare_atan2_f64(m);
    _jug_declare_ceil_f64(m);
    _jug_declare_cos_f64(m);
    _jug_declare_cosh_f64(m);
    _jug_declare_exp_f64(m);
    _jug_declare_floor_f64(m);
    _jug_declare_ln_f64(m);
    _jug_declare_log_f64(m);
    _jug_declare_pow_f64(m);
    _jug_declare_sin_f64(m);
    _jug_declare_sinh_f64(m);
    _jug_declare_sqrt_f64(m);
    _jug_declare_tan_f64(m);
    _jug_declare_tanh_f64(m);
    _jug_declare_fmod_f64(m);

    return INA_SUCCESS;
}

INA_API(void) jug_expression_free(jug_expression_t **expr)
{
    INA_VERIFY_FREE(expr);
    if ((*expr)->engine != NULL) {
        LLVMDisposeExecutionEngine((*expr)->engine);
    }
    // FIXME
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
    //buffer = LLVMCreateMemoryBufferWithMemoryRangeCopy(llvm_bc, llvm_bc_len, "udf");

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
        return INA_ERR_INVALID_ARGUMENT;
    }
    _jug_expr_compile_function(e, "expr_func", expression, num_vars, te_vars);
    jug_te_free(expression);

    if (_jug_prepare_module(e, true)) {
        return INA_ERR_FAILED;
    }

    *function_addr = LLVMGetFunctionAddress(e->engine, "expr_func");

    return INA_SUCCESS;
}

/*int main(int argc, char **argv)
{




    int argc_eval = argc - 2;
    char **argv_eval = argv + 2;

    blosc2_prefilter_params params = {
            .ninputs = argc_eval,  // number of data inputs
            //.inputs = inputs,  // the data inputs
            .input_typesizes[0] = sizeof(double),  // the typesizes for data inputs
            .user_data = NULL,  // user-provided info (optional)
            //.out = out,  // automatically filled
            .out_size = 10 * sizeof(double),  // automatically filled
            .out_typesize = sizeof(double),  // automatically filled
    };

    te_variable *vars = malloc(sizeof(te_variable)*argc_eval);
    memset(vars, 0, sizeof(te_variable)*argc_eval);

    double *out = malloc(10 * sizeof(double));
    // Fill the parameters of the prefilter
    for (int i = 0; i < argc_eval; ++i) {
        vars[i].name = strtok(argv_eval[i], "=");
        double val = strtod(strtok(NULL, "="), NULL);
        params.inputs[i] = malloc(sizeof(double) * 10);
        double *b = (double*)params.inputs[i];
        for (int z = 0; z < 10; ++z) {
            b[z] = val;
        }
    }

    params.out = (uint8_t*)out;

    //_jug_expr_t *expression = _jug_expr_parse_expression(&argv[1]);


    typedef int(*fun_t)(blosc2_prefilter_params *params);
    uint64_t fun_addr = 0;
    fun_t fun = (fun_t)fun_addr;

    // invoke jitted code

    fun(&params);

    double *test = (double*)params.out;
    for (int i = 0; i < 10; ++i) {
        printf("Result: %f\n", test[i]);
    }



    return 0;
}
*/

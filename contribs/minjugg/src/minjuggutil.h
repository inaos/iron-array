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

#ifndef _MINJUGGUTIL_H_
#define _MINJUGGUTIL_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LLVMOpaquePassManagerBuilder *LLVMPassManagerBuilderRef;
typedef struct LLVMOpaqueModule *LLVMModuleRef;
typedef struct LLVMOpaqueExecutionEngine *LLVMExecutionEngineRef;

int jug_util_set_svml_vector_library(void);
int jug_utils_enable_loop_vectorize(LLVMPassManagerBuilderRef PMB);
int jug_utils_create_execution_engine(LLVMModuleRef mod, LLVMExecutionEngineRef *ee);
const char * jug_utils_get_cpu_string(void);

#ifdef __cplusplus
}
#endif

#endif

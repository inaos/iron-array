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

#ifndef _MINJUGGUTIL_H_
#define _MINJUGGUTIL_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LLVMOpaquePassManagerBuilder *LLVMPassManagerBuilderRef;

int jug_util_set_svml_vector_library(void);
int jug_utils_enable_loop_vectorize(LLVMPassManagerBuilderRef PMB);

#ifdef __cplusplus
}
#endif

#endif

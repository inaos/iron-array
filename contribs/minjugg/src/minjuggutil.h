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

//#include <libinac/lib.h>

#include <llvm-c/Core.h>
#include <llvm-c/Target.h>

#ifdef __cplusplus
extern "C" {
#endif

int jug_util_get_svml_vector_library(const char *triple, LLVMTargetLibraryInfoRef *tli);

#ifdef __cplusplus
}
#endif

#endif

/*
 * Copyright (C) 2018 Francesc Alted, Aleix Alcacer.
 * Copyright (C) 2019-present Blosc Development team <blosc@blosc.org>
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

#ifndef IARRAY_GEMM_H
#define IARRAY_GEMM_H

INA_API(ina_rc_t) iarray_gemm(iarray_context_t *ctx,
                              iarray_container_t *a,
                              iarray_container_t *b,
                              iarray_container_t *c);

#endif //IARRAY_GEMM_H

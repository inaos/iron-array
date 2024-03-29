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

#ifndef IARRAY_GEMV_H
#define IARRAY_GEMV_H

INA_API(ina_rc_t) iarray_gemv(iarray_context_t *ctx,
                              iarray_container_t *a,
                              iarray_container_t *b,
                              iarray_container_t *c);

#endif //IARRAY_GEMV_H

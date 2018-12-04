/*
 * Copyright INAOS GmbH, Thalwil, 2018.
 * Copyright Francesc Alted, 2018.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of INAOS GmbH
 * and Francesc Alted ("Confidential Information"). You shall not disclose such Confidential
 * Information and shall use it only in accordance with the terms of the license agreement.
 *
 */

#include <libiarray/iarray.h>

#include <iarray_private.h>

INA_API(ina_rc_t) iarray_rand(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    iarray_random_ctx_t *random_ctx,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(container);

    INA_RETURN_IF_FAILED(_iarray_container_new(ctx, dtshape, store, flags, container));

    /* implement rand */

    return INA_SUCCESS;
}

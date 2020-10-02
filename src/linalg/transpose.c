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


ina_rc_t iarray_linalg_transpose(iarray_context_t *ctx, iarray_container_t *a) {
    INA_VERIFY_NOT_NULL(ctx);
    if (a->dtshape->ndim != 2) {
        IARRAY_TRACE1(iarray.error, "The container dimension is not 2");
        return INA_ERROR(IARRAY_ERR_INVALID_NDIM);
    }

    if (a->transposed == 0) {
        a->transposed = 1;

    }
    else {
        a->transposed = 0;
    }

    if (a->catarr->storage == CATERVA_STORAGE_BLOSC && blosc2_has_metalayer(a->catarr->sc, "iarray") > 0) {
        uint8_t *content;
        uint32_t content_len;
        blosc2_get_metalayer(a->catarr->sc, "iarray", &content, &content_len);
        *(content + 2) = *(content + 2) ^ 64ULL;
        blosc2_update_metalayer(a->catarr->sc, "iarray", content, content_len);
        free(content);
    }

    int64_t aux[IARRAY_DIMENSION_MAX];
    for (int i = 0; i < a->dtshape->ndim; ++i) {
        aux[i] = a->dtshape->shape[i];
    }
    for (int i = 0; i < a->dtshape->ndim; ++i) {
        a->dtshape->shape[i] = aux[a->dtshape->ndim - 1 - i];
    }
    for (int i = 0; i < a->dtshape->ndim; ++i) {
        aux[i] = a->storage->chunkshape[i];
    }
    for (int i = 0; i < a->dtshape->ndim; ++i) {
        a->storage->chunkshape[i] = aux[a->dtshape->ndim - 1 - i];
    }
    return INA_SUCCESS;
}
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

#include "iarray_private.h"
#include <libiarray/iarray.h>

/*
 * Check if a file exist using fopen() function.
 *
 * Return true if the file exist otherwise return false
 */
bool _iarray_file_exists(const char * urlpath)
{
    INA_VERIFY_NOT_NULL(urlpath);

    /* try to open file to read */
    FILE *file;
    if ((file = fopen(urlpath, "r")) != NULL) {
        fclose(file);
        return true;
    }
    return false;
}

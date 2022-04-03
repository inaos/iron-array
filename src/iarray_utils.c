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

#include "iarray_private.h"
#include <libiarray/iarray.h>
#include <sys/stat.h>

/*
 * Check if a path (file or directory) exists using stat() function.
 *
 * Return true if the path exists otherwise return false
 */
bool _iarray_path_exists(const char * urlpath)
{
    INA_VERIFY_NOT_NULL(urlpath);

    struct stat statbuf;
    /* try to access the path to read */
    if (stat(urlpath, &statbuf) == 0){
        return true;
    }
    return false;
}

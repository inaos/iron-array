/* temporary mock of INAC */

#ifndef __INAC__
#define __INAC__

#include <stdlib.h>

#define INA_API(x)
typedef int ina_rc_t;

typedef void* ina_mempool_t;

inline void* ina_mempool_dalloc(ina_mempool_t *mp, size_t size)
{
	return malloc(size);
}

#endif
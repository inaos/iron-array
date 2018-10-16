//
// Created by Francesc Alted on 15/10/2018.
//

#ifndef IARRAY_TEST_COMMON_H
#define IARRAY_TEST_COMMON_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include "blosc.h"


/* This is MinUnit in action (http://www.jera.com/techinfo/jtns/jtn002.html) */
#define mu_assert(message, test) do { if (!(test)) return message; } while (0)
#define mu_run_test(test) do \
    { char *message = test(); tests_run++;                          \
      if (message) { printf("%c", 'F'); return message;}            \
      else printf("%c", '.'); } while (0)

extern int tests_run;

#define KB  1024
#define MB  (1024*KB)
#define GB  (1024*MB)


// Check that two super-chunks with the same partitions are equal
bool test_schunks_equal_double(blosc2_schunk* sc1, blosc2_schunk* sc2) {
	size_t chunksize = (size_t)sc1->chunksize;
	int nitems_in_chunk = (int)chunksize / sc1->typesize;
	double *buffer_sc1 = malloc(chunksize);
	double *buffer_sc2 = malloc(chunksize);
	for (int nchunk=0; nchunk < sc1->nchunks; nchunk++) {
		int dsize = blosc2_schunk_decompress_chunk(sc1, nchunk, buffer_sc1, chunksize);
		dsize = blosc2_schunk_decompress_chunk(sc2, nchunk, buffer_sc2, chunksize);
		for (int nelem=0; nelem < nitems_in_chunk; nelem++) {
			double vdiff = fabs(buffer_sc1[nelem] - buffer_sc2[nelem]);
			if (vdiff > 1e-6) {
				printf("Values differ in (%d nchunk, %d nelem) (diff: %f)\n", nchunk, nelem, vdiff);
				free(buffer_sc1);
				free(buffer_sc2);
				return false;
			}
		}
	}
	free(buffer_sc1);
	free(buffer_sc2);
	return true;
}


#endif //IARRAY_TEST_COMMON_H

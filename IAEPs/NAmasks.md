# Not Available (NA) Masks (IAEP #1)

__Author__: Francesc Alted

__Initial Date__: 2019-05-02

## Rational 

In the real world, it is quite common that datasets have entries were data is Not Available (for multiple reasons, like a sensor failed, or a network failure, or just data the user don't want to deal with).  It would be a great addtion to introduce support for NA masks in IronArray.  The main path for leveraging NA's would be via iterators as well as the computational engine.
  
Following there is a proposal on how to implement NA masks within IronArray.

## Introduce two levels masks: item-wise and partition-wise

__Item-wise mask__: It tells whether an item should be masked out or not.

__Partition-wise mask__: They will tell whether a partition only holds masked values or not.  This opens the possibility to setup very large matrices that are sparse, and still be able to (efficiently) operate with them.

The masks should be added inside a `__item_mask__` and `__part_mask__` [metalayers](https://github.com/Blosc/c-blosc2/blob/master/examples/frame_metalayers.c#L77).  These should be part of the Caterva layer because the caterva_get_slice_buffer() and friends will need to be aware of masks.

The existence of the partition mask is mainly for efficiency: only in the case that the mask bit for the partition is set, there should be a block of mask bits in the item mask; if not, there is no need to store the block of the item mask for the partition, requiring less memory consumption for holding the masks.  Also, by combining partition and item masks this way, we can hold much more items in a dataset having a NA mask.
  
During the construction of a masked dataset, the masks should be kept temporaly in-memory so we need at least 1 byte per item. The worst memory-comsumption scenario is, as the metalayers can only currently host 2^31 bytes each, and we can store 1 NA bits in a byte, a 2 GB mask can hold up to 2^31 items (2 Gitems).  However, with the help of the partition mask, even this 2 Gitems limit can be surpassed in the case whole partitions can be masked out, while not requiring more than 2 GB.

__Note 1__: The 2 Gitem limit can be removed if C-Blosc2 would allow to store general frames as metalayers instead of just plain chunks (2 GB limit).  A ticket has been opened for this: https://github.com/Blosc/c-blosc2/issues/56.

__Note 2__: In the future we may want to pack 8 NA mask bit into 1 actual byte, but this would make code more complex.  For now, using 1 byte per NA bit should be more than enough; furthermore, compression would be responsible to remove much of the 1 bit -> 1 byte overhead in storage.

## Integration with iterators

Masks should be used mainly in IronArray iterators.  For item-wise iterators, just a new field to the `value` struct should be needed, say `.item_mask`.  For block iterators, the `value` struct would need a couple of new fields, say `.item_mask` and `.part_mask`.  For example, for writing block iterators the way to setup a mask would be like this:

```
    while (iarray_iter_write_block_has_next(I)) {
        iarray_iter_write_block_next(I);

        int64_t nelem = 0;
        int64_t inc = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            nelem += val.elem_index[i] * inc;
            inc *= c_x->dtshape->shape[i];
        }
        for (int64_t i = 0; i < val.block_size; ++i) {
            if (nelem % 2) {
                ((double *)val.pointer)[i] = (double) nelem + i;
            }
            else {
                val.item_mask[i] = false;
            }
        }
    }
```

In this case, the even values are masked out.  For odd values, the user won't need to set it to true explicitly because the `value.*_mask` will be set to true automatically for each iteration.

For reading, we have two possibilities:

1) Not passing the masked out value to the user.

2) Passing the masked out value to the user, but with the `value.*_mask` value set to false.

For now, I am leaning more for 1) because this way we can remove a lot of overhead to the iterator.  But in case we decide to go for 2) the user will be able to quickly mask out values with something along these lines:

```
    while (iarray_iter_read_block_has_next(I2) && iarray_iter_read_block_has_next(I3)) {
        iarray_iter_read_block_next(I2);
        iarray_iter_read_block_next(I3);
        if (val2.part_mask && val3.part_mask) {
            for (int64_t i = 0; i < val2.block_size; ++i) {
                if (val2.item_mask[i] && val3.item_mask[i]) {    
                    INA_TEST_ASSERT_EQUAL_FLOATING(((double *) val2.pointer)[i], ((double *) val3.pointer)[i]);
                }
            }
        }
    }
```

Perhaps it would be nice if the user can configure the iterators to behave either as 1) or as 2).  Probably this should be the way to go.

## Integration with the computation engine

For leveraging masked values to a maximum, they should be fully integrated into the computation engine.  This engine is already based on iterators, so we would already have the mask info available; however, the current computationals algorithms.  One way to do this is to get the masks of operands and do a `bit-wise and` prior to do the computations.  When the mask of the result would be computed, then we could either:
 
 1) Do a copy of values to contiguous buffers for evaluation.
 
 2) Do not modify the evaluation functions and carry out operations with NA values in operands as regular values.  This is possible because NA values are normally filled with zeros, and most of operations with zeros are supported in floating point arithmetic.  __Warning__: In the future, a division by 0 in integers may pose some problems, but let's worry about this later.
   
 Initially at least, I'd shoot for 1) because I find it more future proof and, requiring less computations, it can be faster than 2).  On the other hand, 1) does requires a copy.  However, if the operands fit in L1 (and that is typical situation for our current computation engine), perhaps the overhead of the copy should be pretty close to none.  Some benchmarking should help prior to making a decision.

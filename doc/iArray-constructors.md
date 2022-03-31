# iArray constructors

This file contains a brief description of the new iArray constructors: the **block-by-block** constructor and the **element-by-element** constructor.

Both constructors are meant to build arrays using all the c-blosc2 machinery.
In addition, they are designed to be completly modular, so users (ironArray) can define many constructors in a very easy way.


In both constructors, users can define its custom parameters (`[level_name]_custom_info`) in a three different levels:

1. **Array level**: Users can define actions that are done before calling to the constructor using the `custom_array_info` pointer.

2. **Chunk level**: In the same way, users can define actions that are done before each chunk is going to be compressed. To do this, users can use `chunk_init_fn()` and `chunk_destroy_fn()` to fill `custom_chunk_info` pointer.

3. **Block level**: For each block that is processed by blosc, user can also define actions to be done. In this case, users should define `block_init_fn()` and `block_destroy_fn()` to create `custom_block_info` pointer.


Moreover, the constructors define some params called `[level_name]_info`.
For example, the `block_info` contains information about which block is proccesed: where the block starts inside the array, which thread is processing it, the block shape without padding...

NOTE: All blocks of code in this file are pseudocode
## Block-by-block constructor

### Algorithm

```
def function_that_uses_constructor():
    def custom_array_info
    ... // Init custom_array_info

    constructor_block(custom_array_info, ...)

    ... // Destroy custom array info

```

```
def constructor_block(custom_array_info, ...):
    def array_info
    
    for chunk in chunks:
        def chunk_info
        def custom_chunk_info = NULL
        chunk_init_fn()  // Init custom_chunk_info
        
        for block in blocks:  // In parallel inside blosc
            def block_info
            def custom_block_info = NULL
            block_init_fn()  // Init custom_block_info

            generator_fn()

            block_destroy_fn()  // Destroy custom_block_info
        
        chunk_destroy_fn()  // Destroy custom_chunk_info

```

## Element-by-element constructor

This constructor is a particular case of the block-by-block constructor.

### Algorithm

```
def generator_fn():
    for item in items:
        def item_info
        def custom_item_info = NULL

        item_generator_fn()

```

```
def constructor_element(custom_array_info, ...):
    constructor_block(custom_array_info, generator_fn, ...)
```



In the future the `generator_fn()` can be implementend like this:

```
def generator_fn():
    for item in items:
        def item_info
        def custom_item_info = NULL

        item_init_fn()  // Init custom_item_info

        item_generator_fn()

        item_destroy_fn()  // Destroy custom_item_info

```

### Example

Let's see how a tri array can be generated using the element-by-element constructor.
To do this, we just have to define the following function:

```
def item_generator_fn():
     k = array_custom_info.k
    return 1 if item_info.index[1] <= item_info.index[0] + k else 0
```

```
def tri(k):
    def custom_array_info
    custom_array_info.k = k
    constructor_element(custom_array_info, item_generator_fn)
```

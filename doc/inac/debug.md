# Debug
## Overview

INAC provides a set of macro to for tracing and assertion.

## Usage

### Tracing
Enable tracing by defining the trace level. To define the level 
use compile definition `INA_TRACE_LEVEL`. The trace level can be 
between 0 (disabled) and 3. Default is level `1`

Write trace output using `INA_TRACEx()` macro.

    INA_TRACE1("Trace level  %d is enabled", 1)
    INA_TRACE2("Trace level  %d is enabled", 2)
    INA_TRACE3("Trace level  %d is enabled", 2)
    
One can force trace output with `INA_TRACE()`

    INA_TRACE("Always printed");
    
    
The trace functionality is available only for debug builds.


### Asserting
INAC provide a set of assert macro to simply writting assertion

Assertion for not implemented code
    
    INA_NOT_IMPL
    
Arbitrary assertion
  
    INA_ASSERT(3 != 2)
    
Assertion assuming false
    
    INA_ASSERT_FALSE(v)

Assertion assuming true

    INA_ASSERT_TRUE(v)
    
Assertion assuming NULL
    INA_ASSERT_NULL(my_var)
    
Assertion assuming not NULL

    INA_ASSERT_NOT_NULL(my_var)

Assertion assuming equality 

    INA_ASSERT_EQUAL(3, my_counter)

Assertion assuming inequality 
    
    INA_ASSERT_NOTEQUAL(3, my_counter)
    
Assertion assuming return code INA_SUCCESS   
 
    INA_ASSERT_SUCCESS(init())
    
Assertion assuming return code INA_FAILURE
    
    INA_ASSERT_FAILURE(rc)

Assuming a return code indicating success

    INA_ASSERT_SUCCEED(should_succeed())
    
Assuming a return code indicating failure
    
    INA_ASSERT_NOTSUCCEED(should_not_succeed())        

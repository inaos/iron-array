# Error handling

A good error handling should know as much as possible about an error. Easy and fast 
access to the error information are also important. That's why INAC stores 
 error information in one single value. 
We call it 'Return Code' or simply RC. RC is defined by `ina_rc_t` which is 
in fact a 64bit unsigned integer value and can contains an error indicator, 
API version information, Native OS error and application error message. 

INAC pursues also the goal of understandable human error messages therefore the 
error code contains information on adjective / verb, possible negation 
and subject. 

### Define Error Messages
It's best practice to define the error messages instead of using the already 
defined built-in ones directly. For Instance :

    #define INAWS_ERR_CONNECT       INA_ERR_NOT_CONNECTED
    #define INAWS_ERR_NOCONNECTION  INA_ES_CONNECTION|INA_ERR_NOT_CONNECTED

### Basic usage

Generally one will set error state by using `INA_ERROR()` 
    
    ina_rc_t inaws_connect(const char* host, int port) {
            if (!connect(host, port)) {
               return INA_ERROR(INAWS_ERR_NOCONNECTION);
            }
            ...
        }
        return INA_SUCCESS;
     }
     
And check for successful result using `INA_SUCCEED()`

     ....
     if (INA_SUCCEED(inaws_connect("localhost", 2222))) {
          ....
     }
      
.. or `INA_FAILED()` for failure. One can use `ina_err_strerror()` to 
get formatted full error message. To get last error state use 
`ina_err_get_rc()`.

     if (INA_FAILED(inaws_connect("localhost", 2222))) {
        printf("%s", ina_err_strerror(ina_err_get_rc()));
        return EXIT_FAILURE;
     }
     
    
We can get access to the reason of failure by `INA_RC_ERROR` macro.

     ....
     if (INA_FAILED(inaws_connect("localhost", 2222))) {
        switch (INA_RC_ERROR(ina_err_get_rc())) {
            case INA_ERR_IN_USE:
                ...
        }
   switch (INA_RC_REASON(rc)) {
      case INAWS_ERR_CONNECT:
         .....

    
    
### Capture native errors 
Use `INA_OS_ERROR()` to capture the last occurred native error. This is errno on
unix based os and on Windows platforms the return value of return 
`GetLastError()`.
    
    ina_rc_t openfile(const char* fn)
    {
        f = fopen("test.csv", "r");
        if (f == NULL) {
            INA_OS_ERROR(INA_ES_FILE|INA_ERR_NOT_OPEN);
        }
        return INA_SUCCESS
    }

To extract the native error code use the `INA_RC_ERRNO()` macro.    
    
    ...
    if (INA_FAILED(openfile(fn)) {
        if (INA_RC_ERRNO(ina_err_get_rc()) == EFULL) {
            
    
   

### Capture third party error
Under certain circumstances it makes sense to return errors from third 
party libraries. This can be done by calling `ina_err_set_rc()` and 
generating the error code with `INA_RC_PACK()`. The best way to doing this
would be to define a macro. _INA_RC_PACK_ requires to argument: the first
argument must be INAC error and the second one a user defined error.

     #define LIBSSH_ERROR(x) ina_err_set_rc(INA_RC_PACK((x), (libssh2_session_last_errno())))

     ...
     if (libssh2_session_init() == -1) {
        return LIBSSH_ERROR(INA_ERR_FAILED);
     }
     
### Reset and clear

To reset the error state use `ina_err_reset()`. 

   /* make sure error state is clean */
   ina_err_reset();
   /* do the work now */
   if (INA_FAILED(inaws_server_start())) {

To clear the error state use `ina_err_clear()`. 

     INA_API(ina_rc_t) inaws_try_connect(void) {
         if (INA_FAILED(inaws_connect("localhost", 2222))) {
             return ina_err_clear(ina_err_get_rc());
             ...

By clearing the error state will only remove the error bit on the RC. 
the reset of a RC will set the global error state to `INA_SUCCESS`.


### Using checkpoint
INAC provides the 3 ways to define checkpoints.


Hard checkpoints. Abort the running process if result is a
In case of failure the execution will jump to this mark
   
       INA_MUST_SUCCEED(inaws_connect("localhost", 2222));
     
  
  
Checkpoints based on INAC error. One need to declare a `fail` label.
In case of failure the execution will jump to this mark
     
          ...
          IARRAY_FAIL_IF_ERROR(inaws_connect("localhost", 2222))
          ...
       fail:
           inaws_destroy();
           return ina_err_get_rc();
   
   
Checkpoint based on a condition. One need to declare a `fail` label.   

          ...
          IARRAY_FAIL_IF(ctx->h != NULL))
          ....
        fail:
           inaws_destroy();
           return INA_EROR(INA_ERR_FAILED);
       
    
   
### The RC in the deep

  
    RC 64bit mask
    
    |- 16bit header -|-------- 48bit descriptor  --------------------------|
    |                |                 |-- 32bit error message ------------|
    |                |                          |9bit code|                |
    EVVVVVVV|RRRRRRRR|OOOOOOOO|OOOOOOOO|UUUUUUUU|AAAAAAAA|NSSSSSSS|SSSSSSSS|
    |    |       |          |              |         |    |    |
    |    |       |          |              |         |    |    |
    |    |       |          |              |         |    |    +---> 15 bit - Subject
    |    |       |          |              |         |    +-------->  1 bit - Negate flag
    |    |       |          |              |         +------------->  8 bit - Adjective/verb
    |    |       |          |              +----------------------->  8 bit - User defined data
    |    |       |          +--------------------------------------> 16 bit - Native OS error
    |    |       +------------------------------------------------->  8 bit - API revision
    |    +--------------------------------------------------------->  8 bit - API version    
    +-------------------------------------------------------------->  1 bit - Error indicator


#### Error indicator
The highest bit is used (if ON) to indicate if the RC should be treated as a
failure.

One can clear the error indicator with `ina_err_clear()`. This can be useful 
when handling an error situation and one need to return information about the 
handled error.

Use `INA_RC_EFLAG()` to extract the error indicator.

#### API version/revision information
INAC store the API major and minor(revision) version in the RC. Major version 
number is defined by _INA_ERROR_VER_. Use _INA_RC_VER()_ to extract the version
 number from a RC.

Revision number is defined by _INA_ERROR_REV_. Use _INA_RC_REV()_ to extract 
the revision number from a RC.

#### Native OS error 
Contains the captured native error code. Use `INA_RC_ERRNO()` to extract the
error.

#### Error message
An error message has the form 'ADJECTIVE/VERB'  or 'NOT ADJECTIVE/VERB' like 
"Empty", "Not valid", "Not initialized", "Not running", "Unavailable", etc.

Additionally, one can be more specific on the error subject in the form of
'SUBJECT + (NOT) + ADJECTIVE/VERB'.  In this way errors are also supported like
"Argument invalid", "Network not initialized", "File not out", "Disk full", etc.

Therefore an error message consists of an error and optionally a subject.

INAC subjects and adj/verb are defined in `libinac/error.h` and should cover 
almost any kind of error for an application/library.

##### Error
Use `INA_RC_ERROR()` to extract the error from a error message. For 
instance `INA_RC_ERROR(INA_ES_DISK|INA_ERR_FULL)` returns `INA_ERR_FULL` and
for `INA_RC_ERROR(INA_ES_DISK|INA_ERR_NOT_AVAILABLE)` returns `INA_ERR_NOT_AVAILABLE`.
In detail the error consist of an error code and the negate flag.

###### Negate flag
The negate flag indicate whenever the ADJECTIVE/VERB has the negative form.
Use `INA_RC_NFLAG()` to extract the negate flag from a RC. For instance 
For instance `INA_RC_ERROR(INA_ES_DISK|INA_ERR_FULL)` returns `0` and
for `INA_RC_ERROR(INA_ES_DISK|INA_ERR_NOT_AVAILABLE)` returns `1`.

###### Code
The code defines the ADJECTIVE/VERB. There are a max of 256 predefined 
attribute/verb code. See in error.h 
Use `INA_RC_CODE()` to extract the adjective/verb from a RC. For instance 
For instance `INA_RC_ERROR(INA_ES_DISK|INA_ERR_FULL)` returns `INA_ES_DISK`.

##### Subject
INAC provides standard subjects which can be used to compose RCs. INAC provides
already a set of useful subject. See in error.h. To easy identify subject, they
should be prefixed by `INA_ES_`.

###### User defined subject code
Subjects are identified by an 15 bit integer. Therefor there are 32767 possible 
subjects. The first 1024 are reserved by INAC libraries.
  
Subjects can be app-defined. For this purpose INAC provides a user defined 
dictionary callback function. Subjects are resolved by calling the user-provided
dictionary function of type `ina_err_subject_cb_t` which is registered 
by `ina_err_register_dict()`. 

Define your user defined subjects:

    #define INAWS_ES_WORKSTATION      INA_ES_USER_DEFINED+1
    
Then define the error message combining adjective/verb and subject

    #define INAWS_ERR_WS_NOT_FOUND     INAWS_ES_WORKSRATION|INA_ERR_NOT_FOUND
    
Declare and implement a dictionary callback for the user defined subjects.

    static const char* __get_err_getsubject(int id) {
        switch (id) {
            case INAWS_ES_WORKSTATION:
                return "WORKSTATION";
            default:
                return "";
        }
    }

Register your dictionary callback asap at program startup
    
    int main(int argc,  char** argv) {
        ina_err_register_dict(__get_err_subject);
    
        ...
     }
    
    
##### User data
An error message can contain 8 bits of arbitrary user data. To set
the user defined bits use `ina_err_set_ubits()` and `ina_err_get_ubits()`
to retrieve them from a RC. This can be useful for 
 
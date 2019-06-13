## Testing

### Tracing

Tracing feature can be enabled an disabled by combiler time settings 
`INA_TRACE_ENABLED`.  Also the tracing level can be define at compile time. 
The library know about 3 tracing levels. Trace messages are ended by a 
newline "\n" 

INAC provides 2 macros which can be used for print debug messages when DEBUG 
is defined

     INA_TRACE
     INA_TRACE_MSG

Use `INA_TRACE_MSG` to print simple messages and `INA_TRACE` to print debug 
messages having var args.

    INA_TRACE_MSG("Server started");
    INA_TRACE("Buffer size is %d", bufsize);
    INA_TRACE1("Same as the %s macro", "INA_TRACE");
    INA_TRACE2("A bit more %s trace", "detailed");
    INA_TRACE3("A %s trace", "fully detailed"); 

### Unit testing

INAC provides a built-in test framework. This framework is almost independent 
from the library itself. 

#### Features  

 * Easy adding tests with minimal effort. Non header files required.
 * Supports test suites
 * Supports fixtures (setup, teardown)
 * Easy to parse output
 * Colored output
 * Supports skipping
 * Minimal memory footprint (no allocations)
 * Supports test helpers
 * Working the same way on Linux/OS-X/Win 
 * Support tap and junit result format
 
Possibles improvements:

 * Possibility to add small description to each test for documentation purpose.
 * Variable output format
 * Display elapsed time

#### Adding tests 

To add your first test to a test suite simply the following lines of code.

    INA_TEST(my_suite, my_first_test_with_inac) {
        INA_ASSERT_FLOATING(1.0, 1.0);
    }


#### Adding fixtures  

To added fixtures to your test use `INA_TEST_FIXTURE` macro. Fixtures need a 
fixture data struct which is defined by `INA_TEST_DATA` macro.  Optionally you
can define a setup and teardown for your test. Setup and Teardown is call on 
any test in the suite.  Fixture data is passed to Setup/Teardown and Run of 
any test in the suite.  Follow the next sample. 

    INA_TEST_DATA(iscp_tcp) {
        ina_iscp_ctx_t *iscp;
    };

    INA_TEST_SETUP(iscp_tcp) {
        ina_iscp_create_tcp(&data->iscp, "127.0.0.1", 9999);
    }

    INA_TEST_TEARDOWN(iscp_tcp) {
        ina_iscp_destroy(&data->iscp);
    }

    INA_TEST_FIXTURE(iscp_tcp, send_negative_double) {
        INA_TEST_ASSERT_SUCCEED(ina_iscp_register(data->iscp, 3, 3, NULL));
        INA_TEST_ASSERT_SUCCEED(ina_iscp_send(data->iscp, 1, 
            INA_ISCP_TYPE_DBL, -3.2));
    }

NOTE: Do not forget the semicolon after `INA_TEST_DATA()`

#### How to skip tests 

To skip existing test use the _SKIP version of `INA_TEST` or `INA_TEST_FIXTURE`. 

    INA_TEST_SKIP(my_suite, my_first_test_with_inac) {
        INA_ASSERT_FLOATING(1.0, 1.0);
    }

    INA_TEST_FIXTURE(iscp_tcp, send_negative_double) {


#### How to run the test suites

To run the tests simply call `ina_test_run()` by passing arguments count and 
arguments received from the command line.

    int main(int argc, char** argv) 
    { 
        ina_test_run(argc, argv);

From the command line prompt you can start all tests or a single suite

    ./test
    ./test test_suite
    
You can choose alternative result formats like `tap` or `JUnit` with the 
`format` options.

    ./test --format=tap
    ./test --format=junit

    
#### Helpers 
A more advanced feature of this test framework are provided by helper macros.
The framework supports in-situ helper and external helpers as well. Each helper
is started in a new process. Further it's possible chose to between 
wait/or spawn mode.

##### Adding in-situ Helpers
In-situ helpers are compiled directly in the test binary by using the
`INA_TEST_HELPER`macro. The macro takes two arguments: the suite name and
helper name. The `argc` and `argv` from the `main()` function are available in
the code body. For easy use and access use  Each Helper should assign a valid
RC to `retval` before leaving.

    INA_TEST_HELPER(tcp, dummy_dns_server) {
        const char* addr;
        int port;

        /* We need 2 arguments
        INA_TEST_HELPER_CHECK_ARGC(2);
        /* Get arguments */
        addr = INA_TEST_HELPER_CHAR_ARG(0);
        port = INA_TEST_HELPER_INTEGER_ARG(1);

        /* Starting coding your dummy tcp DNS server */
            ...

        INA_TEST_HELPER_SET_RC(EXIT_SUCCESS);
    }


##### Invoking in-situ Helpers	
Use `INA_TEST_HELPER_INVOKE` to start a child helper process.

    INA_TEST(tcp, dns_ping) {
        /* Invoke helper */
        ina_test_hid_t hid;
        INA_TEST_HELPER_INVOKE(&hid, tcp, dummy_dns_server, "127.0.0.1", 
                                "9001", NULL);

        /* Make some tests */
        INA_TEST_ASSERT_TRUE(dns_ping("120.0.0.1", 9001));

        /* Kill helper process */
        INA_TEST_HELPER_TERMINATE(hid);
    }

    INA_TEST(ullc, read_ring_buffer) {
        /* Invoke helper */
        ina_test_hid_t hid;
        INA_TEST_HELPER_INVOKE_WAIT(&hid&, tcp, create_ring_buffer, 5000, NULL);

        /* Make some tests */
        INA_TEST_ASSERT_TRUE(read_ring_buffer());
    }

To test or start an in-situ helper from the command line juste type

    ./test -h suite_name helper_name

  
##### External Helpers

    INA_TEST(tcp, dns_ping) {
        /* Invoke helper */
        ina_test_hid_t hid;
        INA_TEST_HELPER_CMD(&hid, "c:/test/dns.exe" "127.0.0.1", "9001", NULL);

        /* Make some tests */
        INA_TEST_ASSERT_TRUE(dns_ping("120.0.0.1", 9001));

        /* Kill helper process */
        INA_TEST_HELPER_TERMINATE(hid);
    }
  
    INA_TEST(tcp, dns_ping) {
        /* Invoke helper */
        ina_test_hid_t hid;
        INA_TEST_HELPER_CMD_WAIT("c:/test/dns.exe", 5000, "127.0.0.1", 
                                    "9001", NULL);

        /* Make some tests */
        INA_TEST_ASSERT_TRUE(dns_ping("120.0.0.1", 9001));  
    }
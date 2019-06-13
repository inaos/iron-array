## Configuration file

INAC provides a configuration file parser which works for C and Lua as well.

### Overview

The configuration file is a pure Lua script and consists of sections. Those 
section can be named or unnamed and they contains one more key/value pairs.
Sections and keys can be marked as required. Values for key can be string or
number type.  

    -- Unnamed section
    debug {
        command-latency=1000
    }
    -- Named section with key lo1
    iface "lo1" { 
        ip="127.0.0.2", 
        mask="255.0.0.0" 
    }
    -- Named section with key lo0
    iface "lo0" { 
        ip="127.0.0.1", 
        mask="255.0.0.0" 
    }

Configuration definition 

    sections = {}
    sections.debug = {
        name = "debug",
        named = false,
        required = true,
        keys = {
            command-latency = {
                required = true,
                typename = "number"
            },
            other_latency = {
                required = false,
                typename = "number"
            }
        },
        configured = false
    }

    sections.iface = {
        name = "iface",
        named = true,
        required = true,
        keys = {
            ip = {
                required = true,
                typename = "string"
            },
            mask = {
                required = true,
                typename = "string"
            },
        },
        configured = false
    }

### Usage
   
For basic usage use the appropriates macros. Start by declaring a variable to
hold the instance for the configuration file.
   
    ina_conffile_t *cf = NULL;
    
Declare 

    INA_CONFFILE(cf, NULL,
        INA_CONFFILE_SECTION("debug", INA_YES, NULL,
            INA_CONFFILE_NUMBER_KEY("command-latency", INA_YES)),
        INA_CONFFILE_NAMED_SECTION("iface", INA_NO, NULL,
            INA_CONFFILE_STRING_KEY("ip", INA_YES),
            INA_CONFFILE_NUMBER_KEY("mask", INA_NO)));

Create a configuration file instance by calling `ina_conffile_new()`.

    ina_conffile_t *cf = NULL;
   
    if (INA_SUCCEED(ina_conffile_new(&cf, NULL)) {

After calling you will get an new configurations file instance. You can 
optionally pass a file path as second argument to override the standard pattern of
configuration file location. By convention the configuration file path is 
[binary-name].conf in the current working directory if nothing else is 
specified.

Remember that each instance need to be destroyed with `ina_conffile_free()`. 

Define section and keys
   	
    ina_conffile_section_t *section = NULL;
    
    /* Add a unnamed section */
    ina_conffile_add_section(cf, &section, "debug", INA_YES);
    
    /* Add a key for a numeric required value to a section */
    ina_conffile_add_key(section, "command-latency", 
        INA_CONFFILE_VALUE_TYPE_NUMBER, INA_YES);
        
    /* Add a unamed section */
    ina_conffile_add_section(cf, &section, "iface", INA_YES);

Sample processor written in Lua

    -- sample processor
    for sk,s in pairs(sections) do
      if s.configured then
          print(sk)
          if not s.named then
              for k,v in pairs(s.keys) do
                  if v.has_value then
                      print(k,v.value)
                  end
               end
           else
               for nsk, ns in pairs(s.children) do
                   print("Named section: "..nsk)
                   for k,v in pairs(s.keys) do
                       if ns[k].has_value then
                           print(k,ns[k].value)
                       end
                    end
                end
            end
        end
    end
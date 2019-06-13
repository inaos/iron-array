## Tools
INAC provides a set of useful tools supporting software development.

### iDoc
iDoc lets you document code written in C programming language. Its comments are
designed to be very natural and readable so they're just as usable in the 
source code as they are in the generated documentation. No weird syntax or 
tags scattered everywhere. iDoc's generated Markdown documentation is simple, 
pretty and powerful.

#### Command line
When running iDOc, it searches for the project configuration `.idoc` in the
current working directory. If the configuration file is located in a different 
location use the `-c` command line option.
iDoc as two modes for generating source code documentation. The default is
to generate one file containing the complete source code documentation. 
    
    idoc -o /project/doc/api.md -c ./idoc

You can tell iDoc to generate a single file for each parsed source file using
the `-s` command line option and giving the output directory with the `-o` 
command line option.
 
    idoc -o /project/doc -s
    
You can run iDoc in a quit mode. This will show only error messages.

    idoc -o /project/doc -s -q
       
    
#### Configuration file
The .idoc is the main configuration file for your project.  It's where you 
tell iDoc which folders/files to scan and with folders/files to exclude.
When you run iDoc it will interpret any paths relative to the project 
configuration folder.  So if you keep your source code in 
`/home/project/include` and make your project configuration folder 
`/home/project`, the path have to be `include` in the project configuration.  

##### Documentation source  
To include files simply add lines prefixed by a `+`. You can use wildcard 
in order to include multiple files at once.
    
    +include/liba/a.h
    +src/b.h
    +include/liba/*.h

To exclude files add lines prefixed by a `-`. You can also use wildcard 
to exclude multiple files at once.
     
     -include/liba/version.h
     -include/liba/*.in.h
  
Exclusion and inclusion are processed in the order they are defined. Therefore ...

     -include/liba/test.h
     +include/liba/*.h
     
... will include `test.h`  because of the successive `*.h` inclusion rule.

On can include existing markdown file into the generated documentation

     +doc/manual.md
     +include/*.h
     +doc/examples/*.md
          
     
The example above will include `doc/manual.md`, parse all C headers in
`inlcude` and `examples` and also include all markdown files found in
`doc/examples`.

##### Project title
To set a global project title, define it with the `title` keyword in 
the project configuration.

    title: My great Library
    
This will add a headline at the beginning of your documentation. Note
that the title keyword is ignored in single files mode.
        
#### Comments
iDoc comes with only a few comment tags. Tags must end with a `:` or a line feed.

- `Parameters`: Starts a parameters block. Each parameter must starts
on a new line and parameter name and description must be
separated at least by two spaces. A parameters block end at the first
empty comment line.

      *
      * Parameters
      *   param1  Parameters 1
      *   param2  This is parameter 2 and described on
      *           2 lines 
      *
      * This is not part of the parameters block   

- `Return`: Start a "Return" Block. The tag can be used in block or
inline mode.
     
      /*
       * Inline tag
       * 
       * Return: Returns `0` if all went well otherwise an error code
       * is returned.
      
      /* 
       * Block style tag
       *
       * Return
       * - INA_SUCCESS id all went well
       * - INA_ERR_OUT_MEMORY if not enougth memory available
       * - INA_ERR_INVALID_ARGUMENT invalid argument passed
       *
     

     
- `Internal`: All lines afterwards this tag will not be
included in the documentation. The tag can be used in block or inline
mode.

      /*
       * Inline tag
       * This is a pulic documentation. 
       * 
       * Internal: This is a private documentation and
       * will not be included in your documentation
      
      /* 
       * Block style tage 
       * This is a public text
       *
       * Internal
       * This is private
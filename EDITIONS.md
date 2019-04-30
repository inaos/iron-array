## Handling commercial versus community edition

### Agreements

* Community License: GPLv2 (version tdb)
* Create FAQ for the community edition, clearifying common misunderstandings e.g.
  * GPLv2 copy-left why we do it
* Creating a Python script to generate the community edition from the source-tree (which is the commercial edition)
  * This should include an option to commit and push to the github repos directly
  * This will involve some sort of annotations in the source-code for the script to know what to exclude/include


### Feature set

| Feature                                  | Community     | Commercial    |
| ---------------------------------------- | ------------- | ------------- |
| Compressed multi-dimensional containers  | Yes           | Yes           |  
| Efficient expression evaluation          | Yes           | Yes           |
| Support for Intel VML                    | Yes           | Yes           |
| Persistence                              | **No**        | Yes           |
| Random distributions                     | **No**        | Yes           |
| Linear algebra                           | **No**        | Yes           |
| Parallel execution                       | **No**        | Yes           |
| Support                                  | **No**        | Yes           |

#include "minjuggutil.h"

#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/TargetLibraryInfo.h"

using namespace llvm;

extern "C" int jug_util_set_svml_vector_library()
{
    const char *argv[2];
    argv[0] = "opt";
    argv[1] = "-vector-library=SVML";

    llvm::cl::ParseCommandLineOptions(2, argv);

    return 0;
}

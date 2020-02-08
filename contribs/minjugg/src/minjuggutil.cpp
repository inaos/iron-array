#include "minjuggutil.h"

#include <llvm-c/Transforms/PassManagerBuilder.h>

#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"


using namespace llvm;

inline PassManagerBuilder *unwrap(LLVMPassManagerBuilderRef P) {
    return reinterpret_cast<PassManagerBuilder*>(P);
}

extern "C" int jug_util_set_svml_vector_library()
{
    const char *argv[2];
    argv[0] = "opt";
    argv[1] = "-vector-library=SVML";
    llvm::cl::ParseCommandLineOptions(2, argv);
    return 0;
}

extern "C" int jug_utils_enable_loop_vectorize(LLVMPassManagerBuilderRef PMB)
{
    llvm::PassManagerBuilder *pmb = unwrap(PMB);
    pmb->LoopVectorize = 1;
    return 0;
}

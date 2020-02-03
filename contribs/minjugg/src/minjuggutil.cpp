#include "minjuggutil.h"

#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/TargetLibraryInfo.h"

using namespace llvm;

extern "C" int jug_util_get_svml_vector_library(const char *triple, LLVMTargetLibraryInfoRef *tli)
{
    std::string striple(triple);
    Triple t(striple);

    TargetLibraryInfoImpl *TLII = new TargetLibraryInfoImpl(t);
    TLII->addVectorizableFunctionsFromVecLib(TLII->SVML);
    
    *tli = reinterpret_cast<LLVMTargetLibraryInfoRef>(TLII);

    return 0;
}

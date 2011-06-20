//===---- CGObjCCocotron.cpp - Emit LLVM Code from ASTs for a Module ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides Objective-C code generation targeting the Cocotron runtime.  
// The class in this file generates structures used by the Cocotron Objective-C
// runtime library.  These structures are defined in objc/objc.h and 
// objc/objc-api.h in the Cocotron runtime distribution.
//
//===----------------------------------------------------------------------===//

#include "CGObjCRuntime.h"
#include "CodeGenModule.h"
#include "CodeGenFunction.h"
#include "CGCleanup.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtObjC.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"

#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/LLVMContext.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetData.h"

#include <stdarg.h>


using namespace clang;
using namespace CodeGen;
using llvm::dyn_cast;


namespace {
    /// Class that lazily initialises the runtime function.  Avoids inserting the
    /// types and the function declaration into a module if they're not used, and
    /// avoids constructing the type more than once if it's used more than once.
    class LazyRuntimeFunction {
        CodeGenModule *CGM;
        std::vector<const llvm::Type*> ArgTys;
        const char *FunctionName;
        llvm::Function *Function;
    public:
        /// Constructor leaves this class uninitialized, because it is intended to
        /// be used as a field in another class and not all of the types that are
        /// used as arguments will necessarily be available at construction time.
        LazyRuntimeFunction() : CGM(0), FunctionName(0), Function(0) {}
        
        /// Initialises the lazy function with the name, return type, and the types
        /// of the arguments.
        END_WITH_NULL
        void init(CodeGenModule *Mod, const char *name,
                  const llvm::Type *RetTy, ...) {
            CGM =Mod;
            FunctionName = name;
            Function = 0;
            ArgTys.clear();
            va_list Args;
            va_start(Args, RetTy);
            while (const llvm::Type *ArgTy = va_arg(Args, const llvm::Type*))
                ArgTys.push_back(ArgTy);
            va_end(Args);
            // Push the return type on at the end so we can pop it off easily
            ArgTys.push_back(RetTy);
        }
        /// Overloaded cast operator, allows the class to be implicitly cast to an
        /// LLVM constant.
        operator llvm::Function*() {
            if (!Function) {
                if (0 == FunctionName) return 0;
                // We put the return type on the end of the vector, so pop it back off
                const llvm::Type *RetTy = ArgTys.back();
                ArgTys.pop_back();
                llvm::FunctionType *FTy = llvm::FunctionType::get(RetTy, ArgTys, false);
                Function =
                cast<llvm::Function>(CGM->CreateRuntimeFunction(FTy, FunctionName));
                // We won't need to use the types again, so we may as well clean up the
                // vector now
                ArgTys.resize(0);
            }
            return Function;
        }
    };
    
    
    /// GNU Objective-C runtime code generation.  This class implements the parts of
    /// Objective-C support that are specific to the GNU family of runtimes (GCC and
    /// GNUstep).
    class CGObjCCocotron : public CGObjCRuntime {
    protected:
        /// The module that is using this class
        CodeGenModule &CGM;
        /// The LLVM module into which output is inserted
        llvm::Module &TheModule;
        /// strut objc_super.  Used for sending messages to super.  This structure
        /// contains the receiver (object) and the expected class.
        const llvm::StructType *ObjCSuperTy;
        /// struct objc_super*.  The type of the argument to the superclass message
        /// lookup functions.  
        const llvm::PointerType *PtrToObjCSuperTy;
        /// LLVM type for selectors.  Opaque pointer (i8*) unless a header declaring
        /// SEL is included in a header somewhere, in which case it will be whatever
        /// type is declared in that header, most likely {i8*, i8*}.
        const llvm::PointerType *SelectorTy;
        /// LLVM i8 type.  Cached here to avoid repeatedly getting it in all of the
        /// places where it's used
        const llvm::IntegerType *Int8Ty;
        /// Pointer to i8 - LLVM type of char*, for all of the places where the
        /// runtime needs to deal with C strings.
        const llvm::PointerType *PtrToInt8Ty;
        
        /// CacheTy - LLVM type for struct objc_cache.
        const llvm::Type *CacheTy;
        /// CachePtrTy - LLVM type for struct objc_cache *.
        const llvm::Type *CachePtrTy;
        
        /// Instance Method Pointer type.  This is a pointer to a function that takes,
        /// at a minimum, an object and a selector, and is the generic type for
        /// Objective-C methods.  Due to differences between variadic / non-variadic
        /// calling conventions, it must always be cast to the correct type before
        /// actually being used.
        const llvm::PointerType *IMPTy;
        /// Type of an untyped Objective-C object.  Clang treats id as a built-in type
        /// when compiling Objective-C code, so this may be an opaque pointer (i8*),
        /// but if the runtime header declaring it is included then it may be a
        /// pointer to a structure.
        const llvm::PointerType *IdTy;
        /// Pointer to a pointer to an Objective-C object.  Used in the new ABI
        /// message lookup function and some GC-related functions.
        const llvm::PointerType *PtrToIdTy;
        /// The clang type of id.  Used when using the clang CGCall infrastructure to
        /// call Objective-C methods.
        CanQualType ASTIdTy;
        /// LLVM type for C int type.
        const llvm::IntegerType *IntTy;
        /// LLVM type for an opaque pointer.  This is identical to PtrToInt8Ty, but is
        /// used in the code to document the difference between i8* meaning a pointer
        /// to a C string and i8* meaning a pointer to some opaque type.
        const llvm::PointerType *PtrTy;
        /// LLVM type for C long type.  The runtime uses this in a lot of places where
        /// it should be using intptr_t, but we can't fix this without breaking
        /// compatibility with GCC...
        const llvm::IntegerType *LongTy;
        /// LLVM type for C size_t.  Used in various runtime data structures.
        const llvm::IntegerType *SizeTy;
        /// LLVM type for C ptrdiff_t.  Mainly used in property accessor functions.
        const llvm::IntegerType *PtrDiffTy;
        /// LLVM type for C int*.  Used for GCC-ABI-compatible non-fragile instance
        /// variables.
        const llvm::PointerType *PtrToIntTy;
        /// LLVM type for Objective-C BOOL type.
        const llvm::Type *BoolTy;
        /// Metadata kind used to tie method lookups to message sends.  The GNUstep
        /// runtime provides some LLVM passes that can use this to do things like
        /// automatic IMP caching and speculative inlining.
        unsigned msgSendMDKind;
        /// Helper function that generates a constant string and returns a pointer to
        /// the start of the string.  The result of this function can be used anywhere
        /// where the C code specifies const char*.  
        llvm::Constant *MakeConstantString(const std::string &Str,
                                           const std::string &Name="") {
            llvm::Constant *ConstStr = CGM.GetAddrOfConstantCString(Str, Name.c_str());
            return llvm::ConstantExpr::getGetElementPtr(ConstStr, Zeros, 2);
        }
        /// Emits a linkonce_odr string, whose name is the prefix followed by the
        /// string value.  This allows the linker to combine the strings between
        /// different modules.  Used for EH typeinfo names, selector strings, and a
        /// few other things.
        llvm::Constant *ExportUniqueString(const std::string &Str,
                                           const std::string prefix) {
            std::string name = prefix + Str;
            llvm::Constant *ConstStr = TheModule.getGlobalVariable(name);
            if (!ConstStr) {
                llvm::Constant *value = llvm::ConstantArray::get(VMContext, Str, true);
                ConstStr = new llvm::GlobalVariable(TheModule, value->getType(), true,
                                                    llvm::GlobalValue::LinkOnceODRLinkage, value, prefix + Str);
            }
            return llvm::ConstantExpr::getGetElementPtr(ConstStr, Zeros, 2);
        }
        /// Generates a global structure, initialized by the elements in the vector.
        /// The element types must match the types of the structure elements in the
        /// first argument.
        llvm::GlobalVariable *MakeGlobal(const llvm::StructType *Ty,
                                         std::vector<llvm::Constant*> &V,
                                         llvm::StringRef Name="",
                                         llvm::GlobalValue::LinkageTypes linkage
                                         =llvm::GlobalValue::InternalLinkage) {
            llvm::Constant *C = llvm::ConstantStruct::get(Ty, V);
            return new llvm::GlobalVariable(TheModule, Ty, false,
                                            linkage, C, Name);
        }
        /// Generates a global array.  The vector must contain the same number of
        /// elements that the array type declares, of the type specified as the array
        /// element type.
        llvm::GlobalVariable *MakeGlobal(const llvm::ArrayType *Ty,
                                         std::vector<llvm::Constant*> &V,
                                         llvm::StringRef Name="",
                                         llvm::GlobalValue::LinkageTypes linkage
                                         =llvm::GlobalValue::InternalLinkage) {
            llvm::Constant *C = llvm::ConstantArray::get(Ty, V);
            return new llvm::GlobalVariable(TheModule, Ty, false,
                                            linkage, C, Name);
        }
        /// Generates a global array, inferring the array type from the specified
        /// element type and the size of the initialiser.  
        llvm::GlobalVariable *MakeGlobalArray(const llvm::Type *Ty,
                                              std::vector<llvm::Constant*> &V,
                                              llvm::StringRef Name="",
                                              llvm::GlobalValue::LinkageTypes linkage
                                              =llvm::GlobalValue::InternalLinkage) {
            llvm::ArrayType *ArrayTy = llvm::ArrayType::get(Ty, V.size());
            return MakeGlobal(ArrayTy, V, Name, linkage);
        }
        /// Ensures that the value has the required type, by inserting a bitcast if
        /// required.  This function lets us avoid inserting bitcasts that are
        /// redundant.
        llvm::Value* EnforceType(CGBuilderTy B, llvm::Value *V, const llvm::Type *Ty){
            if (V->getType() == Ty) return V;
            return B.CreateBitCast(V, Ty);
        }
        // Some zeros used for GEPs in lots of places.
        llvm::Constant *Zeros[2];
        /// Null pointer value.  Mainly used as a terminator in various arrays.
        llvm::Constant *NULLPtr;
        /// LLVM context.
        llvm::LLVMContext &VMContext;
    private:
        /// Placeholder for the class.  Lots of things refer to the class before we've
        /// actually emitted it.  We use this alias as a placeholder, and then replace
        /// it with a pointer to the class structure before finally emitting the
        /// module.
        llvm::GlobalAlias *ClassPtrAlias;
        /// Placeholder for the metaclass.  Lots of things refer to the class before
        /// we've / actually emitted it.  We use this alias as a placeholder, and then
        /// replace / it with a pointer to the metaclass structure before finally
        /// emitting the / module.
        llvm::GlobalAlias *MetaClassPtrAlias;
        /// All of the classes that have been generated for this compilation units.
        std::vector<llvm::Constant*> Classes;
        /// All of the categories that have been generated for this compilation units.
        std::vector<llvm::Constant*> Categories;
        /// All of the Objective-C constant strings that have been generated for this
        /// compilation units.
        std::vector<llvm::Constant*> ConstantStrings;
        /// Map from string values to Objective-C constant strings in the output.
        /// Used to prevent emitting Objective-C strings more than once.  This should
        /// not be required at all - CodeGenModule should manage this list.
        llvm::StringMap<llvm::Constant*> ObjCStrings;
        /// All of the protocols that have been declared.
        llvm::StringMap<llvm::Constant*> ExistingProtocols;
        /// Type of the selector map.  This is roughly equivalent to the structure
        /// used in the GNUstep runtime, which maintains a list of all of the valid
        /// types for a selector in a table.
        typedef llvm::DenseMap<Selector, llvm::GlobalAlias*> SelectorMap;
        /// A map from selectors to selector types.  This allows us to emit all
        /// selectors of the same name and type together.
        SelectorMap SelectorTable;
        
        /// Selectors related to memory management.  When compiling in GC mode, we
        /// omit these.
        Selector RetainSel, ReleaseSel, AutoreleaseSel;
        /// Runtime functions used for memory management in GC mode.  Note that clang
        /// supports code generation for calling these functions, but neither GNU
        /// runtime actually supports this API properly yet.
        LazyRuntimeFunction IvarAssignFn, StrongCastAssignFn, MemMoveFn, WeakReadFn, 
        WeakAssignFn, GlobalAssignFn;
        
    protected:
        /// Function used for throwing Objective-C exceptions.
        LazyRuntimeFunction ExceptionThrowFn;
        /// Function used for rethrowing exceptions, used at the end of @finally or
        /// @synchronize blocks.
        LazyRuntimeFunction ExceptionReThrowFn;
        /// Function called when entering a catch function.  This is required for
        /// differentiating Objective-C exceptions and foreign exceptions.
        LazyRuntimeFunction EnterCatchFn;
        /// Function called when exiting from a catch block.  Used to do exception
        /// cleanup.
        LazyRuntimeFunction ExitCatchFn;
        /// Function called when entering an @synchronize block.  Acquires the lock.
        LazyRuntimeFunction SyncEnterFn;
        /// Function called when exiting an @synchronize block.  Releases the lock.
        LazyRuntimeFunction SyncExitFn;
        
    private:
        
        /// Function called if fast enumeration detects that the collection is
        /// modified during the update.
        LazyRuntimeFunction EnumerationMutationFn;
        /// Function for implementing synthesized property getters that return an
        /// object.
        LazyRuntimeFunction GetPropertyFn;
        /// Function for implementing synthesized property setters that return an
        /// object.
        LazyRuntimeFunction SetPropertyFn;
        /// Function used for non-object declared property getters.
        LazyRuntimeFunction GetStructPropertyFn;
        /// Function used for non-object declared property setters.
        LazyRuntimeFunction SetStructPropertyFn;
        
        /// The GCC ABI message lookup function.  Returns an IMP pointing to the
        /// method implementation for this message.
        LazyRuntimeFunction MsgLookupFn;
        /// The GCC ABI superclass message lookup function.  Takes a pointer to a
        /// structure describing the receiver and the class, and a selector as
        /// arguments.  Returns the IMP for the corresponding method.
        LazyRuntimeFunction MsgLookupSuperFn;
        
        /// The version of the runtime that this class targets.  Must match the
        /// version in the runtime.
        const int RuntimeVersion;
        /// The version of the protocol class.  Used to differentiate between ObjC1
        /// and ObjC2 protocols.  Objective-C 1 protocols can not contain optional
        /// components and can not contain declared properties.  We always emit
        /// Objective-C 2 property structures, but we have to pretend that they're
        /// Objective-C 1 property structures when targeting the GCC runtime or it
        /// will abort.
        const int ProtocolVersion;
    private:
        /// Generates an instance variable list structure.  This is a structure
        /// containing a size and an array of structures containing instance variable
        /// metadata.  This is used purely for introspection in the fragile ABI.  In
        /// the non-fragile ABI, it's used for instance variable fixup.
        llvm::Constant *GenerateIvarList(
                                         const llvm::SmallVectorImpl<llvm::Constant *>  &IvarNames,
                                         const llvm::SmallVectorImpl<llvm::Constant *>  &IvarTypes,
                                         const llvm::SmallVectorImpl<llvm::Constant *>  &IvarOffsets);
        /// Generates a method list structure.  This is a structure containing a size
        /// and an array of structures containing method metadata.
        ///
        /// This structure is used by both classes and categories, and contains a next
        /// pointer allowing them to be chained together in a linked list.
        llvm::Constant *GenerateMethodList(const llvm::StringRef &ClassName,
                                           const llvm::StringRef &CategoryName,
                                           const llvm::SmallVectorImpl<Selector>  &MethodSels,
                                           const llvm::SmallVectorImpl<llvm::Constant *>  &MethodTypes,
                                           bool isClassMethodList);
        /// Emits an empty protocol.  This is used for @protocol() where no protocol
        /// is found.  The runtime will (hopefully) fix up the pointer to refer to the
        /// real protocol.
        llvm::Constant *GenerateEmptyProtocol(const std::string &ProtocolName);
        /// Generates a list of property metadata structures.  This follows the same
        /// pattern as method and instance variable metadata lists.
        llvm::Constant *GeneratePropertyList(const ObjCImplementationDecl *OID,
                                             llvm::SmallVectorImpl<Selector> &InstanceMethodSels,
                                             llvm::SmallVectorImpl<llvm::Constant*> &InstanceMethodTypes);
        /// Generates a list of referenced protocols.  Classes, categories, and
        /// protocols all use this structure.
        llvm::Constant *GenerateProtocolList(
                                             const llvm::SmallVectorImpl<std::string> &Protocols);
        /// To ensure that all protocols are seen by the runtime, we add a category on
        /// a class defined in the runtime, declaring no methods, but adopting the
        /// protocols.  This is a horribly ugly hack, but it allows us to collect all
        /// of the protocols without changing the ABI.
        void GenerateProtocolHolderCategory(void);
        /// Generates a class structure.
        llvm::Constant *GenerateClassStructure(
                                               llvm::Constant *MetaClass,
                                               llvm::Constant *SuperClass,
                                               unsigned info,
                                               const char *Name,
                                               llvm::Constant *Version,
                                               llvm::Constant *InstanceSize,
                                               llvm::Constant *IVars,
                                               llvm::Constant *Methods,
                                               llvm::Constant *Protocols,
                                               llvm::Constant *IvarOffsets,
                                               llvm::Constant *Properties,
                                               bool isMeta=false);
        /// Generates a method list.  This is used by protocols to define the required
        /// and optional methods.
        llvm::Constant *GenerateProtocolMethodList(
                                                   const llvm::SmallVectorImpl<llvm::Constant *>  &MethodNames,
                                                   const llvm::SmallVectorImpl<llvm::Constant *>  &MethodTypes);
        /// Returns a selector with the specified type encoding.  An empty string is
        /// used to return an untyped selector (with the types field set to NULL).
        llvm::Value *GetSelector(CGBuilderTy &Builder, Selector Sel,
                                 const std::string &TypeEncoding, bool lval);
        /// Returns the variable used to store the offset of an instance variable.
        llvm::GlobalVariable *ObjCIvarOffsetVariable(const ObjCInterfaceDecl *ID,
                                                     const ObjCIvarDecl *Ivar);
        /// Emits a reference to a class.  This allows the linker to object if there
        /// is no class of the matching name.
        void EmitClassRef(const std::string &className);
    protected:
        /// Looks up the method for sending a message to the specified object.
        virtual llvm::Value *LookupIMP(CodeGenFunction &CGF,
                                       llvm::Value *&Receiver,
                                       llvm::Value *cmd,
                                       llvm::MDNode *node) {
            CGBuilderTy &Builder = CGF.Builder;
            llvm::Value *imp = Builder.CreateCall2(MsgLookupFn, 
                                                   EnforceType(Builder, Receiver, IdTy),
                                                   EnforceType(Builder, cmd, SelectorTy));
            cast<llvm::CallInst>(imp)->setMetadata(msgSendMDKind, node);
            return imp;
        }
        /// Looks up the method for sending a message to a superclass.
        virtual llvm::Value *LookupIMPSuper(CodeGenFunction &CGF,
                                            llvm::Value *ObjCSuper,
                                            llvm::Value *cmd) {
            CGBuilderTy &Builder = CGF.Builder;
            llvm::Value *lookupArgs[] = {EnforceType(Builder, ObjCSuper,
                                                     PtrToObjCSuperTy), cmd};
            return Builder.CreateCall(MsgLookupSuperFn, lookupArgs, lookupArgs+2);
        }    public:
        CGObjCCocotron(CodeGenModule &cgm);
        
        virtual llvm::Constant *GenerateConstantString(const StringLiteral *);
        
        virtual RValue
        GenerateMessageSend(CodeGenFunction &CGF,
                            ReturnValueSlot Return,
                            QualType ResultType,
                            Selector Sel,
                            llvm::Value *Receiver,
                            const CallArgList &CallArgs,
                            const ObjCInterfaceDecl *Class,
                            const ObjCMethodDecl *Method);
        virtual RValue
        GenerateMessageSendSuper(CodeGenFunction &CGF,
                                 ReturnValueSlot Return,
                                 QualType ResultType,
                                 Selector Sel,
                                 const ObjCInterfaceDecl *Class,
                                 bool isCategoryImpl,
                                 llvm::Value *Receiver,
                                 bool IsClassMessage,
                                 const CallArgList &CallArgs,
                                 const ObjCMethodDecl *Method);
        virtual llvm::Value *GetClass(CGBuilderTy &Builder,
                                      const ObjCInterfaceDecl *OID);
        virtual llvm::Value *GetSelector(CGBuilderTy &Builder, Selector Sel,
                                         bool lval = false);
        virtual llvm::Value *GetSelector(CGBuilderTy &Builder, const ObjCMethodDecl
                                         *Method);
        virtual llvm::Constant *GetEHType(QualType T);
        
        virtual llvm::Function *GenerateMethod(const ObjCMethodDecl *OMD,
                                               const ObjCContainerDecl *CD);
        virtual void GenerateCategory(const ObjCCategoryImplDecl *CMD);
        virtual void GenerateClass(const ObjCImplementationDecl *ClassDecl);
        virtual llvm::Value *GenerateProtocolRef(CGBuilderTy &Builder,
                                                 const ObjCProtocolDecl *PD);
        virtual void GenerateProtocol(const ObjCProtocolDecl *PD);
        virtual llvm::Function *ModuleInitFunction();
        virtual llvm::Function *GetPropertyGetFunction();
        virtual llvm::Function *GetPropertySetFunction();
        virtual llvm::Function *GetSetStructFunction();
        virtual llvm::Function *GetGetStructFunction();
        virtual llvm::Constant *EnumerationMutationFunction();
        
        virtual void EmitTryStmt(CodeGenFunction &CGF,
                                 const ObjCAtTryStmt &S);
        virtual void EmitSynchronizedStmt(CodeGenFunction &CGF,
                                          const ObjCAtSynchronizedStmt &S);
        virtual void EmitThrowStmt(CodeGenFunction &CGF,
                                   const ObjCAtThrowStmt &S);
        virtual llvm::Value * EmitObjCWeakRead(CodeGenFunction &CGF,
                                               llvm::Value *AddrWeakObj);
        virtual void EmitObjCWeakAssign(CodeGenFunction &CGF,
                                        llvm::Value *src, llvm::Value *dst);
        virtual void EmitObjCGlobalAssign(CodeGenFunction &CGF,
                                          llvm::Value *src, llvm::Value *dest,
                                          bool threadlocal=false);
        virtual void EmitObjCIvarAssign(CodeGenFunction &CGF,
                                        llvm::Value *src, llvm::Value *dest,
                                        llvm::Value *ivarOffset);
        virtual void EmitObjCStrongCastAssign(CodeGenFunction &CGF,
                                              llvm::Value *src, llvm::Value *dest);
        virtual void EmitGCMemmoveCollectable(CodeGenFunction &CGF,
                                              llvm::Value *DestPtr,
                                              llvm::Value *SrcPtr,
                                              llvm::Value *Size);
        virtual LValue EmitObjCValueForIvar(CodeGenFunction &CGF,
                                            QualType ObjectTy,
                                            llvm::Value *BaseValue,
                                            const ObjCIvarDecl *Ivar,
                                            unsigned CVRQualifiers);
        virtual llvm::Value *EmitIvarOffset(CodeGenFunction &CGF,
                                            const ObjCInterfaceDecl *Interface,
                                            const ObjCIvarDecl *Ivar);
        virtual llvm::Constant *BuildGCBlockLayout(CodeGenModule &CGM,
                                                   const CGBlockInfo &blockInfo) {
            return NULLPtr;
        }
        
        virtual llvm::GlobalVariable *GetClassGlobal(const std::string &Name) {
            return 0;
        }
    };
} // end anonymous namespace


/// Emits a reference to a dummy variable which is emitted with each class.
/// This ensures that a linker error will be generated when trying to link
/// together modules where a referenced class is not defined.
void CGObjCCocotron::EmitClassRef(const std::string &className) {
    std::string symbolRef = "__objc_class_ref_" + className;
    // Don't emit two copies of the same symbol
    if (TheModule.getGlobalVariable(symbolRef))
        return;
    std::string symbolName = "__objc_class_name_" + className;
    llvm::GlobalVariable *ClassSymbol = TheModule.getGlobalVariable(symbolName);
    if (!ClassSymbol) {
        ClassSymbol = new llvm::GlobalVariable(TheModule, LongTy, false,
                                               llvm::GlobalValue::ExternalLinkage, 0, symbolName);
    }
    new llvm::GlobalVariable(TheModule, ClassSymbol->getType(), true,
                             llvm::GlobalValue::WeakAnyLinkage, ClassSymbol, symbolRef);
}

static std::string SymbolNameForMethod(const llvm::StringRef &ClassName,
                                       const llvm::StringRef &CategoryName, const Selector MethodName,
                                       bool isClassMethod) {
    std::string MethodNameColonStripped = MethodName.getAsString();
    std::replace(MethodNameColonStripped.begin(), MethodNameColonStripped.end(),
                 ':', '_');
    return (llvm::Twine(isClassMethod ? "_c_" : "_i_") + ClassName + "_" +
            CategoryName + "_" + MethodNameColonStripped).str();
}

CGObjCCocotron::CGObjCCocotron(CodeGenModule &cgm)
: CGM(cgm), TheModule(CGM.getModule()), VMContext(cgm.getLLVMContext()),
ClassPtrAlias(0), MetaClassPtrAlias(0), RuntimeVersion(0),
ProtocolVersion(0) {
    
    
    msgSendMDKind = VMContext.getMDKindID("GNUObjCMessageSend");
    
    CodeGenTypes &Types = CGM.getTypes();
    IntTy = cast<llvm::IntegerType>(
                                    Types.ConvertType(CGM.getContext().IntTy));
    LongTy = cast<llvm::IntegerType>(
                                     Types.ConvertType(CGM.getContext().LongTy));
    SizeTy = cast<llvm::IntegerType>(
                                     Types.ConvertType(CGM.getContext().getSizeType()));
    PtrDiffTy = cast<llvm::IntegerType>(
                                        Types.ConvertType(CGM.getContext().getPointerDiffType()));
    BoolTy = CGM.getTypes().ConvertType(CGM.getContext().BoolTy);
    
    Int8Ty = llvm::Type::getInt8Ty(VMContext);
    // C string type.  Used in lots of places.
    PtrToInt8Ty = llvm::PointerType::getUnqual(Int8Ty);
    
    CacheTy = llvm::OpaqueType::get(VMContext);
    CGM.getModule().addTypeName("struct._objc_cache", CacheTy);
    CachePtrTy = llvm::PointerType::getUnqual(CacheTy);
    
    Zeros[0] = llvm::ConstantInt::get(LongTy, 0);
    Zeros[1] = Zeros[0];
    NULLPtr = llvm::ConstantPointerNull::get(PtrToInt8Ty);
    // Get the selector Type.
    QualType selTy = CGM.getContext().getObjCSelType();
    SelectorTy = PtrToInt8Ty;
    
    PtrToIntTy = llvm::PointerType::getUnqual(IntTy);
    PtrTy = PtrToInt8Ty;
    
    // Object type
    QualType UnqualIdTy = CGM.getContext().getObjCIdType();
    ASTIdTy = CanQualType();
    if (UnqualIdTy != QualType()) {
        ASTIdTy = CGM.getContext().getCanonicalType(UnqualIdTy);
        IdTy = cast<llvm::PointerType>(CGM.getTypes().ConvertType(ASTIdTy));
    } else {
        IdTy = PtrToInt8Ty;
    }
    PtrToIdTy = llvm::PointerType::getUnqual(IdTy);
    
    ObjCSuperTy = llvm::StructType::get(IdTy, IdTy, NULL);
    PtrToObjCSuperTy = llvm::PointerType::getUnqual(ObjCSuperTy);
    
    const llvm::Type *VoidTy = llvm::Type::getVoidTy(VMContext);
    
    // void objc_exception_throw(id);
    ExceptionThrowFn.init(&CGM, "objc_exception_throw", VoidTy, IdTy, NULL);
    ExceptionReThrowFn.init(&CGM, "objc_exception_throw", VoidTy, IdTy, NULL);
    // int objc_sync_enter(id);
    SyncEnterFn.init(&CGM, "objc_sync_enter", IntTy, IdTy, NULL);
    // int objc_sync_exit(id);
    SyncExitFn.init(&CGM, "objc_sync_exit", IntTy, IdTy, NULL);
    
    // void objc_enumerationMutation (id)
    EnumerationMutationFn.init(&CGM, "objc_enumerationMutation", VoidTy,
                               IdTy, NULL);
    
    // id objc_getProperty(id, SEL, ptrdiff_t, BOOL)
    GetPropertyFn.init(&CGM, "objc_getProperty", IdTy, IdTy, SelectorTy,
                       PtrDiffTy, BoolTy, NULL);
    // void objc_setProperty(id, SEL, ptrdiff_t, id, BOOL, BOOL)
    SetPropertyFn.init(&CGM, "objc_setProperty", VoidTy, IdTy, SelectorTy,
                       PtrDiffTy, IdTy, BoolTy, BoolTy, NULL);
    // void objc_setPropertyStruct(void*, void*, ptrdiff_t, BOOL, BOOL)
    GetStructPropertyFn.init(&CGM, "objc_getPropertyStruct", VoidTy, PtrTy, PtrTy, 
                             PtrDiffTy, BoolTy, BoolTy, NULL);
    // void objc_setPropertyStruct(void*, void*, ptrdiff_t, BOOL, BOOL)
    SetStructPropertyFn.init(&CGM, "objc_setPropertyStruct", VoidTy, PtrTy, PtrTy, 
                             PtrDiffTy, BoolTy, BoolTy, NULL);
    
    // IMP type
    std::vector<const llvm::Type*> IMPArgs;
    IMPArgs.push_back(IdTy);
    IMPArgs.push_back(SelectorTy);
    IMPTy = llvm::PointerType::getUnqual(llvm::FunctionType::get(IdTy, IMPArgs,
                                                                 true));
    
    // Don't bother initialising the GC stuff unless we're compiling in GC mode
    if (CGM.getLangOptions().getGCMode() != LangOptions::NonGC) {
        // Get selectors needed in GC mode
        RetainSel = GetNullarySelector("retain", CGM.getContext());
        ReleaseSel = GetNullarySelector("release", CGM.getContext());
        AutoreleaseSel = GetNullarySelector("autorelease", CGM.getContext());
        
        // Get functions needed in GC mode
        
        // id objc_assign_ivar(id, id, ptrdiff_t);
        IvarAssignFn.init(&CGM, "objc_assign_ivar", IdTy, IdTy, IdTy, PtrDiffTy,
                          NULL);
        // id objc_assign_strongCast (id, id*)
        StrongCastAssignFn.init(&CGM, "objc_assign_strongCast", IdTy, IdTy,
                                PtrToIdTy, NULL);
        // id objc_assign_global(id, id*);
        GlobalAssignFn.init(&CGM, "objc_assign_global", IdTy, IdTy, PtrToIdTy,
                            NULL);
        // id objc_assign_weak(id, id*);
        WeakAssignFn.init(&CGM, "objc_assign_weak", IdTy, IdTy, PtrToIdTy, NULL);
        // id objc_read_weak(id*);
        WeakReadFn.init(&CGM, "objc_read_weak", IdTy, PtrToIdTy, NULL);
        // void *objc_memmove_collectable(void*, void *, size_t);
        MemMoveFn.init(&CGM, "objc_memmove_collectable", PtrTy, PtrTy, PtrTy,
                       SizeTy, NULL);
    }
    
    MsgLookupFn.init(&CGM, "objc_msg_lookup", IMPTy, IdTy, SelectorTy, NULL);
    // IMP objc_msg_lookup_super(struct objc_super*, SEL);
    MsgLookupSuperFn.init(&CGM, "objc_msg_lookup_super", IMPTy,
                          PtrToObjCSuperTy, SelectorTy, NULL);
}

// This has to perform the lookup every time, since posing and related
// techniques can modify the name -> class mapping.
llvm::Value *CGObjCCocotron::GetClass(CGBuilderTy &Builder,
                                 const ObjCInterfaceDecl *OID) {
    llvm::Value *ClassName = CGM.GetAddrOfConstantCString(OID->getNameAsString());
    // With the incompatible ABI, this will need to be replaced with a direct
    // reference to the class symbol.  For the compatible nonfragile ABI we are
    // still performing this lookup at run time but emitting the symbol for the
    // class externally so that we can make the switch later.
    EmitClassRef(OID->getNameAsString());
    ClassName = Builder.CreateStructGEP(ClassName, 0);
    
    std::vector<const llvm::Type*> Params(1, PtrToInt8Ty);
    llvm::Constant *ClassLookupFn =
    CGM.CreateRuntimeFunction(llvm::FunctionType::get(IdTy,
                                                      Params,
                                                      true),
                                                      "objc_lookUpClass");
    return Builder.CreateCall(ClassLookupFn, ClassName);
}

llvm::Value *CGObjCCocotron::GetSelector(CGBuilderTy &Builder, Selector Sel,
                                    const std::string &TypeEncoding, bool lval) {
        llvm::GlobalAlias *&SelValue =  SelectorTable[Sel];

        if (0 == SelValue) {
            SelValue = new llvm::GlobalAlias(PtrToInt8Ty,
                                             llvm::GlobalValue::PrivateLinkage,
                                             ".objc_selector_alias_"+Sel.getAsString(), NULL,
                                             &TheModule);
        }
    
        //TODO FIX IT
    
    /*
       if (lval) {
            llvm::Value *tmp = Builder.CreateAlloca(SelValue->getType());
            Builder.CreateStore(SelValue, tmp);
            return tmp;
        }

        return SelValue;
    */
    
    
   /* llvm::Constant *Idxs[] = {Zeros[0],
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), 0), Zeros[0]};
    
    // FIXME: We're generating redundant loads and stores here!
    llvm::Constant *SelPtr = llvm::ConstantExpr::getGetElementPtr(SelectorList,
                                                                  Idxs, 2);*/
       
        return  CGM.GetAddrOfConstantCString(Sel.getAsString());
        /*selectorString = Builder.CreateStructGEP(selectorString, 0);
        
        std::vector<const llvm::Type*> Params(1, PtrToInt8Ty);
        llvm::Constant *sel_getUidFn =
        CGM.CreateRuntimeFunction(llvm::FunctionType::get(IdTy,
                                                          Params,
                                                          true),
                                                          "sel_getUid");
        return Builder.CreateCall(sel_getUidFn, selectorString);*/

}

llvm::Value *CGObjCCocotron::GetSelector(CGBuilderTy &Builder, Selector Sel,
                                    bool lval) {
    return GetSelector(Builder, Sel, std::string(), lval);
}

llvm::Value *CGObjCCocotron::GetSelector(CGBuilderTy &Builder, const ObjCMethodDecl
                                    *Method) {
    std::string SelTypes;
    CGM.getContext().getObjCEncodingForMethodDecl(Method, SelTypes);
    return GetSelector(Builder, Method->getSelector(), SelTypes, false);
}

llvm::Constant *CGObjCCocotron::GetEHType(QualType T) {
    if (!CGM.getLangOptions().CPlusPlus) {
        if (T->isObjCIdType()
            || T->isObjCQualifiedIdType()) {
            // With the old ABI, there was only one kind of catchall, which broke
            // foreign exceptions.  With the new ABI, we use __objc_id_typeinfo as
            // a pointer indicating object catchalls, and NULL to indicate real
            // catchalls
            if (CGM.getLangOptions().ObjCNonFragileABI) {
                return MakeConstantString("@id");
            } else {
                return 0;
            }
        }
        
        // All other types should be Objective-C interface pointer types.
        const ObjCObjectPointerType *OPT =
        T->getAs<ObjCObjectPointerType>();
        assert(OPT && "Invalid @catch type.");
        const ObjCInterfaceDecl *IDecl =
        OPT->getObjectType()->getInterface();
        assert(IDecl && "Invalid @catch type.");
        return MakeConstantString(IDecl->getIdentifier()->getName());
    }
    // For Objective-C++, we want to provide the ability to catch both C++ and
    // Objective-C objects in the same function.
    
    // There's a particular fixed type info for 'id'.
    if (T->isObjCIdType() ||
        T->isObjCQualifiedIdType()) {
        llvm::Constant *IDEHType =
        CGM.getModule().getGlobalVariable("__objc_id_type_info");
        if (!IDEHType)
            IDEHType =
            new llvm::GlobalVariable(CGM.getModule(), PtrToInt8Ty,
                                     false,
                                     llvm::GlobalValue::ExternalLinkage,
                                     0, "__objc_id_type_info");
        return llvm::ConstantExpr::getBitCast(IDEHType, PtrToInt8Ty);
    }
    
    const ObjCObjectPointerType *PT =
    T->getAs<ObjCObjectPointerType>();
    assert(PT && "Invalid @catch type.");
    const ObjCInterfaceType *IT = PT->getInterfaceType();
    assert(IT && "Invalid @catch type.");
    std::string className = IT->getDecl()->getIdentifier()->getName();
    
    std::string typeinfoName = "__objc_eh_typeinfo_" + className;
    
    // Return the existing typeinfo if it exists
    llvm::Constant *typeinfo = TheModule.getGlobalVariable(typeinfoName);
    if (typeinfo) return typeinfo;
    
    // Otherwise create it.
    
    // vtable for gnustep::libobjc::__objc_class_type_info
    // It's quite ugly hard-coding this.  Ideally we'd generate it using the host
    // platform's name mangling.
    const char *vtableName = "_ZTVN7gnustep7libobjc22__objc_class_type_infoE";
    llvm::Constant *Vtable = TheModule.getGlobalVariable(vtableName);
    if (!Vtable) {
        Vtable = new llvm::GlobalVariable(TheModule, PtrToInt8Ty, true,
                                          llvm::GlobalValue::ExternalLinkage, 0, vtableName);
    }
    llvm::Constant *Two = llvm::ConstantInt::get(IntTy, 2);
    Vtable = llvm::ConstantExpr::getGetElementPtr(Vtable, &Two, 1);
    Vtable = llvm::ConstantExpr::getBitCast(Vtable, PtrToInt8Ty);
    
    llvm::Constant *typeName =
    ExportUniqueString(className, "__objc_eh_typename_");
    
    std::vector<llvm::Constant*> fields;
    fields.push_back(Vtable);
    fields.push_back(typeName);
    llvm::Constant *TI = 
    MakeGlobal(llvm::StructType::get(PtrToInt8Ty, PtrToInt8Ty,
                                     NULL), fields, "__objc_eh_typeinfo_" + className,
               llvm::GlobalValue::LinkOnceODRLinkage);
    return llvm::ConstantExpr::getBitCast(TI, PtrToInt8Ty);
}

/// Generate an NSConstantString object.
llvm::Constant *CGObjCCocotron::GenerateConstantString(const StringLiteral *SL) {
    std::string Str = SL->getString().str();

    // Look for an existing one
    llvm::StringMap<llvm::Constant*>::iterator old = ObjCStrings.find(Str);
    if (old != ObjCStrings.end())
        return old->getValue();
    
    llvm::Constant *Zero =
    llvm::Constant::getNullValue(llvm::Type::getInt32Ty(VMContext));
    llvm::Constant *Zeros[] = { Zero, Zero };
    
    const llvm::Type *Ty = CGM.getTypes().ConvertType(CGM.getContext().IntTy);
    Ty = llvm::ArrayType::get(Ty, 0);
    
    llvm::Constant *GV;
    GV = CGM.CreateRuntimeVariable(Ty, "_NSConstantStringClassReference");
    
    llvm::Constant *ConstantStringClassRef = llvm::ConstantExpr::getGetElementPtr(GV, Zeros, 2);    
    
    std::vector<llvm::Constant*> Ivars;
    //Ivars.push_back(llvm::ConstantExpr::getGetElementPtr(GV, Zeros, 2));
    Ivars.push_back(ConstantStringClassRef);
    Ivars.push_back(MakeConstantString(Str));
    Ivars.push_back(llvm::ConstantInt::get(IntTy, Str.size()));
    llvm::Constant *ObjCStr = MakeGlobal(
                                         llvm::StructType::get(PtrToIntTy, PtrToInt8Ty, IntTy, NULL),
                                         Ivars, ".objc_str");
    ObjCStr = llvm::ConstantExpr::getBitCast(ObjCStr, PtrToInt8Ty);
    ObjCStrings[Str] = ObjCStr;
    ConstantStrings.push_back(ObjCStr);
    return ObjCStr;
}

///Generates a message send where the super is the receiver.  This is a message
///send to self with special delivery semantics indicating which class's method
///should be called.
RValue
CGObjCCocotron::GenerateMessageSendSuper(CodeGenFunction &CGF,
                                    ReturnValueSlot Return,
                                    QualType ResultType,
                                    Selector Sel,
                                    const ObjCInterfaceDecl *Class,
                                    bool isCategoryImpl,
                                    llvm::Value *Receiver,
                                    bool IsClassMessage,
                                    const CallArgList &CallArgs,
                                    const ObjCMethodDecl *Method) {
    if (CGM.getLangOptions().getGCMode() != LangOptions::NonGC) {
        if (Sel == RetainSel || Sel == AutoreleaseSel) {
            return RValue::get(Receiver);
        }
        if (Sel == ReleaseSel) {
            return RValue::get(0);
        }
    }
    
    CGBuilderTy &Builder = CGF.Builder;
    llvm::Value *cmd = GetSelector(Builder, Sel);
    
    
    CallArgList ActualArgs;
    
    ActualArgs.add(RValue::get(EnforceType(Builder, Receiver, IdTy)), ASTIdTy);
    ActualArgs.add(RValue::get(cmd), CGF.getContext().getObjCSelType());
    ActualArgs.insert(ActualArgs.end(), CallArgs.begin(), CallArgs.end());
    
    CodeGenTypes &Types = CGM.getTypes();
    const CGFunctionInfo &FnInfo = Types.getFunctionInfo(ResultType, ActualArgs,
                                                         FunctionType::ExtInfo());
    
    llvm::Value *ReceiverClass = 0;
    if (isCategoryImpl) {
        llvm::Constant *classLookupFunction = 0;
        std::vector<const llvm::Type*> Params;
        Params.push_back(PtrTy);
        if (IsClassMessage)  {
            classLookupFunction = CGM.CreateRuntimeFunction(llvm::FunctionType::get(
                                                                                    IdTy, Params, true), "objc_get_meta_class");
        } else {
            classLookupFunction = CGM.CreateRuntimeFunction(llvm::FunctionType::get(
                                                                                    IdTy, Params, true), "objc_get_class");
        }
        ReceiverClass = Builder.CreateCall(classLookupFunction,
                                           MakeConstantString(Class->getNameAsString()));
    } else {
        // Set up global aliases for the metaclass or class pointer if they do not
        // already exist.  These will are forward-references which will be set to
        // pointers to the class and metaclass structure created for the runtime
        // load function.  To send a message to super, we look up the value of the
        // super_class pointer from either the class or metaclass structure.
        if (IsClassMessage)  {
            if (!MetaClassPtrAlias) {
                MetaClassPtrAlias = new llvm::GlobalAlias(IdTy,
                                                          llvm::GlobalValue::InternalLinkage, ".objc_metaclass_ref" +
                                                          Class->getNameAsString(), NULL, &TheModule);
            }
            ReceiverClass = MetaClassPtrAlias;
        } else {
            if (!ClassPtrAlias) {
                ClassPtrAlias = new llvm::GlobalAlias(IdTy,
                                                      llvm::GlobalValue::InternalLinkage, ".objc_class_ref" +
                                                      Class->getNameAsString(), NULL, &TheModule);
            }
            ReceiverClass = ClassPtrAlias;
        }
    }
    // Cast the pointer to a simplified version of the class structure
    ReceiverClass = Builder.CreateBitCast(ReceiverClass,
                                          llvm::PointerType::getUnqual(
                                                                       llvm::StructType::get(IdTy, IdTy, NULL)));
    // Get the superclass pointer
    ReceiverClass = Builder.CreateStructGEP(ReceiverClass, 1);
    // Load the superclass pointer
    ReceiverClass = Builder.CreateLoad(ReceiverClass);
    // Construct the structure used to look up the IMP
    llvm::StructType *ObjCSuperTy = llvm::StructType::get(Receiver->getType(), IdTy, NULL);
    llvm::Value *ObjCSuper = Builder.CreateAlloca(ObjCSuperTy);
    
    Builder.CreateStore(Receiver, Builder.CreateStructGEP(ObjCSuper, 0));
    Builder.CreateStore(ReceiverClass, Builder.CreateStructGEP(ObjCSuper, 1));
    
    ObjCSuper = EnforceType(Builder, ObjCSuper, PtrToObjCSuperTy);
    const llvm::FunctionType *impType =
    Types.GetFunctionType(FnInfo, Method ? Method->isVariadic() : false);
    
    // Get the IMP
    llvm::Value *imp = LookupIMPSuper(CGF, ObjCSuper, cmd);
    imp = EnforceType(Builder, imp, llvm::PointerType::getUnqual(impType));
    
    llvm::Value *impMD[] = {
        llvm::MDString::get(VMContext, Sel.getAsString()),
        llvm::MDString::get(VMContext, Class->getSuperClass()->getNameAsString()),
        llvm::ConstantInt::get(llvm::Type::getInt1Ty(VMContext), IsClassMessage)
    };
    llvm::MDNode *node = llvm::MDNode::get(VMContext, impMD);
    
    llvm::Instruction *call;
    RValue msgRet = CGF.EmitCall(FnInfo, imp, Return, ActualArgs,
                                 0, &call);
    call->setMetadata(msgSendMDKind, node);
    return msgRet;
}

/// Generate code for a message send expression.
RValue
CGObjCCocotron::GenerateMessageSend(CodeGenFunction &CGF,
                               ReturnValueSlot Return,
                               QualType ResultType,
                               Selector Sel,
                               llvm::Value *Receiver,
                               const CallArgList &CallArgs,
                               const ObjCInterfaceDecl *Class,
                               const ObjCMethodDecl *Method) {
    // Strip out message sends to retain / release in GC mode
    if (CGM.getLangOptions().getGCMode() != LangOptions::NonGC) {
        if (Sel == RetainSel || Sel == AutoreleaseSel) {
            return RValue::get(Receiver);
        }
        if (Sel == ReleaseSel) {
            return RValue::get(0);
        }
    }
    
    CGBuilderTy &Builder = CGF.Builder;
    
    // If the return type is something that goes in an integer register, the
    // runtime will handle 0 returns.  For other cases, we fill in the 0 value
    // ourselves.
    //
    // The language spec says the result of this kind of message send is
    // undefined, but lots of people seem to have forgotten to read that
    // paragraph and insist on sending messages to nil that have structure
    // returns.  With GCC, this generates a random return value (whatever happens
    // to be on the stack / in those registers at the time) on most platforms,
    // and generates an illegal instruction trap on SPARC.  With LLVM it corrupts
    // the stack.  
    bool isPointerSizedReturn = (ResultType->isAnyPointerType() ||
                                 ResultType->isIntegralOrEnumerationType() || ResultType->isVoidType());
    
    llvm::BasicBlock *startBB = 0;
    llvm::BasicBlock *messageBB = 0;
    llvm::BasicBlock *continueBB = 0;
    
    if (!isPointerSizedReturn) {
        startBB = Builder.GetInsertBlock();
        messageBB = CGF.createBasicBlock("msgSend");
        continueBB = CGF.createBasicBlock("continue");
        
        llvm::Value *isNil = Builder.CreateICmpEQ(Receiver, 
                                                  llvm::Constant::getNullValue(Receiver->getType()));
        Builder.CreateCondBr(isNil, continueBB, messageBB);
        CGF.EmitBlock(messageBB);
    }
    
    IdTy = cast<llvm::PointerType>(CGM.getTypes().ConvertType(ASTIdTy));
    llvm::Value *cmd;
    if (Method)
        cmd = GetSelector(Builder, Method);
    else
        cmd = GetSelector(Builder, Sel);
    cmd = EnforceType(Builder, cmd, SelectorTy);
    Receiver = EnforceType(Builder, Receiver, IdTy);
    
    llvm::Value *impMD[] = {
        llvm::MDString::get(VMContext, Sel.getAsString()),
        llvm::MDString::get(VMContext, Class ? Class->getNameAsString() :""),
        llvm::ConstantInt::get(llvm::Type::getInt1Ty(VMContext), Class!=0)
    };
    llvm::MDNode *node = llvm::MDNode::get(VMContext, impMD);
    
    // Get the IMP to call
    llvm::Value *imp = LookupIMP(CGF, Receiver, cmd, node);
    
    CallArgList ActualArgs;
    ActualArgs.add(RValue::get(Receiver), ASTIdTy);
    ActualArgs.add(RValue::get(cmd), CGF.getContext().getObjCSelType());
    ActualArgs.insert(ActualArgs.end(), CallArgs.begin(), CallArgs.end());
    
    CodeGenTypes &Types = CGM.getTypes();
    const CGFunctionInfo &FnInfo = Types.getFunctionInfo(ResultType, ActualArgs,
                                                         FunctionType::ExtInfo());
    const llvm::FunctionType *impType =
    Types.GetFunctionType(FnInfo, Method ? Method->isVariadic() : false);
    imp = EnforceType(Builder, imp, llvm::PointerType::getUnqual(impType));
    
    
    // For sender-aware dispatch, we pass the sender as the third argument to a
    // lookup function.  When sending messages from C code, the sender is nil.
    // objc_msg_lookup_sender(id *receiver, SEL selector, id sender);
    llvm::Instruction *call;
    RValue msgRet = CGF.EmitCall(FnInfo, imp, Return, ActualArgs,
                                 0, &call);
    call->setMetadata(msgSendMDKind, node);
    
    
    if (!isPointerSizedReturn) {
        messageBB = CGF.Builder.GetInsertBlock();
        CGF.Builder.CreateBr(continueBB);
        CGF.EmitBlock(continueBB);
        if (msgRet.isScalar()) {
            llvm::Value *v = msgRet.getScalarVal();
            llvm::PHINode *phi = Builder.CreatePHI(v->getType(), 2);
            phi->addIncoming(v, messageBB);
            phi->addIncoming(llvm::Constant::getNullValue(v->getType()), startBB);
            msgRet = RValue::get(phi);
        } else if (msgRet.isAggregate()) {
            llvm::Value *v = msgRet.getAggregateAddr();
            llvm::PHINode *phi = Builder.CreatePHI(v->getType(), 2);
            const llvm::PointerType *RetTy = cast<llvm::PointerType>(v->getType());
            llvm::AllocaInst *NullVal = 
            CGF.CreateTempAlloca(RetTy->getElementType(), "null");
            CGF.InitTempAlloca(NullVal,
                               llvm::Constant::getNullValue(RetTy->getElementType()));
            phi->addIncoming(v, messageBB);
            phi->addIncoming(NullVal, startBB);
            msgRet = RValue::getAggregate(phi);
        } else /* isComplex() */ {
            std::pair<llvm::Value*,llvm::Value*> v = msgRet.getComplexVal();
            llvm::PHINode *phi = Builder.CreatePHI(v.first->getType(), 2);
            phi->addIncoming(v.first, messageBB);
            phi->addIncoming(llvm::Constant::getNullValue(v.first->getType()),
                             startBB);
            llvm::PHINode *phi2 = Builder.CreatePHI(v.second->getType(), 2);
            phi2->addIncoming(v.second, messageBB);
            phi2->addIncoming(llvm::Constant::getNullValue(v.second->getType()),
                              startBB);
            msgRet = RValue::getComplex(phi, phi2);
        }
    }
    return msgRet;
}

/// Generates a MethodList.  Used in construction of a objc_class and
/// objc_category structures.
llvm::Constant *CGObjCCocotron::GenerateMethodList(const llvm::StringRef &ClassName,
                                              const llvm::StringRef &CategoryName,
                                              const llvm::SmallVectorImpl<Selector> &MethodSels,
                                              const llvm::SmallVectorImpl<llvm::Constant *> &MethodTypes,
                                              bool isClassMethodList) {
    if (MethodSels.empty())
        return NULLPtr;
    // Get the method structure type.
    llvm::StructType *ObjCMethodTy = llvm::StructType::get(PtrToInt8Ty, // Really a selector, but the runtime creates it us.
                                                           PtrToInt8Ty, // Method types
                                                           IMPTy, //Method pointer
                                                           NULL);
    std::vector<llvm::Constant*> Methods;
    std::vector<llvm::Constant*> Elements;
    for (unsigned int i = 0, e = MethodTypes.size(); i < e; ++i) {
        Elements.clear();
        llvm::Constant *Method =
        TheModule.getFunction(SymbolNameForMethod(ClassName, CategoryName,
                                                  MethodSels[i],
                                                  isClassMethodList));
        assert(Method && "Can't generate metadata for method that doesn't exist");
        llvm::Constant *C = MakeConstantString(MethodSels[i].getAsString());
        Elements.push_back(C);
        Elements.push_back(MethodTypes[i]);
        Method = llvm::ConstantExpr::getBitCast(Method,
                                                IMPTy);
        Elements.push_back(Method);
        Methods.push_back(llvm::ConstantStruct::get(ObjCMethodTy, Elements));
    }
    
    // Array of method structures
    llvm::ArrayType *ObjCMethodArrayTy = llvm::ArrayType::get(ObjCMethodTy,
                                                              Methods.size());
    llvm::Constant *MethodArray = llvm::ConstantArray::get(ObjCMethodArrayTy,
                                                           Methods);
    
    // Structure containing list pointer, array and array count
    llvm::SmallVector<const llvm::Type*, 16> ObjCMethodListFields;
    llvm::PATypeHolder OpaqueNextTy = llvm::OpaqueType::get(VMContext);
    llvm::Type *NextPtrTy = llvm::PointerType::getUnqual(OpaqueNextTy);
    llvm::StructType *ObjCMethodListTy = llvm::StructType::get(NextPtrTy,
                                                               IntTy,
                                                               ObjCMethodArrayTy,
                                                               NULL);
    // Refine next pointer type to concrete type
    llvm::cast<llvm::OpaqueType>(
                                 OpaqueNextTy.get())->refineAbstractTypeTo(ObjCMethodListTy);
    ObjCMethodListTy = llvm::cast<llvm::StructType>(OpaqueNextTy.get());
    
    Methods.clear();
    Methods.push_back(llvm::ConstantPointerNull::get(
                                                     llvm::PointerType::getUnqual(ObjCMethodListTy)));
    Methods.push_back(llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext),
                                             MethodTypes.size()));
    Methods.push_back(MethodArray);
    
    // Create an instance of the structure
    return MakeGlobal(ObjCMethodListTy, Methods, ".objc_method_list");
}

/// Generates an IvarList.  Used in construction of a objc_class.
llvm::Constant *CGObjCCocotron::GenerateIvarList(
                                            const llvm::SmallVectorImpl<llvm::Constant *>  &IvarNames,
                                            const llvm::SmallVectorImpl<llvm::Constant *>  &IvarTypes,
                                            const llvm::SmallVectorImpl<llvm::Constant *>  &IvarOffsets) {
    if (IvarNames.size() == 0)
        return NULLPtr;
    // Get the method structure type.
    llvm::StructType *ObjCIvarTy = llvm::StructType::get(PtrToInt8Ty,
                                                         PtrToInt8Ty,
                                                         IntTy,
                                                         NULL);
    std::vector<llvm::Constant*> Ivars;
    std::vector<llvm::Constant*> Elements;
    for (unsigned int i = 0, e = IvarNames.size() ; i < e ; i++) {
        Elements.clear();
        Elements.push_back(IvarNames[i]);
        Elements.push_back(IvarTypes[i]);
        Elements.push_back(IvarOffsets[i]);
        Ivars.push_back(llvm::ConstantStruct::get(ObjCIvarTy, Elements));
    }
    
    // Array of method structures
    llvm::ArrayType *ObjCIvarArrayTy = llvm::ArrayType::get(ObjCIvarTy,
                                                            IvarNames.size());
    
    
    Elements.clear();
    Elements.push_back(llvm::ConstantInt::get(IntTy, (int)IvarNames.size()));
    Elements.push_back(llvm::ConstantArray::get(ObjCIvarArrayTy, Ivars));
    // Structure containing array and array count
    llvm::StructType *ObjCIvarListTy = llvm::StructType::get(IntTy,
                                                             ObjCIvarArrayTy,
                                                             NULL);
    
    // Create an instance of the structure
    return MakeGlobal(ObjCIvarListTy, Elements, ".objc_ivar_list");
}

/// Generate a class structure
llvm::Constant *CGObjCCocotron::GenerateClassStructure(
                                                  llvm::Constant *MetaClass,
                                                  llvm::Constant *SuperClass,
                                                  unsigned info,
                                                  const char *Name,
                                                  llvm::Constant *Version,
                                                  llvm::Constant *InstanceSize,
                                                  llvm::Constant *IVars,
                                                  llvm::Constant *Methods,
                                                  llvm::Constant *Protocols,
                                                  llvm::Constant *IvarOffsets,
                                                  llvm::Constant *Properties,
                                                  bool isMeta) {
    // Set up the class structure
    // Note:  Several of these are char*s when they should be ids.  This is
    // because the runtime performs this translation on load.
    //
    // Fields marked New ABI are part of the GNUstep runtime.  We emit them
    // anyway; the classes will still work with the GNU runtime, they will just
    // be ignored.
    
    
    llvm::StructType *ClassTy = llvm::StructType::get(PtrToInt8Ty,        // class_pointer
                                                      PtrToInt8Ty,        // super_class
                                                      PtrToInt8Ty,        // name
                                                      LongTy,             // version
                                                      LongTy,             // info
                                                      LongTy,             // instance_size
                                                      IVars->getType(),   // ivars
                                                      Methods->getType(), // methods
                                                      
                                                      // These are all filled in by the runtime, so we pretend
                                                      CachePtrTy,              // cache
                                                      PtrTy,              // protocols
                                                      PtrToInt8Ty,              // ivar layout
                                                      // New ABI:
                                                      LongTy,                 // abi_version
                                                      IvarOffsets->getType(), // ivar_offsets
                                                      Properties->getType(),  // properties
                                                      NULL);
    llvm::Constant *Zero = llvm::ConstantInt::get(LongTy, 0);
    // Fill in the structure
    std::vector<llvm::Constant*> Elements;
    Elements.push_back(llvm::ConstantExpr::getBitCast(MetaClass, PtrToInt8Ty));
    Elements.push_back(SuperClass);
    Elements.push_back(MakeConstantString(Name, ".class_name"));
    Elements.push_back(Zero);
    Elements.push_back(llvm::ConstantInt::get(LongTy, info));
    if (isMeta) {
        llvm::TargetData td(&TheModule);
        Elements.push_back(
                           llvm::ConstantInt::get(LongTy,
                                                  td.getTypeSizeInBits(ClassTy) /
                                                  CGM.getContext().getCharWidth()));
    } else
        Elements.push_back(InstanceSize);
    Elements.push_back(IVars);
    Elements.push_back(Methods);
    Elements.push_back(llvm::Constant::getNullValue(CachePtrTy)); //cache
    Elements.push_back(llvm::ConstantExpr::getBitCast(Protocols, PtrTy)); //protocol
    Elements.push_back(NULLPtr); //ivar layout
    Elements.push_back(Zero); //abi version
    Elements.push_back(IvarOffsets); //ivar_offset
    Elements.push_back(Properties); //propeties
    // Create an instance of the structure
    // This is now an externally visible symbol, so that we can speed up class
    // messages in the next ABI.
    return MakeGlobal(ClassTy, Elements, (isMeta ? "_OBJC_METACLASS_":
                                          "_OBJC_CLASS_") + std::string(Name), llvm::GlobalValue::ExternalLinkage);
}

llvm::Constant *CGObjCCocotron::GenerateProtocolMethodList(
                                                      const llvm::SmallVectorImpl<llvm::Constant *>  &MethodNames,
                                                      const llvm::SmallVectorImpl<llvm::Constant *>  &MethodTypes) {
    // Get the method structure type.
    llvm::StructType *ObjCMethodDescTy = llvm::StructType::get(PtrToInt8Ty, // Really a selector, but the runtime does the casting for us.
                                                               PtrToInt8Ty,
                                                               NULL);
    std::vector<llvm::Constant*> Methods;
    std::vector<llvm::Constant*> Elements;
    for (unsigned int i = 0, e = MethodTypes.size() ; i < e ; i++) {
        Elements.clear();
        Elements.push_back(MethodNames[i]);
        Elements.push_back(MethodTypes[i]);
        Methods.push_back(llvm::ConstantStruct::get(ObjCMethodDescTy, Elements));
    }
    llvm::ArrayType *ObjCMethodArrayTy = llvm::ArrayType::get(ObjCMethodDescTy,
                                                              MethodNames.size());
    llvm::Constant *Array = llvm::ConstantArray::get(ObjCMethodArrayTy,
                                                     Methods);
    llvm::StructType *ObjCMethodDescListTy = llvm::StructType::get(IntTy, ObjCMethodArrayTy, NULL);
    Methods.clear();
    Methods.push_back(llvm::ConstantInt::get(IntTy, MethodNames.size()));
    Methods.push_back(Array);
    return MakeGlobal(ObjCMethodDescListTy, Methods, ".objc_method_list");
}

// Create the protocol list structure used in classes, categories and so on
llvm::Constant *CGObjCCocotron::GenerateProtocolList(
                                                const llvm::SmallVectorImpl<std::string> &Protocols) {
    llvm::ArrayType *ProtocolArrayTy = llvm::ArrayType::get(PtrToInt8Ty,
                                                            Protocols.size());
    llvm::StructType *ProtocolListTy = llvm::StructType::get(PtrTy, //Should be a recurisve pointer, but it's always NULL here.
                                                             SizeTy,
                                                             ProtocolArrayTy,
                                                             NULL);
    std::vector<llvm::Constant*> Elements;
    for (const std::string *iter = Protocols.begin(), *endIter = Protocols.end();
         iter != endIter ; iter++) {
        llvm::Constant *protocol = 0;
        llvm::StringMap<llvm::Constant*>::iterator value =
        ExistingProtocols.find(*iter);
        if (value == ExistingProtocols.end()) {
            protocol = GenerateEmptyProtocol(*iter);
        } else {
            protocol = value->getValue();
        }
        llvm::Constant *Ptr = llvm::ConstantExpr::getBitCast(protocol,
                                                             PtrToInt8Ty);
        Elements.push_back(Ptr);
    }
    llvm::Constant * ProtocolArray = llvm::ConstantArray::get(ProtocolArrayTy,
                                                              Elements);
    Elements.clear();
    Elements.push_back(NULLPtr);
    Elements.push_back(llvm::ConstantInt::get(LongTy, Protocols.size()));
    Elements.push_back(ProtocolArray);
    return MakeGlobal(ProtocolListTy, Elements, ".objc_protocol_list");
}

llvm::Value *CGObjCCocotron::GenerateProtocolRef(CGBuilderTy &Builder,
                                            const ObjCProtocolDecl *PD) {
    llvm::Value *protocol = ExistingProtocols[PD->getNameAsString()];
    const llvm::Type *T =
    CGM.getTypes().ConvertType(CGM.getContext().getObjCProtoType());
    return Builder.CreateBitCast(protocol, llvm::PointerType::getUnqual(T));
}

llvm::Constant *CGObjCCocotron::GenerateEmptyProtocol(
                                                 const std::string &ProtocolName) {
    llvm::SmallVector<std::string, 0> EmptyStringVector;
    llvm::SmallVector<llvm::Constant*, 0> EmptyConstantVector;
    
    llvm::Constant *ProtocolList = GenerateProtocolList(EmptyStringVector);
    llvm::Constant *MethodList =
    GenerateProtocolMethodList(EmptyConstantVector, EmptyConstantVector);
    // Protocols are objects containing lists of the methods implemented and
    // protocols adopted.
    llvm::StructType *ProtocolTy = llvm::StructType::get(IdTy,
                                                         PtrToInt8Ty,
                                                         ProtocolList->getType(),
                                                         MethodList->getType(),
                                                         MethodList->getType(),
                                                         MethodList->getType(),
                                                         MethodList->getType(),
                                                         NULL);
    std::vector<llvm::Constant*> Elements;
    // The isa pointer must be set to a magic number so the runtime knows it's
    // the correct layout.
    Elements.push_back(llvm::ConstantExpr::getIntToPtr(
                                                       llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext),
                                                                              ProtocolVersion), IdTy));
    Elements.push_back(MakeConstantString(ProtocolName, ".objc_protocol_name"));
    Elements.push_back(ProtocolList);
    Elements.push_back(MethodList);
    Elements.push_back(MethodList);
    Elements.push_back(MethodList);
    Elements.push_back(MethodList);
    return MakeGlobal(ProtocolTy, Elements, ".objc_protocol");
}

void CGObjCCocotron::GenerateProtocol(const ObjCProtocolDecl *PD) {
    ASTContext &Context = CGM.getContext();
    std::string ProtocolName = PD->getNameAsString();
    llvm::SmallVector<std::string, 16> Protocols;
    for (ObjCProtocolDecl::protocol_iterator PI = PD->protocol_begin(),
         E = PD->protocol_end(); PI != E; ++PI)
        Protocols.push_back((*PI)->getNameAsString());
    llvm::SmallVector<llvm::Constant*, 16> InstanceMethodNames;
    llvm::SmallVector<llvm::Constant*, 16> InstanceMethodTypes;
    llvm::SmallVector<llvm::Constant*, 16> OptionalInstanceMethodNames;
    llvm::SmallVector<llvm::Constant*, 16> OptionalInstanceMethodTypes;
    for (ObjCProtocolDecl::instmeth_iterator iter = PD->instmeth_begin(),
         E = PD->instmeth_end(); iter != E; iter++) {
        std::string TypeStr;
        Context.getObjCEncodingForMethodDecl(*iter, TypeStr);
        if ((*iter)->getImplementationControl() == ObjCMethodDecl::Optional) {
            InstanceMethodNames.push_back(
                                          MakeConstantString((*iter)->getSelector().getAsString()));
            InstanceMethodTypes.push_back(MakeConstantString(TypeStr));
        } else {
            OptionalInstanceMethodNames.push_back(
                                                  MakeConstantString((*iter)->getSelector().getAsString()));
            OptionalInstanceMethodTypes.push_back(MakeConstantString(TypeStr));
        }
    }
    // Collect information about class methods:
    llvm::SmallVector<llvm::Constant*, 16> ClassMethodNames;
    llvm::SmallVector<llvm::Constant*, 16> ClassMethodTypes;
    llvm::SmallVector<llvm::Constant*, 16> OptionalClassMethodNames;
    llvm::SmallVector<llvm::Constant*, 16> OptionalClassMethodTypes;
    for (ObjCProtocolDecl::classmeth_iterator
         iter = PD->classmeth_begin(), endIter = PD->classmeth_end();
         iter != endIter ; iter++) {
        std::string TypeStr;
        Context.getObjCEncodingForMethodDecl((*iter),TypeStr);
        if ((*iter)->getImplementationControl() == ObjCMethodDecl::Optional) {
            ClassMethodNames.push_back(
                                       MakeConstantString((*iter)->getSelector().getAsString()));
            ClassMethodTypes.push_back(MakeConstantString(TypeStr));
        } else {
            OptionalClassMethodNames.push_back(
                                               MakeConstantString((*iter)->getSelector().getAsString()));
            OptionalClassMethodTypes.push_back(MakeConstantString(TypeStr));
        }
    }
    
    llvm::Constant *ProtocolList = GenerateProtocolList(Protocols);
    llvm::Constant *InstanceMethodList =
    GenerateProtocolMethodList(InstanceMethodNames, InstanceMethodTypes);
    llvm::Constant *ClassMethodList =
    GenerateProtocolMethodList(ClassMethodNames, ClassMethodTypes);
    llvm::Constant *OptionalInstanceMethodList =
    GenerateProtocolMethodList(OptionalInstanceMethodNames,
                               OptionalInstanceMethodTypes);
    llvm::Constant *OptionalClassMethodList =
    GenerateProtocolMethodList(OptionalClassMethodNames,
                               OptionalClassMethodTypes);
    
    // Property metadata: name, attributes, isSynthesized, setter name, setter
    // types, getter name, getter types.
    // The isSynthesized value is always set to 0 in a protocol.  It exists to
    // simplify the runtime library by allowing it to use the same data
    // structures for protocol metadata everywhere.
    llvm::StructType *PropertyMetadataTy = llvm::StructType::get(PtrToInt8Ty, Int8Ty, Int8Ty, PtrToInt8Ty, PtrToInt8Ty, PtrToInt8Ty,
                                                                 PtrToInt8Ty, NULL);
    std::vector<llvm::Constant*> Properties;
    std::vector<llvm::Constant*> OptionalProperties;
    
    // Add all of the property methods need adding to the method list and to the
    // property metadata list.
    for (ObjCContainerDecl::prop_iterator
         iter = PD->prop_begin(), endIter = PD->prop_end();
         iter != endIter ; iter++) {
        std::vector<llvm::Constant*> Fields;
        ObjCPropertyDecl *property = (*iter);
        
        Fields.push_back(MakeConstantString(property->getNameAsString()));
        Fields.push_back(llvm::ConstantInt::get(Int8Ty,
                                                property->getPropertyAttributes()));
        Fields.push_back(llvm::ConstantInt::get(Int8Ty, 0));
        if (ObjCMethodDecl *getter = property->getGetterMethodDecl()) {
            std::string TypeStr;
            Context.getObjCEncodingForMethodDecl(getter,TypeStr);
            llvm::Constant *TypeEncoding = MakeConstantString(TypeStr);
            InstanceMethodTypes.push_back(TypeEncoding);
            Fields.push_back(MakeConstantString(getter->getSelector().getAsString()));
            Fields.push_back(TypeEncoding);
        } else {
            Fields.push_back(NULLPtr);
            Fields.push_back(NULLPtr);
        }
        if (ObjCMethodDecl *setter = property->getSetterMethodDecl()) {
            std::string TypeStr;
            Context.getObjCEncodingForMethodDecl(setter,TypeStr);
            llvm::Constant *TypeEncoding = MakeConstantString(TypeStr);
            InstanceMethodTypes.push_back(TypeEncoding);
            Fields.push_back(MakeConstantString(setter->getSelector().getAsString()));
            Fields.push_back(TypeEncoding);
        } else {
            Fields.push_back(NULLPtr);
            Fields.push_back(NULLPtr);
        }
        if (property->getPropertyImplementation() == ObjCPropertyDecl::Optional) {
            OptionalProperties.push_back(llvm::ConstantStruct::get(PropertyMetadataTy, Fields));
        } else {
            Properties.push_back(llvm::ConstantStruct::get(PropertyMetadataTy, Fields));
        }
    }
    llvm::Constant *PropertyArray = llvm::ConstantArray::get(
                                                             llvm::ArrayType::get(PropertyMetadataTy, Properties.size()), Properties);
    llvm::Constant* PropertyListInitFields[] =
    {llvm::ConstantInt::get(IntTy, Properties.size()), NULLPtr, PropertyArray};
    
    llvm::Constant *PropertyListInit =
    llvm::ConstantStruct::getAnon(PropertyListInitFields);
    llvm::Constant *PropertyList = new llvm::GlobalVariable(TheModule,
                                                            PropertyListInit->getType(), false, llvm::GlobalValue::InternalLinkage,
                                                            PropertyListInit, ".objc_property_list");
    
    llvm::Constant *OptionalPropertyArray =
    llvm::ConstantArray::get(llvm::ArrayType::get(PropertyMetadataTy,
                                                  OptionalProperties.size()) , OptionalProperties);
    llvm::Constant* OptionalPropertyListInitFields[] = {
        llvm::ConstantInt::get(IntTy, OptionalProperties.size()), NULLPtr,
        OptionalPropertyArray };
    
    llvm::Constant *OptionalPropertyListInit =
    llvm::ConstantStruct::getAnon(OptionalPropertyListInitFields);
    llvm::Constant *OptionalPropertyList = new llvm::GlobalVariable(TheModule,
                                                                    OptionalPropertyListInit->getType(), false,
                                                                    llvm::GlobalValue::InternalLinkage, OptionalPropertyListInit,
                                                                    ".objc_property_list");
    
    // Protocols are objects containing lists of the methods implemented and
    // protocols adopted.
    llvm::StructType *ProtocolTy = llvm::StructType::get(IdTy,
                                                         PtrToInt8Ty,
                                                         ProtocolList->getType(),
                                                         InstanceMethodList->getType(),
                                                         ClassMethodList->getType(),
                                                         OptionalInstanceMethodList->getType(),
                                                         OptionalClassMethodList->getType(),
                                                         PropertyList->getType(),
                                                         OptionalPropertyList->getType(),
                                                         NULL);
    std::vector<llvm::Constant*> Elements;
    // The isa pointer must be set to a magic number so the runtime knows it's
    // the correct layout.
    Elements.push_back(llvm::ConstantExpr::getIntToPtr(
                                                       llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext),
                                                                              ProtocolVersion), IdTy));
    Elements.push_back(MakeConstantString(ProtocolName, ".objc_protocol_name"));
    Elements.push_back(ProtocolList);
    Elements.push_back(InstanceMethodList);
    Elements.push_back(ClassMethodList);
    Elements.push_back(OptionalInstanceMethodList);
    Elements.push_back(OptionalClassMethodList);
    Elements.push_back(PropertyList);
    Elements.push_back(OptionalPropertyList);
    ExistingProtocols[ProtocolName] =
    llvm::ConstantExpr::getBitCast(MakeGlobal(ProtocolTy, Elements,
                                              ".objc_protocol"), IdTy);
}
void CGObjCCocotron::GenerateProtocolHolderCategory(void) {
    // Collect information about instance methods
    llvm::SmallVector<Selector, 1> MethodSels;
    llvm::SmallVector<llvm::Constant*, 1> MethodTypes;
    
    std::vector<llvm::Constant*> Elements;
    const std::string ClassName = "__ObjC_Protocol_Holder_Ugly_Hack";
    const std::string CategoryName = "AnotherHack";
    Elements.push_back(MakeConstantString(CategoryName));
    Elements.push_back(MakeConstantString(ClassName));
    // Instance method list
    Elements.push_back(llvm::ConstantExpr::getBitCast(GenerateMethodList(
                                                                         ClassName, CategoryName, MethodSels, MethodTypes, false), PtrTy));
    // Class method list
    Elements.push_back(llvm::ConstantExpr::getBitCast(GenerateMethodList(
                                                                         ClassName, CategoryName, MethodSels, MethodTypes, true), PtrTy));
    // Protocol list
    llvm::ArrayType *ProtocolArrayTy = llvm::ArrayType::get(PtrTy,
                                                            ExistingProtocols.size());
    llvm::StructType *ProtocolListTy = llvm::StructType::get(PtrTy, //Should be a recurisve pointer, but it's always NULL here.
                                                             SizeTy,
                                                             ProtocolArrayTy,
                                                             NULL);
    std::vector<llvm::Constant*> ProtocolElements;
    for (llvm::StringMapIterator<llvm::Constant*> iter =
         ExistingProtocols.begin(), endIter = ExistingProtocols.end();
         iter != endIter ; iter++) {
        llvm::Constant *Ptr = llvm::ConstantExpr::getBitCast(iter->getValue(),
                                                             PtrTy);
        ProtocolElements.push_back(Ptr);
    }
    llvm::Constant * ProtocolArray = llvm::ConstantArray::get(ProtocolArrayTy,
                                                              ProtocolElements);
    ProtocolElements.clear();
    ProtocolElements.push_back(NULLPtr);
    ProtocolElements.push_back(llvm::ConstantInt::get(LongTy,
                                                      ExistingProtocols.size()));
    ProtocolElements.push_back(ProtocolArray);
    Elements.push_back(llvm::ConstantExpr::getBitCast(MakeGlobal(ProtocolListTy,
                                                                 ProtocolElements, ".objc_protocol_list"), PtrTy));
    Categories.push_back(llvm::ConstantExpr::getBitCast(
                                                        MakeGlobal(llvm::StructType::get(PtrToInt8Ty, PtrToInt8Ty,
                                                                                         PtrTy, PtrTy, PtrTy, NULL), Elements), PtrTy));
}

void CGObjCCocotron::GenerateCategory(const ObjCCategoryImplDecl *OCD) {
    std::string ClassName = OCD->getClassInterface()->getNameAsString();
    std::string CategoryName = OCD->getNameAsString();
    // Collect information about instance methods
    llvm::SmallVector<Selector, 16> InstanceMethodSels;
    llvm::SmallVector<llvm::Constant*, 16> InstanceMethodTypes;
    for (ObjCCategoryImplDecl::instmeth_iterator
         iter = OCD->instmeth_begin(), endIter = OCD->instmeth_end();
         iter != endIter ; iter++) {
        InstanceMethodSels.push_back((*iter)->getSelector());
        std::string TypeStr;
        CGM.getContext().getObjCEncodingForMethodDecl(*iter,TypeStr);
        InstanceMethodTypes.push_back(MakeConstantString(TypeStr));
    }
    
    // Collect information about class methods
    llvm::SmallVector<Selector, 16> ClassMethodSels;
    llvm::SmallVector<llvm::Constant*, 16> ClassMethodTypes;
    for (ObjCCategoryImplDecl::classmeth_iterator
         iter = OCD->classmeth_begin(), endIter = OCD->classmeth_end();
         iter != endIter ; iter++) {
        ClassMethodSels.push_back((*iter)->getSelector());
        std::string TypeStr;
        CGM.getContext().getObjCEncodingForMethodDecl(*iter,TypeStr);
        ClassMethodTypes.push_back(MakeConstantString(TypeStr));
    }
    
    // Collect the names of referenced protocols
    llvm::SmallVector<std::string, 16> Protocols;
    const ObjCCategoryDecl *CatDecl = OCD->getCategoryDecl();
    const ObjCList<ObjCProtocolDecl> &Protos = CatDecl->getReferencedProtocols();
    for (ObjCList<ObjCProtocolDecl>::iterator I = Protos.begin(),
         E = Protos.end(); I != E; ++I)
        Protocols.push_back((*I)->getNameAsString());
    
    std::vector<llvm::Constant*> Elements;
    Elements.push_back(MakeConstantString(CategoryName));
    Elements.push_back(MakeConstantString(ClassName));
    // Instance method list
    Elements.push_back(llvm::ConstantExpr::getBitCast(GenerateMethodList(
                                                                         ClassName, CategoryName, InstanceMethodSels, InstanceMethodTypes,
                                                                         false), PtrTy));
    // Class method list
    Elements.push_back(llvm::ConstantExpr::getBitCast(GenerateMethodList(
                                                                         ClassName, CategoryName, ClassMethodSels, ClassMethodTypes, true),
                                                      PtrTy));
    // Protocol list
    Elements.push_back(llvm::ConstantExpr::getBitCast(
                                                      GenerateProtocolList(Protocols), PtrTy));
    Categories.push_back(llvm::ConstantExpr::getBitCast(
                                                        MakeGlobal(llvm::StructType::get(PtrToInt8Ty, PtrToInt8Ty,
                                                                                         PtrTy, PtrTy, PtrTy, NULL), Elements), PtrTy));
}

llvm::Constant *CGObjCCocotron::GeneratePropertyList(const ObjCImplementationDecl *OID,
                                                llvm::SmallVectorImpl<Selector> &InstanceMethodSels,
                                                llvm::SmallVectorImpl<llvm::Constant*> &InstanceMethodTypes) {
    ASTContext &Context = CGM.getContext();
    //
    // Property metadata: name, attributes, isSynthesized, setter name, setter
    // types, getter name, getter types.
    llvm::StructType *PropertyMetadataTy = llvm::StructType::get(PtrToInt8Ty, Int8Ty, Int8Ty, PtrToInt8Ty, PtrToInt8Ty, PtrToInt8Ty,
                                                                 PtrToInt8Ty, NULL);
    std::vector<llvm::Constant*> Properties;
    
    
    // Add all of the property methods need adding to the method list and to the
    // property metadata list.
    for (ObjCImplDecl::propimpl_iterator
         iter = OID->propimpl_begin(), endIter = OID->propimpl_end();
         iter != endIter ; iter++) {
        std::vector<llvm::Constant*> Fields;
        ObjCPropertyDecl *property = (*iter)->getPropertyDecl();
        ObjCPropertyImplDecl *propertyImpl = *iter;
        bool isSynthesized = (propertyImpl->getPropertyImplementation() == 
                              ObjCPropertyImplDecl::Synthesize);
        
        Fields.push_back(MakeConstantString(property->getNameAsString()));
        Fields.push_back(llvm::ConstantInt::get(Int8Ty,
                                                property->getPropertyAttributes()));
        Fields.push_back(llvm::ConstantInt::get(Int8Ty, isSynthesized));
        if (ObjCMethodDecl *getter = property->getGetterMethodDecl()) {
            std::string TypeStr;
            Context.getObjCEncodingForMethodDecl(getter,TypeStr);
            llvm::Constant *TypeEncoding = MakeConstantString(TypeStr);
            if (isSynthesized) {
                InstanceMethodTypes.push_back(TypeEncoding);
                InstanceMethodSels.push_back(getter->getSelector());
            }
            Fields.push_back(MakeConstantString(getter->getSelector().getAsString()));
            Fields.push_back(TypeEncoding);
        } else {
            Fields.push_back(NULLPtr);
            Fields.push_back(NULLPtr);
        }
        if (ObjCMethodDecl *setter = property->getSetterMethodDecl()) {
            std::string TypeStr;
            Context.getObjCEncodingForMethodDecl(setter,TypeStr);
            llvm::Constant *TypeEncoding = MakeConstantString(TypeStr);
            if (isSynthesized) {
                InstanceMethodTypes.push_back(TypeEncoding);
                InstanceMethodSels.push_back(setter->getSelector());
            }
            Fields.push_back(MakeConstantString(setter->getSelector().getAsString()));
            Fields.push_back(TypeEncoding);
        } else {
            Fields.push_back(NULLPtr);
            Fields.push_back(NULLPtr);
        }
        Properties.push_back(llvm::ConstantStruct::get(PropertyMetadataTy, Fields));
    }
    llvm::ArrayType *PropertyArrayTy =
    llvm::ArrayType::get(PropertyMetadataTy, Properties.size());
    llvm::Constant *PropertyArray = llvm::ConstantArray::get(PropertyArrayTy,
                                                             Properties);
    llvm::Constant* PropertyListInitFields[] =
    {llvm::ConstantInt::get(IntTy, Properties.size()), NULLPtr, PropertyArray};
    
    llvm::Constant *PropertyListInit =
    llvm::ConstantStruct::getAnon(PropertyListInitFields);
    return new llvm::GlobalVariable(TheModule, PropertyListInit->getType(), false,
                                    llvm::GlobalValue::InternalLinkage, PropertyListInit,
                                    ".objc_property_list");
}

void CGObjCCocotron::GenerateClass(const ObjCImplementationDecl *OID) {
    ASTContext &Context = CGM.getContext();
    
    // Get the superclass name.
    const ObjCInterfaceDecl * SuperClassDecl =
    OID->getClassInterface()->getSuperClass();
    std::string SuperClassName;
    if (SuperClassDecl) {
        SuperClassName = SuperClassDecl->getNameAsString();
        EmitClassRef(SuperClassName);
    }
    
    // Get the class name
    ObjCInterfaceDecl *ClassDecl =
    const_cast<ObjCInterfaceDecl *>(OID->getClassInterface());
    std::string ClassName = ClassDecl->getNameAsString();
    // Emit the symbol that is used to generate linker errors if this class is
    // referenced in other modules but not declared.
    std::string classSymbolName = "__objc_class_name_" + ClassName;
    if (llvm::GlobalVariable *symbol =
        TheModule.getGlobalVariable(classSymbolName)) {
        symbol->setInitializer(llvm::ConstantInt::get(LongTy, 0));
    } else {
        new llvm::GlobalVariable(TheModule, LongTy, false,
                                 llvm::GlobalValue::ExternalLinkage, llvm::ConstantInt::get(LongTy, 0),
                                 classSymbolName);
    }
    
    // Get the size of instances.
    int instanceSize = 
    Context.getASTObjCImplementationLayout(OID).getSize().getQuantity();
    
    // Collect information about instance variables.
    llvm::SmallVector<llvm::Constant*, 16> IvarNames;
    llvm::SmallVector<llvm::Constant*, 16> IvarTypes;
    llvm::SmallVector<llvm::Constant*, 16> IvarOffsets;
    
    std::vector<llvm::Constant*> IvarOffsetValues;
    
    int superInstanceSize = !SuperClassDecl ? 0 :
    Context.getASTObjCInterfaceLayout(SuperClassDecl).getSize().getQuantity();
    // For non-fragile ivars, set the instance size to 0 - {the size of just this
    // class}.  The runtime will then set this to the correct value on load.
    if (CGM.getContext().getLangOptions().ObjCNonFragileABI && !true) {
        instanceSize = 0 - (instanceSize - superInstanceSize);
    }
    
    // Collect declared and synthesized ivars.
    llvm::SmallVector<ObjCIvarDecl*, 16> OIvars;
    CGM.getContext().ShallowCollectObjCIvars(ClassDecl, OIvars);
    
    for (unsigned i = 0, e = OIvars.size(); i != e; ++i) {
        ObjCIvarDecl *IVD = OIvars[i];
        // Store the name
        IvarNames.push_back(MakeConstantString(IVD->getNameAsString()));
        // Get the type encoding for this ivar
        std::string TypeStr;
        Context.getObjCEncodingForType(IVD->getType(), TypeStr);
        IvarTypes.push_back(MakeConstantString(TypeStr));
        // Get the offset
        uint64_t BaseOffset = ComputeIvarBaseOffset(CGM, OID, IVD);
        uint64_t Offset = BaseOffset;
        if (CGM.getContext().getLangOptions().ObjCNonFragileABI && !true) {
            Offset = BaseOffset - superInstanceSize;
        }
        IvarOffsets.push_back(
                              llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), Offset));
        IvarOffsetValues.push_back(new llvm::GlobalVariable(TheModule, IntTy,
                                                            false, llvm::GlobalValue::ExternalLinkage,
                                                            llvm::ConstantInt::get(IntTy, Offset),
                                                            "__objc_ivar_offset_value_" + ClassName +"." +
                                                            IVD->getNameAsString()));
    }
    llvm::GlobalVariable *IvarOffsetArray =
    MakeGlobalArray(PtrToIntTy, IvarOffsetValues, ".ivar.offsets");
    
    
    // Collect information about instance methods
    llvm::SmallVector<Selector, 16> InstanceMethodSels;
    llvm::SmallVector<llvm::Constant*, 16> InstanceMethodTypes;
    for (ObjCImplementationDecl::instmeth_iterator
         iter = OID->instmeth_begin(), endIter = OID->instmeth_end();
         iter != endIter ; iter++) {
        InstanceMethodSels.push_back((*iter)->getSelector());
        std::string TypeStr;
        Context.getObjCEncodingForMethodDecl((*iter),TypeStr);
        InstanceMethodTypes.push_back(MakeConstantString(TypeStr));
    }
    
    llvm::Constant *Properties = GeneratePropertyList(OID, InstanceMethodSels,
                                                      InstanceMethodTypes);
    
    
    // Collect information about class methods
    llvm::SmallVector<Selector, 16> ClassMethodSels;
    llvm::SmallVector<llvm::Constant*, 16> ClassMethodTypes;
    for (ObjCImplementationDecl::classmeth_iterator
         iter = OID->classmeth_begin(), endIter = OID->classmeth_end();
         iter != endIter ; iter++) {
        ClassMethodSels.push_back((*iter)->getSelector());
        std::string TypeStr;
        Context.getObjCEncodingForMethodDecl((*iter),TypeStr);
        ClassMethodTypes.push_back(MakeConstantString(TypeStr));
    }
    // Collect the names of referenced protocols
    llvm::SmallVector<std::string, 16> Protocols;
    const ObjCList<ObjCProtocolDecl> &Protos =ClassDecl->getReferencedProtocols();
    for (ObjCList<ObjCProtocolDecl>::iterator I = Protos.begin(),
         E = Protos.end(); I != E; ++I)
        Protocols.push_back((*I)->getNameAsString());
    
    
    
    // Get the superclass pointer.
    llvm::Constant *SuperClass;
    if (!SuperClassName.empty()) {
        SuperClass = MakeConstantString(SuperClassName, ".super_class_name");
    } else {
        SuperClass = llvm::ConstantPointerNull::get(PtrToInt8Ty);
    }
    // Empty vector used to construct empty method lists
    llvm::SmallVector<llvm::Constant*, 1>  empty;
    // Generate the method and instance variable lists
    llvm::Constant *MethodList = GenerateMethodList(ClassName, "",
                                                    InstanceMethodSels, InstanceMethodTypes, false);
    llvm::Constant *ClassMethodList = GenerateMethodList(ClassName, "",
                                                         ClassMethodSels, ClassMethodTypes, true);
    llvm::Constant *IvarList = GenerateIvarList(IvarNames, IvarTypes,
                                                IvarOffsets);
    // Irrespective of whether we are compiling for a fragile or non-fragile ABI,
    // we emit a symbol containing the offset for each ivar in the class.  This
    // allows code compiled for the non-Fragile ABI to inherit from code compiled
    // for the legacy ABI, without causing problems.  The converse is also
    // possible, but causes all ivar accesses to be fragile.
    
    // Offset pointer for getting at the correct field in the ivar list when
    // setting up the alias.  These are: The base address for the global, the
    // ivar array (second field), the ivar in this list (set for each ivar), and
    // the offset (third field in ivar structure)
    const llvm::Type *IndexTy = llvm::Type::getInt32Ty(VMContext);
    llvm::Constant *offsetPointerIndexes[] = {Zeros[0],
        llvm::ConstantInt::get(IndexTy, 1), 0,
        llvm::ConstantInt::get(IndexTy, 2) };
    
    
    for (unsigned i = 0, e = OIvars.size(); i != e; ++i) {
        ObjCIvarDecl *IVD = OIvars[i];
        const std::string Name = "__objc_ivar_offset_" + ClassName + '.'
        + IVD->getNameAsString();
        offsetPointerIndexes[2] = llvm::ConstantInt::get(IndexTy, i);
        // Get the correct ivar field
        llvm::Constant *offsetValue = llvm::ConstantExpr::getGetElementPtr(
                                                                           IvarList, offsetPointerIndexes, 4);
        // Get the existing variable, if one exists.
        llvm::GlobalVariable *offset = TheModule.getNamedGlobal(Name);
        if (offset) {
            offset->setInitializer(offsetValue);
            // If this is the real definition, change its linkage type so that
            // different modules will use this one, rather than their private
            // copy.
            offset->setLinkage(llvm::GlobalValue::ExternalLinkage);
        } else {
            // Add a new alias if there isn't one already.
            offset = new llvm::GlobalVariable(TheModule, offsetValue->getType(),
                                              false, llvm::GlobalValue::ExternalLinkage, offsetValue, Name);
        }
    }
    //Generate metaclass for class methods
    llvm::Constant *MetaClassStruct = GenerateClassStructure(NULLPtr,
                                                             NULLPtr, 0x12L, ClassName.c_str(), 0, Zeros[0], GenerateIvarList(
                                                                                                                              empty, empty, empty), ClassMethodList, NULLPtr, NULLPtr, NULLPtr, true);
    
    // Generate the class structure
    llvm::Constant *ClassStruct =
    GenerateClassStructure(MetaClassStruct, SuperClass, 0x11L,
                           ClassName.c_str(), 0,
                           llvm::ConstantInt::get(LongTy, instanceSize), IvarList,
                           MethodList, GenerateProtocolList(Protocols), IvarOffsetArray,
                           Properties);
    
    // Resolve the class aliases, if they exist.
    if (ClassPtrAlias) {
        ClassPtrAlias->replaceAllUsesWith(
                                          llvm::ConstantExpr::getBitCast(ClassStruct, IdTy));
        ClassPtrAlias->eraseFromParent();
        ClassPtrAlias = 0;
    }
    if (MetaClassPtrAlias) {
        MetaClassPtrAlias->replaceAllUsesWith(
                                              llvm::ConstantExpr::getBitCast(MetaClassStruct, IdTy));
        MetaClassPtrAlias->eraseFromParent();
        MetaClassPtrAlias = 0;
    }
    
    // Add class structure to list to be added to the symtab later
    ClassStruct = llvm::ConstantExpr::getBitCast(ClassStruct, PtrToInt8Ty);
    Classes.push_back(ClassStruct);
}


llvm::Function *CGObjCCocotron::ModuleInitFunction() {
    // Only emit an ObjC load function if no Objective-C stuff has been called
    if (Classes.empty() && Categories.empty() && ConstantStrings.empty() &&
        ExistingProtocols.empty() && SelectorTable.empty())
        return NULL;
    
    // Add all referenced protocols to a category.
    GenerateProtocolHolderCategory();
    
    // Name the ObjC types to make the IR a bit easier to read
    TheModule.addTypeName(".objc_selector", PtrToInt8Ty);
    TheModule.addTypeName(".objc_id", IdTy);
    TheModule.addTypeName(".objc_imp", IMPTy);
    
    std::vector<llvm::Constant*> Elements;
    llvm::Constant *Statics = NULLPtr;
    // Generate statics list:
    if (ConstantStrings.size()) {
        llvm::ArrayType *StaticsArrayTy = llvm::ArrayType::get(PtrToInt8Ty,
                                                               ConstantStrings.size() + 1);
        ConstantStrings.push_back(NULLPtr);
        
        llvm::StringRef StringClass = CGM.getLangOptions().ObjCConstantStringClass;
        
        StringClass = "NSConstantString";
        Elements.push_back(MakeConstantString(StringClass,
                                              ".objc_static_class_name"));
        Elements.push_back(llvm::ConstantArray::get(StaticsArrayTy,
                                                    ConstantStrings));
        llvm::StructType *StaticsListTy =
        llvm::StructType::get(PtrToInt8Ty, StaticsArrayTy, NULL);
        llvm::Type *StaticsListPtrTy =
        llvm::PointerType::getUnqual(StaticsListTy);
        Statics = MakeGlobal(StaticsListTy, Elements, ".objc_statics");
        llvm::ArrayType *StaticsListArrayTy =
        llvm::ArrayType::get(StaticsListPtrTy, 2);
        Elements.clear();
        Elements.push_back(Statics);
        Elements.push_back(llvm::Constant::getNullValue(StaticsListPtrTy));
        Statics = MakeGlobal(StaticsListArrayTy, Elements, ".objc_statics_ptr");
        Statics = llvm::ConstantExpr::getBitCast(Statics, PtrTy);
    }
            
    // Array of classes, categories, and constant objects
    llvm::ArrayType *ClassListTy = llvm::ArrayType::get(PtrToInt8Ty,
                                                        Classes.size() + Categories.size()  + 2);
    llvm::StructType *SymTabTy = llvm::StructType::get(LongTy, PtrToInt8Ty,
                                                       llvm::Type::getInt16Ty(VMContext),
                                                       llvm::Type::getInt16Ty(VMContext),
                                                       ClassListTy, NULL);
    
    Elements.clear();
    // Pointer to an array of selectors used in this module.
    std::vector<llvm::Constant*> Selectors;
    std::vector<llvm::GlobalAlias*> SelectorAliases;
    for (SelectorMap::iterator iter = SelectorTable.begin(),
         iterEnd = SelectorTable.end(); iter != iterEnd ; ++iter) {
        
        std::string SelNameStr = iter->first.getAsString();
        llvm::Constant *SelName = ExportUniqueString(SelNameStr, ".objc_sel_name");
        Selectors.push_back(SelName);
        SelectorAliases.push_back(iter->second);
    }
    
    unsigned SelectorCount = Selectors.size();
    // NULL-terminate the selector list. Cocotron needs this in the current implementation    
    Selectors.push_back(NULLPtr);
    Elements.clear();
        
    
    // Number of static selectors
    Elements.push_back(llvm::ConstantInt::get(LongTy, SelectorCount));
    
    
    llvm::Constant *SelectorList = MakeGlobalArray(PtrToInt8Ty, Selectors,
                                                   ".objc_selector_list");
    
    Elements.push_back(llvm::ConstantExpr::getBitCast(SelectorList,
                                                      PtrToInt8Ty));
    
    // Now that all of the static selectors exist, create pointers to them.
    for (unsigned int i=0 ; i<SelectorCount ; i++) {
        llvm::Constant *Idxs[] = {Zeros[0],
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), i), Zeros[0]};
        
        // FIXME: We're generating redundant loads and stores here!
        llvm::Constant *SelPtr = llvm::ConstantExpr::getGetElementPtr(SelectorList,
                                                                      Idxs, 2);
        
        // If selectors are defined as an opaque type, cast the pointer to this
        // type.
        SelPtr = llvm::ConstantExpr::getBitCast(SelPtr, PtrToInt8Ty);
        
        SelectorAliases[i]->replaceAllUsesWith(SelPtr);
        
        SelectorAliases[i]->eraseFromParent();
    }
    
    // Number of classes defined.
    Elements.push_back(llvm::ConstantInt::get(llvm::Type::getInt16Ty(VMContext),
                                              Classes.size()));
    
    // Number of categories defined
    Elements.push_back(llvm::ConstantInt::get(llvm::Type::getInt16Ty(VMContext),
                                              Categories.size()));
    // Create an array of classes, then categories, then static object instances
    Classes.insert(Classes.end(), Categories.begin(), Categories.end());
    
    //  NULL-terminated list of static object instances (mainly constant strings)
    Classes.push_back(Statics);
    Classes.push_back(NULLPtr);
    
    llvm::Constant *ClassList = llvm::ConstantArray::get(ClassListTy, Classes);
    Elements.push_back(ClassList);
    // Construct the symbol table
    
    llvm::Constant *SymTab= MakeGlobal(SymTabTy, Elements);
    
    
    // The symbol table is contained in a module which has some version-checking
    // constants
    llvm::StructType * ModuleTy = llvm::StructType::get(LongTy, LongTy,
                                                        PtrToInt8Ty, llvm::PointerType::getUnqual(SymTabTy), NULL);
    Elements.clear();
    // Runtime version, used for ABI compatibility checking.
    Elements.push_back(llvm::ConstantInt::get(LongTy, RuntimeVersion));
    // sizeof(ModuleTy)
    llvm::TargetData td(&TheModule);
    Elements.push_back(
                       llvm::ConstantInt::get(LongTy,
                                              td.getTypeSizeInBits(ModuleTy) /
                                              CGM.getContext().getCharWidth()));
    
    
    // The path to the source file where this module was declared
    SourceManager &SM = CGM.getContext().getSourceManager();
    const FileEntry *mainFile = SM.getFileEntryForID(SM.getMainFileID());
    std::string path =
    std::string(mainFile->getDir()->getName()) + '/' + mainFile->getName();
    Elements.push_back(MakeConstantString(path, ".objc_source_file_name"));
    
    Elements.push_back(SymTab);
    llvm::Value *Module = MakeGlobal(ModuleTy, Elements);
    
    // Create the load function calling the runtime entry point with the module
    // structure
    llvm::Function * LoadFunction = llvm::Function::Create(
                                                           llvm::FunctionType::get(llvm::Type::getVoidTy(VMContext), false),
                                                           llvm::GlobalValue::InternalLinkage, ".objc_load_function",
                                                           &TheModule);
    llvm::BasicBlock *EntryBB =
    llvm::BasicBlock::Create(VMContext, "entry", LoadFunction);
    CGBuilderTy Builder(VMContext);
    Builder.SetInsertPoint(EntryBB);
    
    std::vector<const llvm::Type*> Params(1,
                                          llvm::PointerType::getUnqual(ModuleTy));
    llvm::Value *Register = CGM.CreateRuntimeFunction(llvm::FunctionType::get(
                                                                              llvm::Type::getVoidTy(VMContext), Params, true), "__objc_exec_class");
    Builder.CreateCall(Register, Module);
    Builder.CreateRetVoid();
    
    return LoadFunction;
}

llvm::Function *CGObjCCocotron::GenerateMethod(const ObjCMethodDecl *OMD,
                                          const ObjCContainerDecl *CD) {
    const ObjCCategoryImplDecl *OCD =
    dyn_cast<ObjCCategoryImplDecl>(OMD->getDeclContext());
    llvm::StringRef CategoryName = OCD ? OCD->getName() : "";
    llvm::StringRef ClassName = CD->getName();
    Selector MethodName = OMD->getSelector();
    bool isClassMethod = !OMD->isInstanceMethod();
    
    CodeGenTypes &Types = CGM.getTypes();
    const llvm::FunctionType *MethodTy =
    Types.GetFunctionType(Types.getFunctionInfo(OMD), OMD->isVariadic());
    std::string FunctionName = SymbolNameForMethod(ClassName, CategoryName,
                                                   MethodName, isClassMethod);
    
    llvm::Function *Method
    = llvm::Function::Create(MethodTy,
                             llvm::GlobalValue::InternalLinkage,
                             FunctionName,
                             &TheModule);
    return Method;
}

llvm::Function *CGObjCCocotron::GetPropertyGetFunction() {
    return GetPropertyFn;
}

llvm::Function *CGObjCCocotron::GetPropertySetFunction() {
    return SetPropertyFn;
}

llvm::Function *CGObjCCocotron::GetGetStructFunction() {
    return GetStructPropertyFn;
}
llvm::Function *CGObjCCocotron::GetSetStructFunction() {
    return SetStructPropertyFn;
}

llvm::Constant *CGObjCCocotron::EnumerationMutationFunction() {
    return EnumerationMutationFn;
}

void CGObjCCocotron::EmitSynchronizedStmt(CodeGenFunction &CGF,
                                     const ObjCAtSynchronizedStmt &S) {
    EmitAtSynchronizedStmt(CGF, S, SyncEnterFn, SyncExitFn);
}


void CGObjCCocotron::EmitTryStmt(CodeGenFunction &CGF,
                            const ObjCAtTryStmt &S) {
    // Unlike the Apple non-fragile runtimes, which also uses
    // unwind-based zero cost exceptions, the GNU Objective C runtime's
    // EH support isn't a veneer over C++ EH.  Instead, exception
    // objects are created by __objc_exception_throw and destroyed by
    // the personality function; this avoids the need for bracketing
    // catch handlers with calls to __blah_begin_catch/__blah_end_catch
    // (or even _Unwind_DeleteException), but probably doesn't
    // interoperate very well with foreign exceptions.
    //
    // In Objective-C++ mode, we actually emit something equivalent to the C++
    // exception handler. 
    EmitTryCatchStmt(CGF, S, EnterCatchFn, ExitCatchFn, ExceptionReThrowFn);
    return ;
}

void CGObjCCocotron::EmitThrowStmt(CodeGenFunction &CGF,
                              const ObjCAtThrowStmt &S) {
    llvm::Value *ExceptionAsObject;
    
    if (const Expr *ThrowExpr = S.getThrowExpr()) {
        llvm::Value *Exception = CGF.EmitScalarExpr(ThrowExpr);
        ExceptionAsObject = Exception;
    } else {
        assert((!CGF.ObjCEHValueStack.empty() && CGF.ObjCEHValueStack.back()) &&
               "Unexpected rethrow outside @catch block.");
        ExceptionAsObject = CGF.ObjCEHValueStack.back();
    }
    ExceptionAsObject =
    CGF.Builder.CreateBitCast(ExceptionAsObject, IdTy, "tmp");
    
    // Note: This may have to be an invoke, if we want to support constructs like:
    // @try {
    //  @throw(obj);
    // }
    // @catch(id) ...
    //
    // This is effectively turning @throw into an incredibly-expensive goto, but
    // it may happen as a result of inlining followed by missed optimizations, or
    // as a result of stupidity.
    llvm::BasicBlock *UnwindBB = CGF.getInvokeDest();
    if (!UnwindBB) {
        CGF.Builder.CreateCall(ExceptionThrowFn, ExceptionAsObject);
        CGF.Builder.CreateUnreachable();
    } else {
        CGF.Builder.CreateInvoke(ExceptionThrowFn, UnwindBB, UnwindBB, &ExceptionAsObject,
                                 &ExceptionAsObject+1);
    }
    // Clear the insertion point to indicate we are in unreachable code.
    CGF.Builder.ClearInsertionPoint();
}

llvm::Value * CGObjCCocotron::EmitObjCWeakRead(CodeGenFunction &CGF,
                                          llvm::Value *AddrWeakObj) {
    CGBuilderTy B = CGF.Builder;
    AddrWeakObj = EnforceType(B, AddrWeakObj, IdTy);
    return B.CreateCall(WeakReadFn, AddrWeakObj);
}

void CGObjCCocotron::EmitObjCWeakAssign(CodeGenFunction &CGF,
                                   llvm::Value *src, llvm::Value *dst) {
    CGBuilderTy B = CGF.Builder;
    src = EnforceType(B, src, IdTy);
    dst = EnforceType(B, dst, PtrToIdTy);
    B.CreateCall2(WeakAssignFn, src, dst);
}

void CGObjCCocotron::EmitObjCGlobalAssign(CodeGenFunction &CGF,
                                     llvm::Value *src, llvm::Value *dst,
                                     bool threadlocal) {
    CGBuilderTy B = CGF.Builder;
    src = EnforceType(B, src, IdTy);
    dst = EnforceType(B, dst, PtrToIdTy);
    if (!threadlocal)
        B.CreateCall2(GlobalAssignFn, src, dst);
    else
        // FIXME. Add threadloca assign API
        assert(false && "EmitObjCGlobalAssign - Threal Local API NYI");
}

void CGObjCCocotron::EmitObjCIvarAssign(CodeGenFunction &CGF,
                                   llvm::Value *src, llvm::Value *dst,
                                   llvm::Value *ivarOffset) {
    CGBuilderTy B = CGF.Builder;
    src = EnforceType(B, src, IdTy);
    dst = EnforceType(B, dst, PtrToIdTy);
    B.CreateCall3(IvarAssignFn, src, dst, ivarOffset);
}

void CGObjCCocotron::EmitObjCStrongCastAssign(CodeGenFunction &CGF,
                                         llvm::Value *src, llvm::Value *dst) {
    CGBuilderTy B = CGF.Builder;
    src = EnforceType(B, src, IdTy);
    dst = EnforceType(B, dst, PtrToIdTy);
    B.CreateCall2(StrongCastAssignFn, src, dst);
}

void CGObjCCocotron::EmitGCMemmoveCollectable(CodeGenFunction &CGF,
                                         llvm::Value *DestPtr,
                                         llvm::Value *SrcPtr,
                                         llvm::Value *Size) {
    CGBuilderTy B = CGF.Builder;
    DestPtr = EnforceType(B, DestPtr, IdTy);
    SrcPtr = EnforceType(B, SrcPtr, PtrToIdTy);
    
    B.CreateCall3(MemMoveFn, DestPtr, SrcPtr, Size);
}

llvm::GlobalVariable *CGObjCCocotron::ObjCIvarOffsetVariable(
                                                        const ObjCInterfaceDecl *ID,
                                                        const ObjCIvarDecl *Ivar) {
    const std::string Name = "__objc_ivar_offset_" + ID->getNameAsString()
    + '.' + Ivar->getNameAsString();
    // Emit the variable and initialize it with what we think the correct value
    // is.  This allows code compiled with non-fragile ivars to work correctly
    // when linked against code which isn't (most of the time).
    llvm::GlobalVariable *IvarOffsetPointer = TheModule.getNamedGlobal(Name);
    if (!IvarOffsetPointer) {
        // This will cause a run-time crash if we accidentally use it.  A value of
        // 0 would seem more sensible, but will silently overwrite the isa pointer
        // causing a great deal of confusion.
        uint64_t Offset = -1;
        // We can't call ComputeIvarBaseOffset() here if we have the
        // implementation, because it will create an invalid ASTRecordLayout object
        // that we are then stuck with forever, so we only initialize the ivar
        // offset variable with a guess if we only have the interface.  The
        // initializer will be reset later anyway, when we are generating the class
        // description.
        if (!CGM.getContext().getObjCImplementation(
                                                    const_cast<ObjCInterfaceDecl *>(ID)))
            Offset = ComputeIvarBaseOffset(CGM, ID, Ivar);
        
        llvm::ConstantInt *OffsetGuess =
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), Offset, "ivar");
        // Don't emit the guess in non-PIC code because the linker will not be able
        // to replace it with the real version for a library.  In non-PIC code you
        // must compile with the fragile ABI if you want to use ivars from a
        // GCC-compiled class.
        if (CGM.getLangOptions().PICLevel) {
            llvm::GlobalVariable *IvarOffsetGV = new llvm::GlobalVariable(TheModule,
                                                                          llvm::Type::getInt32Ty(VMContext), false,
                                                                          llvm::GlobalValue::PrivateLinkage, OffsetGuess, Name+".guess");
            IvarOffsetPointer = new llvm::GlobalVariable(TheModule,
                                                         IvarOffsetGV->getType(), false, llvm::GlobalValue::LinkOnceAnyLinkage,
                                                         IvarOffsetGV, Name);
        } else {
            IvarOffsetPointer = new llvm::GlobalVariable(TheModule,
                                                         llvm::Type::getInt32PtrTy(VMContext), false,
                                                         llvm::GlobalValue::ExternalLinkage, 0, Name);
        }
    }
    return IvarOffsetPointer;
}

LValue CGObjCCocotron::EmitObjCValueForIvar(CodeGenFunction &CGF,
                                       QualType ObjectTy,
                                       llvm::Value *BaseValue,
                                       const ObjCIvarDecl *Ivar,
                                       unsigned CVRQualifiers) {
    const ObjCInterfaceDecl *ID =
    ObjectTy->getAs<ObjCObjectType>()->getInterface();
    return EmitValueForIvarAtOffset(CGF, ID, BaseValue, Ivar, CVRQualifiers,
                                    EmitIvarOffset(CGF, ID, Ivar));
}

static const ObjCInterfaceDecl *FindIvarInterface(ASTContext &Context,
                                                  const ObjCInterfaceDecl *OID,
                                                  const ObjCIvarDecl *OIVD) {
    llvm::SmallVector<ObjCIvarDecl*, 16> Ivars;
    Context.ShallowCollectObjCIvars(OID, Ivars);
    for (unsigned k = 0, e = Ivars.size(); k != e; ++k) {
        if (OIVD == Ivars[k])
            return OID;
    }
    
    // Otherwise check in the super class.
    if (const ObjCInterfaceDecl *Super = OID->getSuperClass())
        return FindIvarInterface(Context, Super, OIVD);
    
    return 0;
}

llvm::Value *CGObjCCocotron::EmitIvarOffset(CodeGenFunction &CGF,
                                       const ObjCInterfaceDecl *Interface,
                                       const ObjCIvarDecl *Ivar) {
    if (CGM.getLangOptions().ObjCNonFragileABI && !CGM.getLangOptions().ObjCNonFragileABI2) {
        Interface = FindIvarInterface(CGM.getContext(), Interface, Ivar);
        return CGF.Builder.CreateZExtOrBitCast(
                                               CGF.Builder.CreateLoad(CGF.Builder.CreateLoad(
                                                                                             ObjCIvarOffsetVariable(Interface, Ivar), false, "ivar")),
                                               PtrDiffTy);
    }
    uint64_t Offset = ComputeIvarBaseOffset(CGF.CGM, Interface, Ivar);
    return llvm::ConstantInt::get(PtrDiffTy, Offset, "ivar");
}

CGObjCRuntime *
clang::CodeGen::CreateCocotronObjCRuntime(CodeGenModule &CGM) {
    return new CGObjCCocotron(CGM);
}

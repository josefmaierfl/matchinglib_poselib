
// This is a generated file. Do not edit!

#ifndef POSELIB_COMPILER_DETECTION_H
#define POSELIB_COMPILER_DETECTION_H

#ifdef __cplusplus
# define POSELIB_COMPILER_IS_Comeau 0
# define POSELIB_COMPILER_IS_Intel 0
# define POSELIB_COMPILER_IS_PathScale 0
# define POSELIB_COMPILER_IS_Embarcadero 0
# define POSELIB_COMPILER_IS_Borland 0
# define POSELIB_COMPILER_IS_Watcom 0
# define POSELIB_COMPILER_IS_OpenWatcom 0
# define POSELIB_COMPILER_IS_SunPro 0
# define POSELIB_COMPILER_IS_HP 0
# define POSELIB_COMPILER_IS_Compaq 0
# define POSELIB_COMPILER_IS_zOS 0
# define POSELIB_COMPILER_IS_XL 0
# define POSELIB_COMPILER_IS_VisualAge 0
# define POSELIB_COMPILER_IS_PGI 0
# define POSELIB_COMPILER_IS_Cray 0
# define POSELIB_COMPILER_IS_TI 0
# define POSELIB_COMPILER_IS_Fujitsu 0
# define POSELIB_COMPILER_IS_SCO 0
# define POSELIB_COMPILER_IS_AppleClang 0
# define POSELIB_COMPILER_IS_Clang 0
# define POSELIB_COMPILER_IS_GNU 0
# define POSELIB_COMPILER_IS_MSVC 0
# define POSELIB_COMPILER_IS_ADSP 0
# define POSELIB_COMPILER_IS_IAR 0
# define POSELIB_COMPILER_IS_ARMCC 0
# define POSELIB_COMPILER_IS_MIPSpro 0

#if defined(__COMO__)
# undef POSELIB_COMPILER_IS_Comeau
# define POSELIB_COMPILER_IS_Comeau 1

#elif defined(__INTEL_COMPILER) || defined(__ICC)
# undef POSELIB_COMPILER_IS_Intel
# define POSELIB_COMPILER_IS_Intel 1

#elif defined(__PATHCC__)
# undef POSELIB_COMPILER_IS_PathScale
# define POSELIB_COMPILER_IS_PathScale 1

#elif defined(__BORLANDC__) && defined(__CODEGEARC_VERSION__)
# undef POSELIB_COMPILER_IS_Embarcadero
# define POSELIB_COMPILER_IS_Embarcadero 1

#elif defined(__BORLANDC__)
# undef POSELIB_COMPILER_IS_Borland
# define POSELIB_COMPILER_IS_Borland 1

#elif defined(__WATCOMC__) && __WATCOMC__ < 1200
# undef POSELIB_COMPILER_IS_Watcom
# define POSELIB_COMPILER_IS_Watcom 1

#elif defined(__WATCOMC__)
# undef POSELIB_COMPILER_IS_OpenWatcom
# define POSELIB_COMPILER_IS_OpenWatcom 1

#elif defined(__SUNPRO_CC)
# undef POSELIB_COMPILER_IS_SunPro
# define POSELIB_COMPILER_IS_SunPro 1

#elif defined(__HP_aCC)
# undef POSELIB_COMPILER_IS_HP
# define POSELIB_COMPILER_IS_HP 1

#elif defined(__DECCXX)
# undef POSELIB_COMPILER_IS_Compaq
# define POSELIB_COMPILER_IS_Compaq 1

#elif defined(__IBMCPP__) && defined(__COMPILER_VER__)
# undef POSELIB_COMPILER_IS_zOS
# define POSELIB_COMPILER_IS_zOS 1

#elif defined(__IBMCPP__) && !defined(__COMPILER_VER__) && __IBMCPP__ >= 800
# undef POSELIB_COMPILER_IS_XL
# define POSELIB_COMPILER_IS_XL 1

#elif defined(__IBMCPP__) && !defined(__COMPILER_VER__) && __IBMCPP__ < 800
# undef POSELIB_COMPILER_IS_VisualAge
# define POSELIB_COMPILER_IS_VisualAge 1

#elif defined(__PGI)
# undef POSELIB_COMPILER_IS_PGI
# define POSELIB_COMPILER_IS_PGI 1

#elif defined(_CRAYC)
# undef POSELIB_COMPILER_IS_Cray
# define POSELIB_COMPILER_IS_Cray 1

#elif defined(__TI_COMPILER_VERSION__)
# undef POSELIB_COMPILER_IS_TI
# define POSELIB_COMPILER_IS_TI 1

#elif defined(__FUJITSU) || defined(__FCC_VERSION) || defined(__fcc_version)
# undef POSELIB_COMPILER_IS_Fujitsu
# define POSELIB_COMPILER_IS_Fujitsu 1

#elif defined(__SCO_VERSION__)
# undef POSELIB_COMPILER_IS_SCO
# define POSELIB_COMPILER_IS_SCO 1

#elif defined(__clang__) && defined(__apple_build_version__)
# undef POSELIB_COMPILER_IS_AppleClang
# define POSELIB_COMPILER_IS_AppleClang 1

#elif defined(__clang__)
# undef POSELIB_COMPILER_IS_Clang
# define POSELIB_COMPILER_IS_Clang 1

#elif defined(__GNUC__)
# undef POSELIB_COMPILER_IS_GNU
# define POSELIB_COMPILER_IS_GNU 1

#elif defined(_MSC_VER)
# undef POSELIB_COMPILER_IS_MSVC
# define POSELIB_COMPILER_IS_MSVC 1

#elif defined(__VISUALDSPVERSION__) || defined(__ADSPBLACKFIN__) || defined(__ADSPTS__) || defined(__ADSP21000__)
# undef POSELIB_COMPILER_IS_ADSP
# define POSELIB_COMPILER_IS_ADSP 1

#elif defined(__IAR_SYSTEMS_ICC__ ) || defined(__IAR_SYSTEMS_ICC)
# undef POSELIB_COMPILER_IS_IAR
# define POSELIB_COMPILER_IS_IAR 1

#elif defined(__ARMCC_VERSION)
# undef POSELIB_COMPILER_IS_ARMCC
# define POSELIB_COMPILER_IS_ARMCC 1

#elif defined(_SGI_COMPILER_VERSION) || defined(_COMPILER_VERSION)
# undef POSELIB_COMPILER_IS_MIPSpro
# define POSELIB_COMPILER_IS_MIPSpro 1


#endif

#  if POSELIB_COMPILER_IS_AppleClang

#    if !(((__clang_major__ * 100) + __clang_minor__) >= 400)
#      error Unsupported compiler version
#    endif

# define POSELIB_COMPILER_VERSION_MAJOR (__clang_major__)
# define POSELIB_COMPILER_VERSION_MINOR (__clang_minor__)
# define POSELIB_COMPILER_VERSION_PATCH (__clang_patchlevel__)
# if defined(_MSC_VER)
   /* _MSC_VER = VVRR */
#  define POSELIB_SIMULATE_VERSION_MAJOR (_MSC_VER / 100)
#  define POSELIB_SIMULATE_VERSION_MINOR (_MSC_VER % 100)
# endif
# define POSELIB_COMPILER_VERSION_TWEAK (__apple_build_version__)

#    if ((__clang_major__ * 100) + __clang_minor__) >= 400 && __has_feature(cxx_alignas)
#      define POSELIB_COMPILER_CXX_ALIGNAS 1
#    else
#      define POSELIB_COMPILER_CXX_ALIGNAS 0
#    endif

#    if ((__clang_major__ * 100) + __clang_minor__) >= 400 && __has_feature(cxx_alignas)
#      define POSELIB_COMPILER_CXX_ALIGNOF 1
#    else
#      define POSELIB_COMPILER_CXX_ALIGNOF 0
#    endif

#    if ((__clang_major__ * 100) + __clang_minor__) >= 400 && __has_feature(cxx_constexpr)
#      define POSELIB_COMPILER_CXX_CONSTEXPR 1
#    else
#      define POSELIB_COMPILER_CXX_CONSTEXPR 0
#    endif

#    if ((__clang_major__ * 100) + __clang_minor__) >= 400 && __has_feature(cxx_override_control)
#      define POSELIB_COMPILER_CXX_FINAL 1
#    else
#      define POSELIB_COMPILER_CXX_FINAL 0
#    endif

#    if ((__clang_major__ * 100) + __clang_minor__) >= 400 && __has_feature(cxx_noexcept)
#      define POSELIB_COMPILER_CXX_NOEXCEPT 1
#    else
#      define POSELIB_COMPILER_CXX_NOEXCEPT 0
#    endif

#    if ((__clang_major__ * 100) + __clang_minor__) >= 400 && __has_feature(cxx_nullptr)
#      define POSELIB_COMPILER_CXX_NULLPTR 1
#    else
#      define POSELIB_COMPILER_CXX_NULLPTR 0
#    endif

#    if ((__clang_major__ * 100) + __clang_minor__) >= 400 && __cplusplus >= 201103L
#      define POSELIB_COMPILER_CXX_SIZEOF_MEMBER 1
#    else
#      define POSELIB_COMPILER_CXX_SIZEOF_MEMBER 0
#    endif

#    if ((__clang_major__ * 100) + __clang_minor__) >= 400 && __has_feature(cxx_thread_local)
#      define POSELIB_COMPILER_CXX_THREAD_LOCAL 1
#    else
#      define POSELIB_COMPILER_CXX_THREAD_LOCAL 0
#    endif

#  elif POSELIB_COMPILER_IS_Clang

#    if !(((__clang_major__ * 100) + __clang_minor__) >= 304)
#      error Unsupported compiler version
#    endif

# define POSELIB_COMPILER_VERSION_MAJOR (__clang_major__)
# define POSELIB_COMPILER_VERSION_MINOR (__clang_minor__)
# define POSELIB_COMPILER_VERSION_PATCH (__clang_patchlevel__)
# if defined(_MSC_VER)
   /* _MSC_VER = VVRR */
#  define POSELIB_SIMULATE_VERSION_MAJOR (_MSC_VER / 100)
#  define POSELIB_SIMULATE_VERSION_MINOR (_MSC_VER % 100)
# endif

#    if ((__clang_major__ * 100) + __clang_minor__) >= 304 && __has_feature(cxx_alignas)
#      define POSELIB_COMPILER_CXX_ALIGNAS 1
#    else
#      define POSELIB_COMPILER_CXX_ALIGNAS 0
#    endif

#    if ((__clang_major__ * 100) + __clang_minor__) >= 304 && __has_feature(cxx_alignas)
#      define POSELIB_COMPILER_CXX_ALIGNOF 1
#    else
#      define POSELIB_COMPILER_CXX_ALIGNOF 0
#    endif

#    if ((__clang_major__ * 100) + __clang_minor__) >= 304 && __has_feature(cxx_constexpr)
#      define POSELIB_COMPILER_CXX_CONSTEXPR 1
#    else
#      define POSELIB_COMPILER_CXX_CONSTEXPR 0
#    endif

#    if ((__clang_major__ * 100) + __clang_minor__) >= 304 && __has_feature(cxx_override_control)
#      define POSELIB_COMPILER_CXX_FINAL 1
#    else
#      define POSELIB_COMPILER_CXX_FINAL 0
#    endif

#    if ((__clang_major__ * 100) + __clang_minor__) >= 304 && __has_feature(cxx_noexcept)
#      define POSELIB_COMPILER_CXX_NOEXCEPT 1
#    else
#      define POSELIB_COMPILER_CXX_NOEXCEPT 0
#    endif

#    if ((__clang_major__ * 100) + __clang_minor__) >= 304 && __has_feature(cxx_nullptr)
#      define POSELIB_COMPILER_CXX_NULLPTR 1
#    else
#      define POSELIB_COMPILER_CXX_NULLPTR 0
#    endif

#    if ((__clang_major__ * 100) + __clang_minor__) >= 304 && __cplusplus >= 201103L
#      define POSELIB_COMPILER_CXX_SIZEOF_MEMBER 1
#    else
#      define POSELIB_COMPILER_CXX_SIZEOF_MEMBER 0
#    endif

#    if ((__clang_major__ * 100) + __clang_minor__) >= 304 && __has_feature(cxx_thread_local)
#      define POSELIB_COMPILER_CXX_THREAD_LOCAL 1
#    else
#      define POSELIB_COMPILER_CXX_THREAD_LOCAL 0
#    endif

#  elif POSELIB_COMPILER_IS_GNU

#    if !((__GNUC__ * 100 + __GNUC_MINOR__) >= 404)
#      error Unsupported compiler version
#    endif

# define POSELIB_COMPILER_VERSION_MAJOR (__GNUC__)
# if defined(__GNUC_MINOR__)
#  define POSELIB_COMPILER_VERSION_MINOR (__GNUC_MINOR__)
# endif
# if defined(__GNUC_PATCHLEVEL__)
#  define POSELIB_COMPILER_VERSION_PATCH (__GNUC_PATCHLEVEL__)
# endif

#    if (__GNUC__ * 100 + __GNUC_MINOR__) >= 408 && __cplusplus >= 201103L
#      define POSELIB_COMPILER_CXX_ALIGNAS 1
#    else
#      define POSELIB_COMPILER_CXX_ALIGNAS 0
#    endif

#    if (__GNUC__ * 100 + __GNUC_MINOR__) >= 408 && __cplusplus >= 201103L
#      define POSELIB_COMPILER_CXX_ALIGNOF 1
#    else
#      define POSELIB_COMPILER_CXX_ALIGNOF 0
#    endif

#    if (__GNUC__ * 100 + __GNUC_MINOR__) >= 406 && (__cplusplus >= 201103L || (defined(__GXX_EXPERIMENTAL_CXX0X__) && __GXX_EXPERIMENTAL_CXX0X__))
#      define POSELIB_COMPILER_CXX_CONSTEXPR 1
#    else
#      define POSELIB_COMPILER_CXX_CONSTEXPR 0
#    endif

#    if (__GNUC__ * 100 + __GNUC_MINOR__) >= 407 && __cplusplus >= 201103L
#      define POSELIB_COMPILER_CXX_FINAL 1
#    else
#      define POSELIB_COMPILER_CXX_FINAL 0
#    endif

#    if (__GNUC__ * 100 + __GNUC_MINOR__) >= 406 && (__cplusplus >= 201103L || (defined(__GXX_EXPERIMENTAL_CXX0X__) && __GXX_EXPERIMENTAL_CXX0X__))
#      define POSELIB_COMPILER_CXX_NOEXCEPT 1
#    else
#      define POSELIB_COMPILER_CXX_NOEXCEPT 0
#    endif

#    if (__GNUC__ * 100 + __GNUC_MINOR__) >= 406 && (__cplusplus >= 201103L || (defined(__GXX_EXPERIMENTAL_CXX0X__) && __GXX_EXPERIMENTAL_CXX0X__))
#      define POSELIB_COMPILER_CXX_NULLPTR 1
#    else
#      define POSELIB_COMPILER_CXX_NULLPTR 0
#    endif

#    if (__GNUC__ * 100 + __GNUC_MINOR__) >= 404 && (__cplusplus >= 201103L || (defined(__GXX_EXPERIMENTAL_CXX0X__) && __GXX_EXPERIMENTAL_CXX0X__))
#      define POSELIB_COMPILER_CXX_SIZEOF_MEMBER 1
#    else
#      define POSELIB_COMPILER_CXX_SIZEOF_MEMBER 0
#    endif

#    if (__GNUC__ * 100 + __GNUC_MINOR__) >= 408 && __cplusplus >= 201103L
#      define POSELIB_COMPILER_CXX_THREAD_LOCAL 1
#    else
#      define POSELIB_COMPILER_CXX_THREAD_LOCAL 0
#    endif

#  elif POSELIB_COMPILER_IS_MSVC

#    if !(_MSC_VER >= 1600)
#      error Unsupported compiler version
#    endif

  /* _MSC_VER = VVRR */
# define POSELIB_COMPILER_VERSION_MAJOR (_MSC_VER / 100)
# define POSELIB_COMPILER_VERSION_MINOR (_MSC_VER % 100)
# if defined(_MSC_FULL_VER)
#  if _MSC_VER >= 1400
    /* _MSC_FULL_VER = VVRRPPPPP */
#   define POSELIB_COMPILER_VERSION_PATCH (_MSC_FULL_VER % 100000)
#  else
    /* _MSC_FULL_VER = VVRRPPPP */
#   define POSELIB_COMPILER_VERSION_PATCH (_MSC_FULL_VER % 10000)
#  endif
# endif
# if defined(_MSC_BUILD)
#  define POSELIB_COMPILER_VERSION_TWEAK (_MSC_BUILD)
# endif

#    if _MSC_VER >= 1900
#      define POSELIB_COMPILER_CXX_ALIGNAS 1
#    else
#      define POSELIB_COMPILER_CXX_ALIGNAS 0
#    endif

#    if _MSC_VER >= 1900
#      define POSELIB_COMPILER_CXX_ALIGNOF 1
#    else
#      define POSELIB_COMPILER_CXX_ALIGNOF 0
#    endif

#    if _MSC_VER >= 1900
#      define POSELIB_COMPILER_CXX_CONSTEXPR 1
#    else
#      define POSELIB_COMPILER_CXX_CONSTEXPR 0
#    endif

#    if _MSC_VER >= 1700
#      define POSELIB_COMPILER_CXX_FINAL 1
#    else
#      define POSELIB_COMPILER_CXX_FINAL 0
#    endif

#    if _MSC_VER >= 1900
#      define POSELIB_COMPILER_CXX_NOEXCEPT 1
#    else
#      define POSELIB_COMPILER_CXX_NOEXCEPT 0
#    endif

#    if _MSC_VER >= 1600
#      define POSELIB_COMPILER_CXX_NULLPTR 1
#    else
#      define POSELIB_COMPILER_CXX_NULLPTR 0
#    endif

#    if _MSC_VER >= 1900
#      define POSELIB_COMPILER_CXX_SIZEOF_MEMBER 1
#    else
#      define POSELIB_COMPILER_CXX_SIZEOF_MEMBER 0
#    endif

#    if _MSC_VER >= 1900
#      define POSELIB_COMPILER_CXX_THREAD_LOCAL 1
#    else
#      define POSELIB_COMPILER_CXX_THREAD_LOCAL 0
#    endif

#  else
#    error Unsupported compiler
#  endif

#  if POSELIB_COMPILER_CXX_ALIGNAS
#    define POSELIB_ALIGNAS(X) alignas(X)
#  elif POSELIB_COMPILER_IS_GNU || POSELIB_COMPILER_IS_Clang || POSELIB_COMPILER_IS_AppleClang
#    define POSELIB_ALIGNAS(X) __attribute__ ((__aligned__(X)))
#  elif POSELIB_COMPILER_IS_MSVC
#    define POSELIB_ALIGNAS(X) __declspec(align(X))
#  else
#    define POSELIB_ALIGNAS(X)
#  endif


#  if POSELIB_COMPILER_CXX_ALIGNOF
#    define POSELIB_ALIGNOF(X) alignof(X)
#  elif POSELIB_COMPILER_IS_GNU || POSELIB_COMPILER_IS_Clang || POSELIB_COMPILER_IS_AppleClang
#    define POSELIB_ALIGNOF(X) __alignof__(X)
#  elif POSELIB_COMPILER_IS_MSVC
#    define POSELIB_ALIGNOF(X) __alignof(X)
#  endif


#  if POSELIB_COMPILER_CXX_CONSTEXPR
#    define POSELIB_CONSTEXPR constexpr
#  else
#    define POSELIB_CONSTEXPR
#  endif


#  if POSELIB_COMPILER_CXX_FINAL
#    define POSELIB_FINAL final
#  else
#    define POSELIB_FINAL
#  endif


#  if POSELIB_COMPILER_CXX_NOEXCEPT
#    define POSELIB_NOEXCEPT noexcept
#    define POSELIB_NOEXCEPT_EXPR(X) noexcept(X)
#  else
#    define POSELIB_NOEXCEPT
#    define POSELIB_NOEXCEPT_EXPR(X)
#  endif


#  if POSELIB_COMPILER_CXX_NULLPTR
#    define POSELIB_NULLPTR nullptr
#  else
#    define POSELIB_NULLPTR 0
#  endif


#  if POSELIB_COMPILER_CXX_THREAD_LOCAL
#    define POSELIB_THREAD_LOCAL thread_local
#  elif POSELIB_COMPILER_IS_GNU || POSELIB_COMPILER_IS_Clang || POSELIB_COMPILER_IS_AppleClang
#    define POSELIB_THREAD_LOCAL __thread
#  elif POSELIB_COMPILER_IS_MSVC
#    define POSELIB_THREAD_LOCAL __declspec(thread)
#  else
// POSELIB_THREAD_LOCAL not defined for this configuration.
#  endif

#endif

#endif

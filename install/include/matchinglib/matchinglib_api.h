
#ifndef MATCHINGLIB_API_H
#define MATCHINGLIB_API_H

#ifdef MATCHINGLIB_STATIC_DEFINE
#  define MATCHINGLIB_API
#  define MATCHINGLIB_NO_EXPORT
#else
#  ifndef MATCHINGLIB_API
#    ifdef matchinglib_EXPORTS
        /* We are building this library */
#      define MATCHINGLIB_API __declspec(dllexport)
#    else
        /* We are using this library */
#      define MATCHINGLIB_API __declspec(dllimport)
#    endif
#  endif

#  ifndef MATCHINGLIB_NO_EXPORT
#    define MATCHINGLIB_NO_EXPORT 
#  endif
#endif

#ifndef MATCHINGLIB_DEPRECATED
#  define MATCHINGLIB_DEPRECATED __declspec(deprecated)
#endif

#ifndef MATCHINGLIB_DEPRECATED_EXPORT
#  define MATCHINGLIB_DEPRECATED_EXPORT MATCHINGLIB_API MATCHINGLIB_DEPRECATED
#endif

#ifndef MATCHINGLIB_DEPRECATED_NO_EXPORT
#  define MATCHINGLIB_DEPRECATED_NO_EXPORT MATCHINGLIB_NO_EXPORT MATCHINGLIB_DEPRECATED
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define MATCHINGLIB_NO_DEPRECATED
#endif

#endif

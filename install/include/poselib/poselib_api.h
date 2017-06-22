
#ifndef POSELIB_API_H
#define POSELIB_API_H

#ifdef POSELIB_STATIC_DEFINE
#  define POSELIB_API
#  define POSELIB_NO_EXPORT
#else
#  ifndef POSELIB_API
#    ifdef poselib_EXPORTS
        /* We are building this library */
#      define POSELIB_API __declspec(dllexport)
#    else
        /* We are using this library */
#      define POSELIB_API __declspec(dllimport)
#    endif
#  endif

#  ifndef POSELIB_NO_EXPORT
#    define POSELIB_NO_EXPORT 
#  endif
#endif

#ifndef POSELIB_DEPRECATED
#  define POSELIB_DEPRECATED __declspec(deprecated)
#endif

#ifndef POSELIB_DEPRECATED_EXPORT
#  define POSELIB_DEPRECATED_EXPORT POSELIB_API POSELIB_DEPRECATED
#endif

#ifndef POSELIB_DEPRECATED_NO_EXPORT
#  define POSELIB_DEPRECATED_NO_EXPORT POSELIB_NO_EXPORT POSELIB_DEPRECATED
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define POSELIB_NO_DEPRECATED
#endif

#endif

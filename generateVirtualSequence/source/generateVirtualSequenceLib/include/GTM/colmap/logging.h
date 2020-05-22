// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#ifndef COLMAP_SRC_UTIL_LOGGING_H_
#define COLMAP_SRC_UTIL_LOGGING_H_

#include <iostream>
#include <exception>
#include <utility>

//#include <glog/logging.h>

#include "GTM/colmap/string.h"
#include <sstream>
#include <string>

// Option checker macros. In contrast to glog, this function does not abort the
// program, but simply returns false on failure.
#define CHECK_OPTION_IMPL(expr) \
  __CheckOptionImpl(__FILE__, __LINE__, (expr), #expr)
#define CHECK_OPTION(expr)                                     \
  if (!__CheckOptionImpl(__FILE__, __LINE__, (expr), #expr)) { \
    return false;                                              \
  }
#define CHECK_OPTION_OP(name, op, val1, val2)                              \
  if (!__CheckOptionOpImpl(__FILE__, __LINE__, (val1 op val2), val1, val2, \
                           #val1, #val2, #op)) {                           \
    return false;                                                          \
  }
//class colmapException;
#define CHECK_OPTION_OP_RAISE(name, op, val1, val2)                        \
  if (!__CheckOptionOpImpl(__FILE__, __LINE__, (val1 op val2), val1, val2, \
                           #val1, #val2, #op)) {                           \
    throw colmapException("Error during preparation of Colmap data");      \
  }
#define CHECK_OPTION_EQ(val1, val2) CHECK_OPTION_OP(_EQ, ==, val1, val2)
#define CHECK_OPTION_NE(val1, val2) CHECK_OPTION_OP(_NE, !=, val1, val2)
#define CHECK_OPTION_LE(val1, val2) CHECK_OPTION_OP(_LE, <=, val1, val2)
#define CHECK_OPTION_LT(val1, val2) CHECK_OPTION_OP(_LT, <, val1, val2)
#define CHECK_OPTION_GE(val1, val2) CHECK_OPTION_OP(_GE, >=, val1, val2)
#define CHECK_OPTION_GT(val1, val2) CHECK_OPTION_OP(_GT, >, val1, val2)

#define CHECK_GT(val1, val2) CHECK_OPTION_OP_RAISE(_GT, >, val1, val2)
#define CHECK_GE(val1, val2) CHECK_OPTION_OP_RAISE(_GE, >=, val1, val2)
#define CHECK_LE(val1, val2) CHECK_OPTION_OP_RAISE(_LE, <=, val1, val2)
#define CHECK_EQ(val1, val2) CHECK_OPTION_OP_RAISE(_EQ, ==, val1, val2)
#define CHECK_NE(val1, val2) CHECK_OPTION_OP_RAISE(_NE, !=, val1, val2)
#define CHECK_LT(val1, val2) CHECK_OPTION_OP_RAISE(_LT, <, val1, val2)
//#define CHECK_NOTNULL(val1) CHECK_OPTION_OP_RAISE(_NE, !=, val1, nullptr)

#define FATAL 1
#define CHECK(condition)                                                                                    \
    if (!__CheckOptionImpl(__FILE__, __LINE__, (condition), #condition)) {                                  \
        throw colmapException("Error during preparation of Colmap data. Check failed: " #condition " ");    \
    }
#define LOG(severity, message)                                              \
    if (severity == FATAL) {                                                \
        __CheckOptionImpl(__FILE__, __LINE__, false, message);              \
        throw colmapException("Error during preparation of Colmap data");   \
    }

namespace colmap {

// Initialize glog at the beginning of the program.
//void InitializeGlog(char** argv);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

class colmapException: public std::exception{
    std::string _msg;
public:
    explicit colmapException(std::string  msg) : _msg(std::move(msg)){}
    const char* what() const noexcept override{
        return _msg.c_str();
    }
};

const char* __GetConstFileBaseName(const char* file);

bool __CheckOptionImpl(const char* file, const int line, const bool result,
                       const char* expr_str);

template <typename T1, typename T2>
bool __CheckOptionOpImpl(const char* file, const int line, const bool result,
                         const T1& val1, const T2& val2, const char* val1_str,
                         const char* val2_str, const char* op_str) {
  if (result) {
    return true;
  } else {
    std::cerr << StringPrintf("[%s:%d] Check failed: %s %s %s (%s vs. %s)",
                              __GetConstFileBaseName(file), line, val1_str,
                              op_str, val2_str, std::to_string(val1).c_str(),
                              std::to_string(val2).c_str())
              << std::endl;
    return false;
  }
}

}  // namespace colmap

#endif  // COLMAP_SRC_UTIL_LOGGING_H_

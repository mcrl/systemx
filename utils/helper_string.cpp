/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "helper_string.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include <fstream>
#include <string>
#include <vector>

#ifndef STRCASECMP
#define STRCASECMP strcasecmp
#endif
#ifndef STRNCASECMP
#define STRNCASECMP strncasecmp
#endif
#ifndef STRCPY
#define STRCPY(sFilePath, nLength, sPath) strcpy(sFilePath, sPath)
#endif

#ifndef FOPEN
#define FOPEN(fHandle, filename, mode) (fHandle = fopen(filename, mode))
#endif
#ifndef FOPEN_FAIL
#define FOPEN_FAIL(result) (result == NULL)
#endif
#ifndef SSCANF
#define SSCANF sscanf
#endif
#ifndef SPRINTF
#define SPRINTF sprintf
#endif

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

using namespace std;

namespace SYSTEMX {
namespace utils {
int stringRemoveDelimiter(char delimiter, const char *string) {
  int string_start = 0;

  while (string[string_start] == delimiter) {
    string_start++;
  }

  if (string_start >= static_cast<int>(strlen(string) - 1)) {
    return 0;
  }

  return string_start;
}

vector<string> stringSplit(string s, string delimiter) {
  size_t pos_start = 0, pos_end, delim_len = delimiter.length();
  string token;
  vector<string> res;

  while ((pos_end = s.find(delimiter, pos_start)) != string::npos) {
    token = s.substr(pos_start, pos_end - pos_start);
    pos_start = pos_end + delim_len;
    res.push_back(token);
  }

  res.push_back(s.substr(pos_start));
  return res;
}

int getFileExtension(char *filename, char **extension) {
  int string_length = static_cast<int>(strlen(filename));

  while (filename[string_length--] != '.') {
    if (string_length == 0) break;
  }

  if (string_length > 0) string_length += 2;

  if (string_length == 0)
    *extension = NULL;
  else
    *extension = &filename[string_length];

  return string_length;
}

bool checkCmdLineFlag(const int argc, const char **argv,
                             const char *string_ref) {
  bool bFound = false;

  if (argc >= 1) {
    for (int i = 1; i < argc; i++) {
      int string_start = stringRemoveDelimiter('-', argv[i]);
      const char *string_argv = &argv[i][string_start];

      const char *equal_pos = strchr(string_argv, '=');
      int argv_length = static_cast<int>(
        equal_pos == 0 ? strlen(string_argv) : equal_pos - string_argv);

      int length = static_cast<int>(strlen(string_ref));

      if (length == argv_length &&
          !STRNCASECMP(string_argv, string_ref, length)) {
        bFound = true;
        continue;
      }
    }
  }

  return bFound;
}

template <class T>
bool getCmdLineArgumentValue(const int argc, const char **argv,
                                    const char *string_ref, T *value) {
  bool bFound = false;

  if (argc >= 1) {
    for (int i = 1; i < argc; i++) {
      int string_start = stringRemoveDelimiter('-', argv[i]);
      const char *string_argv = &argv[i][string_start];
      int length = static_cast<int>(strlen(string_ref));

      if (!STRNCASECMP(string_argv, string_ref, length)) {
        if (length + 1 <= static_cast<int>(strlen(string_argv))) {
          int auto_inc = (string_argv[length] == '=') ? 1 : 0;
          *value = (T)atoi(&string_argv[length + auto_inc]);
        }

        bFound = true;
        i = argc;
      }
    }
  }

  return bFound;
}

int getCmdLineArgumentInt(const int argc, const char **argv,
                                 const char *string_ref) {
  bool bFound = false;
  int value = -1;

  if (argc >= 1) {
    for (int i = 1; i < argc; i++) {
      int string_start = stringRemoveDelimiter('-', argv[i]);
      const char *string_argv = &argv[i][string_start];
      int length = static_cast<int>(strlen(string_ref));

      if (!STRNCASECMP(string_argv, string_ref, length)) {
        if (length + 1 <= static_cast<int>(strlen(string_argv))) {
          int auto_inc = (string_argv[length] == '=') ? 1 : 0;
          value = atoi(&string_argv[length + auto_inc]);
        } else {
          value = 0;
        }

        bFound = true;
        continue;
      }
    }
  }

  if (bFound) {
    return value;
  } else {
    return 0;
  }
}

float getCmdLineArgumentFloat(const int argc, const char **argv,
                                     const char *string_ref) {
  bool bFound = false;
  float value = -1;

  if (argc >= 1) {
    for (int i = 1; i < argc; i++) {
      int string_start = stringRemoveDelimiter('-', argv[i]);
      const char *string_argv = &argv[i][string_start];
      int length = static_cast<int>(strlen(string_ref));

      if (!STRNCASECMP(string_argv, string_ref, length)) {
        if (length + 1 <= static_cast<int>(strlen(string_argv))) {
          int auto_inc = (string_argv[length] == '=') ? 1 : 0;
          value = static_cast<float>(atof(&string_argv[length + auto_inc]));
        } else {
          value = 0.f;
        }

        bFound = true;
        continue;
      }
    }
  }

  if (bFound) {
    return value;
  } else {
    return 0;
  }
}

bool getCmdLineArgumentString(const int argc, const char **argv,
                                     const char *string_ref,
                                     char **string_retval) {
  bool bFound = false;

  if (argc >= 1) {
    for (int i = 1; i < argc; i++) {
      int string_start = stringRemoveDelimiter('-', argv[i]);
      char *string_argv = const_cast<char *>(&argv[i][string_start]);
      int length = static_cast<int>(strlen(string_ref));

      if (!STRNCASECMP(string_argv, string_ref, length)) {
        *string_retval = &string_argv[length + 1];
        bFound = true;
        continue;
      }
    }
  }

  if (!bFound) {
    *string_retval = NULL;
  }

  return bFound;
}
}
}
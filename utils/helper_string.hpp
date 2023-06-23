#pragma once

#include <vector>
#include <string>

using namespace std;

namespace SYSTEMX {
namespace utils {
int stringRemoveDelimiter(char delimiter, const char* string);
vector<string> stringSplit(string s, string delimiter);
int getFileExtension(char* filename, char** extension);
bool checkCmdLineFlag(const int argc, const char** argv,
                             const char* string_ref);
template <class T>
bool getCmdLineArgumentValue(const int argc, const char** argv,
                                    const char* string_ref, T* value);
int getCmdLineArgumentInt(const int argc, const char** argv,
                                 const char* string_ref);
float getCmdLineArgumentFloat(const int argc, const char** argv,
                                     const char* string_ref);
bool getCmdLineArgumentString(const int argc, const char** argv,
                                     const char* string_ref, char** string_retval);
}
}
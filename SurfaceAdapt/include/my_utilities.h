#ifndef MYUTILITIES_H
#define MYUTILITIES_H

#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>


class MyUtilities{

 public:

  MyUtilities(){};
  ~MyUtilities(){};
  static std::string double_to_string(double value);
  static std::string int_to_string(int value);
  static int mkpath(const char *path, mode_t mode);

};

#endif

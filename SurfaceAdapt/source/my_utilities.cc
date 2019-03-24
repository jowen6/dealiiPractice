#include "my_utilities.h"
#include <sstream>

#include <fstream>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>

typedef struct stat Stat;

static int do_mkdir(const char *path, mode_t mode)
{
  Stat            st;
    int             status = 0;

    if (stat(path, &st) != 0)
    {
        /* Directory does not exist */
        if (mkdir(path, mode) != 0)
            status = -1;
    }
    else if (!S_ISDIR(st.st_mode))
    {
        errno = ENOTDIR;
        status = -1;
    }

    return(status);
}


std::string MyUtilities::double_to_string(double value)
{
  std::ostringstream strs;
  strs << value;
  return strs.str();
}

std::string MyUtilities::int_to_string(int value)
{
  std::ostringstream strs;
  strs << value;
  return strs.str();
}


int MyUtilities::mkpath(const char *path, mode_t mode)
{
  char           *pp;
  char           *sp;
  int             status;
  char           *copypath = strdup(path);

  status = 0;
  pp = copypath;
  while (status == 0 && (sp = strchr(pp, '/')) != 0)
  {
    if (sp != pp)
    {
            /* Neither root nor double slash in path */
            *sp = '\0';
	    status = do_mkdir(copypath, mode);     
            *sp = '/';
        }
    pp = sp + 1;
  }
  if (status == 0)
    status = do_mkdir(path, mode);     

  delete copypath;
  return (status);
}

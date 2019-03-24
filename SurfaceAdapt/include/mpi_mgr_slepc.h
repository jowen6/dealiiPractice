#ifndef MPIMGRSLEPC_H
#define MPIMGRSLEPC_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <fstream>

using namespace dealii;

class MPIMGRSLEPC{

 public:
  
    MPIMGRSLEPC(int argc, char **argv);
  ~MPIMGRSLEPC();


  int get_nb_processes(){ return nb_processes;  };
  int get_this_process(){ return this_process;  };

  MPI_Comm get_communicator(){ return mpi_communicator; };

  void write(std::string str) { pcout<<str<<std::endl; }
  
 private:
  
  int nb_processes;
  int this_process;

  MPI_Comm mpi_communicator;

  ConditionalOStream pcout;

};



#endif



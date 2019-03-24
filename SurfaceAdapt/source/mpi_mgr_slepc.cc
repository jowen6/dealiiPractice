#include "mpi_mgr_slepc.h"
#include <deal.II/base/utilities.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/slepc_solver.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/mpi.h>

#include <iostream>

using namespace dealii;



MPIMGRSLEPC::MPIMGRSLEPC(int argc,
                         char **argv)
 :
  mpi_communicator(MPI_COMM_WORLD),
  pcout(std::cout)
{

  SlepcInitialize(&argc,&argv,0,0);
  MultithreadInfo::set_thread_limit(1);
    
    
  nb_processes = Utilities::MPI::n_mpi_processes(mpi_communicator);
  this_process = Utilities::MPI::this_mpi_process(mpi_communicator);
  
  pcout.set_condition(this_process == 0);
  

 
  // Get the extra PETSc options:
  //const std::string PETSc_options = "all_of_argv_from_command_line";

  // or set the extra PETSc options by hand:
  //const std::string PETSc_options = "-eps_type krylovschur -st_ksp_type gmres -st_pc_type jacobi";
//  const std::string PETSc_options = "-eps_type krylovschur -st_ksp_type gmres -st_pc_type hypre -st_pc_hypre_type boomeramg -st_pc_hypre_boomeramg_max_iter 1";
    
    
    
  /*
    const std::string PETSc_options = "-eps_interval "+left_spectrum+","+right_spectrum+" -st_type sinvert -st_ksp_type preonly -st_pc_type cholesky -st_pc_factor_mat_solver_package mumps -mat_mumps_icntl_13 1";


  write("SLEPc options:     "+PETSc_options);
  
  // Set the extra PETSc options:
  PetscOptionsInsertString(NULL, PETSc_options.c_str());
*/



}


MPIMGRSLEPC::~MPIMGRSLEPC()
{
  SlepcFinalize();

}

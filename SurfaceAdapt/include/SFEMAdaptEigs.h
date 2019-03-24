//
//  SFEMAdapt.h
//  
//
//  Created by Justin Owen on 3/12/19.
//

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/mapping_q.h> //for MappingQ
#include <deal.II/grid/manifold_lib.h> //dealii SphericalManifold
#include <deal.II/base/function.h> //for Function class
#include <deal.II/fe/mapping_q_eulerian.h>//for MappingQEulerian
#include <deal.II/fe/fe_system.h>//for interpolation of lift
#include <deal.II/numerics/matrix_tools.h>//for boundary values

#include <deal.II/base/index_set.h>//PETSc wrapper
#include <deal.II/lac/petsc_sparse_matrix.h>//PETSc wrapper
#include <deal.II/lac/petsc_parallel_vector.h>//PETSc wrapper
//#include <deal.II/lac/slepc_solver.h>//SLEPc solver
#include "slepc_solver_MOD.h"//SLEPc solver with modified SolverBase class
#include <petscsys.h> //PETSc scalar

#include <fstream>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/numerics/error_estimator.h>

#include "Lifts.h" //for lift definitions
#include "Shapes.h" //for Shape definitions
#include "my_utilities.h" //for creating folder to write data to
#include "mpi_mgr_slepc.h" //for slepc controls
#include <deal.II/base/mpi.h> //for mpi
#include <memory>

using namespace dealii;


#ifndef SFEMAdaptEigs_h
#define SFEMAdaptEigs_h

//COMPLETE
template <int dim>
class SFEMAdaptEigs
{
public:
    SFEMAdaptEigs(MPIMGRSLEPC &mpi_mgr,
                  Shape<dim> &shape,
                  PetscReal cluster_lower_bound,
                  PetscReal cluster_upper_bound,
                  const unsigned degree = 1,
                  const unsigned mapping_degree = 1);
    
    ~SFEMAdaptEigs();
    void run_uniform_refinement();
    void run_adaptive_refinement();
    
private:
    void setup_system();
    void assemble_system();
    void Eigs_solve();
    void Eigs_estimate(double &estimated_pde_error);
    void solve_estimate_mark_refine(double PDE_tolerance);
    void GEOMETRY_estimate(double &max_estimated_geometric_error);
    void GEOMETRY_estimate_mark_refine(double GEOMETRY_tolerance);
    void Eigs_output_results(const unsigned int cycle);// const;
    void Eigs_compute_error() const;
    
    MPIMGRSLEPC&                     mpi_mgr;
    
    Triangulation<dim-1, dim>        triangulation;
    FE_Q<dim-1, dim>                 fe;
    FESystem<dim-1,dim>              fe_mapping; //System for interpolating
    DoFHandler<dim-1, dim>           dof_handler, dof_handler_mapping;
    AffineConstraints<double>        constraints, mapping_constraints; //for hanging nodes
    PETScWrappers::SparseMatrix      stiffness_matrix, mass_matrix;
    SparsityPattern                  sparsity_pattern;
    const unsigned int               fe_degree;

    PetscReal                               cluster_lower_bound;
    PetscReal                               cluster_upper_bound;
    std::vector<PETScWrappers::MPI::Vector> eigenfunctions;
    std::vector<double>                     eigenvalues;
    
    //Eigenfunction Cluster Estimator Values Vector
    Vector<float>                    estimated_eigenfunction_error_per_cell;
    Vector<float>                    estimated_cluster_error_per_cell;
    
    //Geometric Estimator Values Vector
    Vector<float>                    estimated_geometric_error_per_cell;
    Vector<double>                   approximate_lift;
    const unsigned int               mapping_degree;
    
    Shape<dim>                       &shape;
    Vector<double>                   InterpSolution; //FEM Interpolation of Solution;
    
};


template <int dim>
SFEMAdaptEigs<dim>::SFEMAdaptEigs(MPIMGRSLEPC &mpi_mgr,
                                  Shape<dim> &shape,
                                  PetscReal cluster_lower_bound,
                                  PetscReal cluster_upper_bound,
                                  const unsigned degree,
                                  const unsigned mapping_degree)
: mpi_mgr(mpi_mgr)
, fe(degree)
, fe_mapping(FE_Q<dim-1, dim>(mapping_degree), dim)
, dof_handler(triangulation)
, dof_handler_mapping(triangulation)
, fe_degree(degree)
, cluster_lower_bound(cluster_lower_bound)
, cluster_upper_bound(cluster_upper_bound)
, mapping_degree(mapping_degree)
, shape(shape)
{}


template <int dim>
SFEMAdaptEigs<dim>::~SFEMAdaptEigs ()
{
    dof_handler_mapping.clear();
    dof_handler.clear ();
    //delete lift;
}


//COMPLETE
template <int dim>
void SFEMAdaptEigs<dim>::setup_system()
{
    // Construct the mapping first
    dof_handler_mapping.distribute_dofs(fe_mapping);
    approximate_lift.reinit(dof_handler_mapping.n_dofs());
    mapping_constraints.clear();
    DoFTools::make_hanging_node_constraints (dof_handler_mapping,
                                             mapping_constraints);
    mapping_constraints.close();
    VectorTools::interpolate(dof_handler_mapping,
                             *(shape.GetLift()),
                             approximate_lift);
    mapping_constraints.distribute(approximate_lift);
    
    //Since mapping is within the scope of this method, we have to call it everywhere throughout the code. I'd rather have it as a variable of the class.
    MappingQEulerian<dim-1, Vector<double>, dim> mapping(mapping_degree,
                                                         dof_handler_mapping,
                                                         approximate_lift);
    
    dof_handler.distribute_dofs(fe);
    
    //solution.reinit(dof_handler.n_dofs());
    //system_rhs.reinit(dof_handler.n_dofs());
    
    // clear constraints then populate the AffineConstraints object with the hanging node constraints.
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    
    constraints.close();
    
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ false);
    sparsity_pattern.copy_from(dsp);
    //system_matrix.reinit(sparsity_pattern);
    stiffness_matrix.reinit(sparsity_pattern);
    mass_matrix.reinit(sparsity_pattern);
    //NEW
    /*
    stiffness_matrix.reinit (dof_handler.n_dofs(),
                             dof_handler.n_dofs(),
                             dof_handler.max_couplings_between_dofs());
    
    mass_matrix.reinit (dof_handler.n_dofs(),
                        dof_handler.n_dofs(),
                        dof_handler.max_couplings_between_dofs());
    */
    IndexSet eigenfunction_index_set = dof_handler.locally_owned_dofs();
    eigenfunctions.resize (8); //MAX number of eigenfunctions
    for (unsigned int i=0; i < eigenfunctions.size(); ++i)
        eigenfunctions[i].reinit (eigenfunction_index_set, MPI_COMM_WORLD);
    eigenvalues.resize (eigenfunctions.size ());

}


//COMPLETE
template <int dim>
void SFEMAdaptEigs<dim>::assemble_system()
{
    //create mapping
    MappingQEulerian<dim-1, Vector<double>, dim> mapping(mapping_degree,
                                                         dof_handler_mapping,
                                                         approximate_lift);
    
    const QGauss<dim-1>                     quadrature_formula(3);
    
    FEValues<2, 3>                          fe_values(mapping,
                                                      fe,
                                                      quadrature_formula,
                                                      update_values |
                                                      update_gradients |
                                                      update_quadrature_points |
                                                      update_JxW_values);
    
    const unsigned int                      dofs_per_cell(fe.dofs_per_cell);
    std::vector<types::global_dof_index>    local_dof_indices(dofs_per_cell);
    
    FullMatrix<double>                      cell_stiffness_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double>                      cell_mass_matrix(dofs_per_cell, dofs_per_cell);
    const unsigned int                      n_q_points(quadrature_formula.size());
    
    
    auto cell = dof_handler.begin_active();
    auto endc = dof_handler.end();
    
    for(; cell != endc; ++cell)
    {
        cell_stiffness_matrix = 0;
        cell_mass_matrix = 0;
        
        fe_values.reinit(cell);
        //rhs.value_list(fe_values.get_quadrature_points(), rhs_values);
        
        for(unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        {
            for(unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for(unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    cell_stiffness_matrix(i, j) += (fe_values.shape_grad(i, q_index)
                                                    * fe_values.shape_grad(j, q_index)
                                                    * fe_values.JxW(q_index));
                    
                    cell_mass_matrix(i, j) += (fe_values.shape_value(i, q_index)
                                               * fe_values.shape_value(j, q_index)
                                               * fe_values.JxW(q_index));
                    
                }
            }
        }
        
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_stiffness_matrix,
                                               local_dof_indices,
                                               stiffness_matrix);
        
        constraints.distribute_local_to_global(cell_mass_matrix,
                                               local_dof_indices,
                                               mass_matrix);
    }
    //NEW to Eigs
    //Combine contributions from individual processors
    stiffness_matrix.compress (VectorOperation::add);
    mass_matrix.compress (VectorOperation::add);
    
    double min_spurious_eigenvalue = std::numeric_limits<double>::max();
    double max_spurious_eigenvalue = -std::numeric_limits<double>::max();
    
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
        if (constraints.is_constrained(i))
        {
            const double ev = stiffness_matrix(i,i)/mass_matrix(i,i);
            min_spurious_eigenvalue = std::min (min_spurious_eigenvalue, ev);
            max_spurious_eigenvalue = std::max (max_spurious_eigenvalue, ev);
        }
    std::cout << "   Spurious eigenvalues are all in the interval "
    << "[" << min_spurious_eigenvalue << "," << max_spurious_eigenvalue << "]"
    << std::endl;
}


//COMPLETE
template <int dim>
void SFEMAdaptEigs<dim>::Eigs_solve()
{

    SolverControl                    solver_control(dof_handler.n_dofs(), 1e-9);
    SLEPcWrappers::SolverKrylovSchur eigensolver(solver_control);
    
    eigensolver.set_target_interval(cluster_lower_bound, cluster_upper_bound);
    eigensolver.set_problem_type(EPS_GHEP);
     
    eigensolver.solve(stiffness_matrix,
                      mass_matrix,
                      eigenvalues,
                      eigenfunctions);
     
    for(unsigned int i=0; i < eigenfunctions.size(); ++i){
        constraints.distribute(eigenfunctions[i]);
    }
    
    //create mapping
    MappingQEulerian<dim-1,Vector<double >, dim> mapping(mapping_degree,
                                                         dof_handler_mapping,
                                                         approximate_lift);
    QGauss<dim-1>               quad(3); //Degree 3 gauss quadrature
    for (unsigned int i = 0; i < eigenfunctions.size(); ++i){
        //compute and subtract off mean
        double mean = VectorTools::compute_mean_value(mapping,
                                                      dof_handler,
                                                      QGauss<dim-1>(3),
                                                      eigenfunctions[i],
                                                      0);
        eigenfunctions[i].add(-mean);
        //need to L2 normalize
        eigenfunctions[i] /= eigenfunctions[i].linfty_norm();
    }
}


template <int dim>
void SFEMAdaptEigs<dim>::Eigs_estimate(double &estimated_cluster_error)
{
    //create mapping
    MappingQEulerian<dim-1,Vector<double >, dim> mapping(mapping_degree,
                                                         dof_handler_mapping,
                                                         approximate_lift);
    
    //ESTIMATE
    estimated_cluster_error_per_cell.reinit(triangulation.n_active_cells());
    
    QGauss<dim-1>               quad(3); //Degree 3 gauss quadrature
    FEValues<2, dim>            fe_values(mapping,
                                          fe,
                                          quad,
                                          update_values |
                                          update_quadrature_points |
                                          update_hessians |
                                          update_JxW_values);
    
    
    
    const unsigned int          n_q_points = quad.size();
    std::vector<double >        laplacians_at_q_points(n_q_points);
    std::vector<double >        eigenfunction_at_q_points(n_q_points);

    double                      bulk_value;
    
    unsigned int                present_cell(0); // Keeps track of the number of the cell we're on
    auto                        cell(dof_handler.begin_active());
    auto                        endc(dof_handler.end());

    
    //Loop over Eigenfunctions
    for(unsigned int i = 0; i < eigenfunctions.size(); ++i)
    {
        //eigenfunction_at_q_points = 0;
        //laplacians_at_q_points = 0;
        estimated_eigenfunction_error_per_cell.reinit(triangulation.n_active_cells());
    
        //Jump Estimator
        KellyErrorEstimator<dim-1, dim>::estimate(mapping,
                                                  dof_handler,
                                                  QGauss<dim - 2>(3),
                                                  std::map<types::boundary_id,
                                                  const Function<dim> *>(),
                                                  eigenfunctions[i],
                                                  estimated_eigenfunction_error_per_cell);
       
        estimated_eigenfunction_error_per_cell.scale(estimated_eigenfunction_error_per_cell);
        
        //Squaring for adding to bulk
        estimated_cluster_error_per_cell += estimated_eigenfunction_error_per_cell;
    
        //Bulk Estimator
        for(; cell!=endc; ++cell, ++present_cell)
        {
            fe_values.reinit(cell);
            //Evaluate FEM solution's Laplacian at quadrature points and write them to
            // laplacians_at_q_points vector
            fe_values.get_function_laplacians(eigenfunctions[i], laplacians_at_q_points);
            fe_values.get_function_values(eigenfunctions[i], eigenfunction_at_q_points);
            //rhs.value_list(fe_values.get_quadrature_points(), rhs_at_q_points);
        
            for(unsigned int q_point=0; q_point < n_q_points; ++q_point)
            {
                // h*|lambda*u + Delta u|
                bulk_value = (cell->diameter())
                        *(eigenvalues[i]*eigenfunction_at_q_points[q_point] + laplacians_at_q_points[q_point]);
                // 1/24 done to match KellyEstimator
                estimated_eigenfunction_error_per_cell(present_cell) += (1/24)*bulk_value*bulk_value*fe_values.JxW(q_point);
            }
        }
        estimated_cluster_error_per_cell += estimated_eigenfunction_error_per_cell;
    }
    //Using L1 norm to sum up terms
    estimated_cluster_error = std::sqrt(estimated_cluster_error_per_cell.l1_norm());
}

//INCOMPLETE
template <int dim>
void SFEMAdaptEigs<dim>::solve_estimate_mark_refine(double CLUSTER_tolerance)
{
    double estimated_cluster_error = 0.0;
    unsigned int count = 0;
    
    //SOLVE
    setup_system();
    assemble_system();
    Eigs_solve();
    
    //ESTIMATE
    Eigs_estimate(estimated_cluster_error);
    
    std::cout << "Estimated Error: " << estimated_cluster_error << "\n";
    Eigs_output_results(count);
    
    while(estimated_cluster_error > CLUSTER_tolerance)
    {
        //MARK
        GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                          estimated_cluster_error_per_cell,
                                                          0.3,
                                                          0.0);
        
        //REFINE
        triangulation.execute_coarsening_and_refinement();
        
        //SOLVE
        setup_system();
        assemble_system();
        Eigs_solve();
        
        //ESTIMATE
        Eigs_estimate(estimated_cluster_error);
        
        //Output Info
        ++count;
        std::cout << "  Loop: " << count << "\n";
        std::cout << "      Active Cells: " << triangulation.n_active_cells() << "\n";
        std::cout << "      Estimated Error: " << estimated_cluster_error << "\n";
        
        Eigs_output_results(count);
        
        Eigs_compute_error(); //show actual error for test case
    }
}


template <int dim>
void SFEMAdaptEigs<dim>::GEOMETRY_estimate(double &estimated_geometric_error)
{
    //ESTIMATE
    estimated_geometric_error_per_cell.reinit(triangulation.n_active_cells());
    
    
    /*
     VectorTools::integrate_difference (dof_handler_mapping,
     approximate_lift,
     lift,
     estimated_geometric_error_per_cell,
     QGauss<2>(mapping_degree + 1),
     VectorTools::W1infty_seminorm);
     */
    
    VectorTools::integrate_difference(dof_handler_mapping,
                                      approximate_lift,
                                      *(shape.GetLift())/* *lift*/,
                                      estimated_geometric_error_per_cell,
                                      QGauss<2>(mapping_degree + 1),
                                      VectorTools::Linfty_norm);
    
    estimated_geometric_error = estimated_geometric_error_per_cell.linfty_norm();
}


template <int dim>
void SFEMAdaptEigs<dim>::GEOMETRY_estimate_mark_refine(double GEOMETRY_tolerance)
{
    double estimated_geometric_error(0.0);
    
    setup_system();
    GEOMETRY_estimate(estimated_geometric_error);
    std::cout << GEOMETRY_tolerance << "\n";
    int count = 0;
    
    while(estimated_geometric_error > GEOMETRY_tolerance)
    {
        //MARK
        GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                          estimated_geometric_error_per_cell,
                                                          0.3,
                                                          0.0);
        
        //REFINE
        triangulation.execute_coarsening_and_refinement();
        setup_system(); //Need to reset approximate lifts for mappings
        
        //ESTIMATE
        GEOMETRY_estimate(estimated_geometric_error);
        
        //Output Info
        ++count;
        std::cout<< "  Loop: "<< count << std::endl;
        std::cout << "      Active Cells: " << triangulation.n_active_cells() << std::endl;
        std::cout << "      Geometric Error: " << estimated_geometric_error << std::endl;
    }
}

//INCOMPLETE

template <int dim>
void SFEMAdaptEigs<dim>::Eigs_compute_error() const
{
    
}
/*
template <int dim>
void SFEMAdaptEigs<dim>::Eigs_compute_error() const
{
    //create mapping
    MappingQEulerian<dim-1, Vector<double>, dim> mapping(mapping_degree,
                                                         dof_handler_mapping,
                                                         approximate_lift);
    
    Vector<float> difference_per_cell(triangulation.n_active_cells());
    
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      solution,
                                      Solution,
                                      difference_per_cell,
                                      QGauss<dim-1>(2*fe.degree+1),
                                      VectorTools::H1_norm);
    
    double h1_error = VectorTools::compute_global_error(triangulation,
                                                        difference_per_cell,
                                                        VectorTools::H1_norm);
    
    std::cout << "      H1 error = " << h1_error << std::endl;
    
    difference_per_cell = 0;
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      solution,
                                      Solution,
                                      difference_per_cell,
                                      QGauss<dim-1>(2*fe.degree+1),
                                      VectorTools::L2_norm);
    
    double L2_error = VectorTools::compute_global_error(triangulation,
                                                        difference_per_cell,
                                                        VectorTools::L2_norm);
    
    std::cout << "      L2 error = " << L2_error << std::endl;
}
*/


//COMPLETED
template <int dim>
void SFEMAdaptEigs<dim>::Eigs_output_results(const unsigned int cycle) //const
{
    //create mapping
    MappingQEulerian<dim-1, Vector<double >, dim> mapping(mapping_degree,
                                                          dof_handler_mapping,
                                                          approximate_lift);
    
    
    std::string folder( "output/"
                        + shape.GetShapeName()
                        + "_"
                        + shape.GetLift()->GetLiftName()
                        + "_ClusterInterval_"
                        + std::to_string(cluster_lower_bound)
                        + "_"
                        + std::to_string(cluster_upper_bound)
                        + "_PDEDeg_" + std::to_string(fe_degree)
                        + "_MapDeg_" + std::to_string(mapping_degree)
                        + "/");
    
    MyUtilities::mkpath(folder.c_str(),0777);
    
    //Write eigenfunctions for given cycle to file
    DataOut<dim-1, DoFHandler<dim-1, dim> > data_out;
    data_out.attach_dof_handler(dof_handler);
    for (unsigned int i=0; i<eigenfunctions.size(); ++i)
    {
        data_out.add_data_vector (eigenfunctions[i],
                                  std::string("eigenfunction_") + Utilities::int_to_string(i),
                                  DataOut<dim-1, DoFHandler<dim-1, dim>>::type_dof_data);
    }
    data_out.build_patches(mapping, mapping.get_degree());
    
    
    std::string filename(folder + "Eigenfunctions-cycle-");
    filename += std::to_string(cycle);
    filename += ".vtk";
    std::ofstream output(filename);
    data_out.write_vtk (output);
    
    std::string eigvals_filename(folder + "Eigenvalues-cycle-");
    eigvals_filename += std::to_string(cycle);
    eigvals_filename += ".txt";
    std::ofstream of_eigenvalues(eigvals_filename,std::ios::app);
    of_eigenvalues.precision(10);
    of_eigenvalues<<std::scientific;
    
    for (unsigned int i=0;i<eigenfunctions.size();++i){
        of_eigenvalues<<eigenvalues[i]<<std::endl;
    }
}

//INCOMPLETE
template <int dim>
void SFEMAdaptEigs<dim>::run_uniform_refinement()
{
    shape.AssignPolyhedron(triangulation);  //Attach Polyhedron
    for (unsigned int cycle = 0; cycle < 3; ++cycle)
    {
        triangulation.refine_global(1);
        setup_system();
        assemble_system();
        Eigs_solve();
        
        std::cout << "Loop:                            "
        << cycle << std::endl;
        std::cout << "   Number of active cells:       "
        << triangulation.n_active_cells() << std::endl;
        
        std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
        << std::endl;
        
        //Eigs_compute_error();
        Eigs_output_results(cycle);
    }
}

//INCOMPLETE
template <int dim>
void SFEMAdaptEigs<dim>::run_adaptive_refinement()
{
    //shape->AssignPolyhedron(triangulation);  //Attach Polyhedron
    shape.AssignPolyhedron(triangulation);
    //std::cout << "ADAPT_SURFACE" << "\n";
    //GEOMETRY_estimate_mark_refine(0.00001);
    
    triangulation.refine_global(2);
    
    std::cout << "ADAPT_PDE" << "\n";
    solve_estimate_mark_refine(0.01);
}


template <int dim>
class SFEMTest
{
public:
    SFEMTest(std::unique_ptr<Shape<dim>> shape_ptr, int a, int b): shape(std::move(shape_ptr)), a(a), b(b)
    {std::cout << "success" << std::endl; shape->GetLift()->print_Lift(); }
    
private:
    std::unique_ptr<Shape<dim>>     shape;
    int a;
    int b;
};


#endif /* SFEMAdapt_h */

//This is a modified version of Step-6 within the dealii library examples for my own reference. I have added such things as a bulk estimator and played with various GridGenerators.


/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2018 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 2000
 */


// @sect3{Include files}

// The first few files have already been covered in previous examples and will
// thus not be further commented on.
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
#include <deal.II/grid/manifold_lib.h> //SphericalManifold
#include <deal.II/base/function.h> //for Function class
#include <deal.II/fe/mapping_q_eulerian.h>//for MappingQEulerian
#include <deal.II/fe/fe_system.h>//for interpolation of lift

#include <fstream>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/numerics/error_estimator.h>

using namespace dealii;




//Abstract Lift Class
template <int spacedim>
class Lift :  public Function<spacedim>
{
public:
    Lift () : Function<spacedim>(spacedim) {};
    
    virtual void vector_value (const Point<spacedim> &p,
                               Vector<double>   &values) const = 0;
    
    
    virtual void vector_gradient(const Point<spacedim> &p,
                                 std::vector<Tensor<1,spacedim,double> > &gradients) const = 0;
    virtual void print_Lift() = 0;
};


//RadialLift
template <int spacedim>
class RadialLift : public Lift<spacedim>{
    void vector_value (const Point<spacedim> &p,
                       Vector<double>   &values) const;
    
    void vector_gradient(const Point<spacedim> &p,
                         std::vector<Tensor<1,spacedim,double> > &gradients) const;
    
    void print_Lift();
};


//Define RadialLift vector pointing from point to exact surface
template <int spacedim>
void RadialLift<spacedim>::vector_value(const Point<spacedim> &p, Vector<double> &values) const{
    double length = p.norm();
    
    for (unsigned int i=0; i<spacedim; ++i){
        values(i) = (p(i)/length)-p(i);
    }
}


//Define RadialLift gradient for 2D
template <>
void RadialLift<2>::vector_gradient(const Point<2> &p,
                                    std::vector<Tensor<1, 2, double> > &gradients) const{
    double length = p.norm();
    //first component
    gradients[0][0] = -1./length/length/length * p(0)*p(0) + (1./length-1.0);
    gradients[0][1] = -1./length/length/length * p(1)*p(0);
    
    
    //second component
    gradients[1][0] = -1./length/length/length * p(0)*p(1);
    gradients[1][1] = -1./length/length/length * p(1)*p(1) + (1./length-1.0);
}

//Define RadialLift gradient for 3D
template <>
void RadialLift<3>::vector_gradient(const Point<3> &p,
                                    std::vector<Tensor<1, 3, double> > &gradients) const{
    double length = p.norm();
    //first component
    gradients[0][0] = -1./length/length/length * p(0)*p(0) + (1./length-1.0);
    gradients[0][1] = -1./length/length/length * p(1)*p(0);
    gradients[0][2] = -1./length/length/length * p(2)*p(0);
    
    
    //second component
    gradients[1][0] = -1./length/length/length * p(0)*p(1);
    gradients[1][1] = -1./length/length/length * p(1)*p(1) + (1./length-1.0);
    gradients[1][2] = -1./length/length/length * p(2)*p(1);
    
    //third component
    gradients[2][0] = -1./length/length/length * p(0)*p(2);
    gradients[2][1] = -1./length/length/length * p(1)*p(2);
    gradients[2][2] = -1./length/length/length * p(2)*p(2) + (1./length-1.0);
}

//Print RadialLift details
template <int spacedim>
void RadialLift<spacedim>::print_Lift(){
    std::cout<< "RadialLift given by mapping x->x/|x|" << "\n";
}


//Half-Sphere x-axis is height
template <int dim>
double coefficient(const Point<dim> &p)
{
    if (p(2)*p(2) + p(1)*p(1) < 0.4 * 0.4)
        return 5;
    else if (p(2)*p(2) + p(1)*p(1) > 0.8 * 0.8)
        return 10;
    else
        return 15;
}


//Half-Sphere x-axis is height
/*
template <int dim>
double coefficient(const Point<dim> &p)
{
    if (p(0) < 0.0)
        return 5;
    //else if (p(2)*p(2) + p(0)*p(0) > 0.8 * 0.8)
    //    return 10;
    else
        return 15;
}
*/

template <int dim>
class Solution : public Function<dim>
{
public:
    Solution()
    : Function<dim>()
    {}
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
    virtual Tensor<1, dim>
    gradient(const Point<dim> & p,
             const unsigned int component = 0) const override;
};


template <>
double Solution<2>::value(const Point<2> &p, const unsigned int) const
{
    return (-2. * p(0) * p(1));
}


template <>
Tensor<1, 2> Solution<2>::gradient(const Point<2> &p,
                                   const unsigned int) const
{
    Tensor<1, 2> return_value;
    return_value[0] = -2. * p(1) * (1 - 2. * p(0) * p(0));
    return_value[1] = -2. * p(0) * (1 - 2. * p(1) * p(1));
    return return_value;
}


template <>
double Solution<3>::value(const Point<3> &p, const unsigned int) const
{
    return (std::sin(numbers::PI * p(0)) * std::cos(numbers::PI * p(1)) *
            exp(p(2)));
}


template <>
Tensor<1, 3> Solution<3>::gradient(const Point<3> &p,
                                   const unsigned int) const
{
    using numbers::PI;
    Tensor<1, 3> return_value;
    return_value[0] = PI * cos(PI * p(0)) * cos(PI * p(1)) * exp(p(2));
    return_value[1] = -PI * sin(PI * p(0)) * sin(PI * p(1)) * exp(p(2));
    return_value[2] = sin(PI * p(0)) * cos(PI * p(1)) * exp(p(2));
    return return_value;
}


template <int dim>
class RightHandSide : public Function<dim>
{
public:
    RightHandSide()
    : Function<dim>()
    {}
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
};

template <>
double RightHandSide<2>::value(const Point<2> &p, const unsigned int /*component*/) const
{
    return (-8. * p(0) * p(1));
}

//Sphere
//template <>
//double RightHandSide<3>::value(const Point<3> &p, const unsigned int /*component*/) const
/*
{
    using numbers::PI;
    Tensor<2, 3> hessian;
    hessian[0][0] = -PI * PI * sin(PI * p(0)) * cos(PI * p(1)) * exp(p(2));
    hessian[1][1] = -PI * PI * sin(PI * p(0)) * cos(PI * p(1)) * exp(p(2));
    hessian[2][2] = sin(PI * p(0)) * cos(PI * p(1)) * exp(p(2));
    hessian[0][1] = -PI * PI * cos(PI * p(0)) * sin(PI * p(1)) * exp(p(2));
    hessian[1][0] = -PI * PI * cos(PI * p(0)) * sin(PI * p(1)) * exp(p(2));
    hessian[0][2] = PI * cos(PI * p(0)) * cos(PI * p(1)) * exp(p(2));
    hessian[2][0] = PI * cos(PI * p(0)) * cos(PI * p(1)) * exp(p(2));
    hessian[1][2] = -PI * sin(PI * p(0)) * sin(PI * p(1)) * exp(p(2));
    hessian[2][1] = -PI * sin(PI * p(0)) * sin(PI * p(1)) * exp(p(2));
    Tensor<1, 3> gradient;
    gradient[0] = PI * cos(PI * p(0)) * cos(PI * p(1)) * exp(p(2));
    gradient[1] = -PI * sin(PI * p(0)) * sin(PI * p(1)) * exp(p(2));
    gradient[2] = sin(PI * p(0)) * cos(PI * p(1)) * exp(p(2));
    Point<3> normal = p;
    normal /= p.norm();
    return (-trace(hessian) + 2 * (gradient * normal) +
            (hessian * normal) * normal);
}
*/
//Torus: y is axis of symmetry
template <>
double RightHandSide<3>::value(const Point<3> &p, const unsigned int /*component*/) const
{
    //return p(2);
    return std::sin(numbers::PI * p(2));
}


template <int dim>
class Step6
{
public:
    Step6(const unsigned degree = 1, const unsigned mapping_degree = 1);
    
    void run();
    
private:
    void setup_system();
    void assemble_system();
    void solve();
    void PDE_estimate(double &estimated_pde_error);
    void PDE_solve_estimate_mark_refine(double PDE_tolerance);
    void GEOMETRY_estimate(double &max_estimated_geometric_error);
    void GEOMETRY_estimate_mark_refine(double GEOMETRY_tolerance);
    void output_results(const unsigned int cycle) const;
    
    
    
    Triangulation<dim-1, dim>        triangulation;
    FE_Q<dim-1, dim>                 fe;
    FESystem<dim-1,dim>              fe_mapping; //System for interpolating
    DoFHandler<dim-1, dim>           dof_handler, dof_handler_mapping;
    AffineConstraints<double> constraints, mapping_constraints; //for hanging nodes
    SparseMatrix<double> system_matrix;
    SparsityPattern      sparsity_pattern;
    
    Vector<double> solution;
    Vector<double> system_rhs;
    
    //PDE Estimator Values Vector
    Vector<float> estimated_error_per_cell;
    
    //Geometric Estimator Values Vector
    Vector<float> estimated_geometric_error_per_cell;
    Vector<double>   approximate_lift;
    const unsigned int mapping_degree;
    RadialLift<dim>   lift;
    
};


template <int dim>
Step6<dim>::Step6(const unsigned degree, const unsigned mapping_degree)
  : fe(degree)
  , fe_mapping(FE_Q<dim-1, dim>(mapping_degree), dim)
  , dof_handler(triangulation)
  , dof_handler_mapping(triangulation)
  , mapping_degree(mapping_degree)
{}


template <int dim>
void Step6<dim>::setup_system()
{
  // Construct the mapping first
  //==============================
  //fe_mapping(FE_Q<dim-1, dim>(mapping_degree), dim);
  dof_handler_mapping.distribute_dofs(fe_mapping);
  approximate_lift.reinit(dof_handler_mapping.n_dofs());

    
    
  mapping_constraints.clear ();
  DoFTools::make_hanging_node_constraints (dof_handler_mapping, mapping_constraints);
    
  mapping_constraints.close ();
    
    
  VectorTools::interpolate(dof_handler_mapping,lift,approximate_lift);
  mapping_constraints.distribute (approximate_lift);
    
    
    
  //Since mapping is within the scope of this method, we have to call it everywhere throughout the code. I'd rather have it as a variable of the class.
  MappingQEulerian<dim-1,Vector<double >,dim> mapping(mapping_degree,dof_handler_mapping,approximate_lift);
  //==============================
    
    
  dof_handler.distribute_dofs(fe);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  // clear constraints then populate the AffineConstraints object with the hanging node constraints.
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);

  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           constraints);

  // After all constraints have been added, they need to be sorted and
  // rearranged to perform some actions more efficiently. This postprocessing
  // is done using the <code>close()</code> function, after which no further
  // constraints may be added any more:
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ false);

  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
    
}

template <int dim>
void Step6<dim>::assemble_system()
{
  //create mapping
  MappingQEulerian<dim-1,Vector<double >,dim> mapping(mapping_degree,dof_handler_mapping,approximate_lift);
    
  const QGauss<dim-1> quadrature_formula(3);

    FEValues<2,3> fe_values(mapping,
                            fe,
                            quadrature_formula,
                            update_values |
                            update_gradients |
                            update_quadrature_points |
                            update_JxW_values);
    
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
    
  std::vector<double>    rhs_values(n_q_points);
  const RightHandSide<dim> rhs;
    

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  typename DoFHandler<dim-1, dim>::active_cell_iterator cell =
                                                   dof_handler.begin_active(),
                                                 endc = dof_handler.end();
  for (; cell != endc; ++cell)
    {
      cell_matrix = 0;
      cell_rhs    = 0;

      fe_values.reinit(cell);
      rhs.value_list(fe_values.get_quadrature_points(), rhs_values);
        
      for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        {
            const double current_coefficient = coefficient<dim>(fe_values.quadrature_point(q_index));
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                cell_matrix(i, j) +=
                  (current_coefficient * fe_values.shape_grad(i, q_index) *
                   fe_values.shape_grad(j, q_index) * fe_values.JxW(q_index));

                cell_rhs(i) += (fe_values.shape_value(i, q_index) * rhs_values[q_index] *
                                fe_values.JxW(q_index));
            }
        }

      // Finally, transfer the contributions from @p cell_matrix and
      // @p cell_rhs into the global objects.
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
}


//Had to loosen up solver_controls to get convergence
template <int dim>
void Step6<dim>::solve()
{
  //SolverControl solver_control(1000, 1e-12);
  SolverControl solver_control(3000, 0.00001);
  SolverCG<>    solver(solver_control);

  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);

  solver.solve(system_matrix, solution, system_rhs, preconditioner);

  constraints.distribute(solution);
}



template <int dim>
void Step6<dim>::PDE_estimate(double &estimated_pde_error)
{
    //create mapping
    MappingQEulerian<dim-1,Vector<double >,dim> mapping(mapping_degree,dof_handler_mapping,approximate_lift);
  
    //ESTIMATE
    estimated_error_per_cell.reinit(triangulation.n_active_cells());
    
    //Jump Estimator
    KellyErrorEstimator<dim-1, dim>::estimate(mapping,
                                       dof_handler,
                                       QGauss<dim - 2>(3),
                                       std::map<types::boundary_id,
                                       const Function<dim> *>(),
                                       solution,
                                       estimated_error_per_cell);
    
    //Bulk Estimator
    QGauss<dim-1>  quad(3); //Degree 3 gauss quadrature
    FEValues<2, dim> fe_values (mapping,
                                fe,
                                quad,
                                update_values |
                                update_quadrature_points |
                                update_hessians |
                                update_JxW_values);
    
    const unsigned int   n_q_points = quad.size();
    
    std::vector<double > laplacians_at_q_points(n_q_points);
    std::vector<double > rhs_at_q_points(n_q_points);
    const RightHandSide<dim> rhs;
    
    auto cell = dof_handler.begin_active();
    auto endc = dof_handler.end();
    
    
    int present_cell = 0; // Keeps track of the number of the cell we're on
    double dValue;
    

    for (; cell!=endc; ++cell,++present_cell)
    {

            fe_values.reinit(cell);
            
            //Evaluate FEM solution's Laplacian at quadrature points and write them to
            // laplacians_at_q_points vector
            fe_values.get_function_laplacians(solution, laplacians_at_q_points);
            rhs.value_list(fe_values.get_quadrature_points(), rhs_at_q_points);
            
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point){
                // h*|f+Delta u|
                const double current_coefficient = coefficient<dim>(fe_values.quadrature_point(q_point));
                
                dValue = cell->diameter()/fe.degree*(rhs_at_q_points[q_point] + current_coefficient*laplacians_at_q_points[q_point]);
                
                estimated_error_per_cell(present_cell) += dValue*dValue*fe_values.JxW(q_point);
                
                
            }
        
    }
    
    estimated_pde_error = estimated_error_per_cell.l2_norm();
}


template <int dim>
void Step6<dim>::PDE_solve_estimate_mark_refine(double PDE_tolerance)
{
    double estimated_pde_error = 0.0;
    int count = 0;
    std::cout << "ADAPT_PDE" << "\n";
    //SOLVE
    assemble_system();
    solve();
    
    //ESTIMATE
    PDE_estimate(estimated_pde_error);
    
    std::cout << estimated_pde_error << "\n";
    output_results(count);
    while(estimated_pde_error > PDE_tolerance){
        
        //MARK
        GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                          estimated_error_per_cell,
                                                          0.3,
                                                          0.0);
        
        //REFINE
        triangulation.execute_coarsening_and_refinement();
        setup_system();
        
        //SOLVE
        assemble_system();
        solve();
        
        //ESTIMATE
        PDE_estimate(estimated_pde_error);
        
        ++count;
        std::cout << count << "\n";
        std::cout << triangulation.n_active_cells() << "\n";
        std::cout << estimated_pde_error << "\n";
        
        output_results(count);
    }
}


template <int dim>
void Step6<dim>::GEOMETRY_estimate(double &max_estimated_geometric_error)
{
    std::cout << "ADAPT_SURFACE" << "\n";
    //ESTIMATE
    estimated_geometric_error_per_cell.reinit(triangulation.n_active_cells());
    
    /*
    VectorTools::integrate_difference (dof_handler_mapping, approximate_lift,
                                       lift, estimated_geometric_error_per_cell,
                                       QGauss<2>(mapping_degree + 1),
                                       VectorTools::W1infty_seminorm);
    */
    
    VectorTools::integrate_difference (dof_handler_mapping, approximate_lift,
                                       lift, estimated_geometric_error_per_cell,
                                       QGauss<2>(mapping_degree + 1),
                                       VectorTools::L2_norm);
    
    max_estimated_geometric_error = estimated_geometric_error_per_cell.linfty_norm();
}


template <int dim>
void Step6<dim>::GEOMETRY_estimate_mark_refine(double GEOMETRY_tolerance)
{
    double max_estimated_geometric_error = 0.0;
    GEOMETRY_estimate(max_estimated_geometric_error);
    std::cout << GEOMETRY_tolerance << "\n";
    int count = 0;
    //estimated_geometric_error_per_cell.print();
    
    while(max_estimated_geometric_error > GEOMETRY_tolerance){
        //MARK
        GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                          estimated_geometric_error_per_cell,
                                                          0.3,
                                                          0.0);
        
        //REFINE
        triangulation.execute_coarsening_and_refinement();
        setup_system(); //Need to reset approximate lifts for mappings
        
        //ESTIMATE
        GEOMETRY_estimate(max_estimated_geometric_error);
        
        ++count;
        std::cout<< count <<"\n";
        std::cout << triangulation.n_active_cells() << "\n";
        std::cout << max_estimated_geometric_error << "\n";
    }
    
}


template <int dim>
void Step6<dim>::output_results(const unsigned int cycle) const
{
    //create mapping
    MappingQEulerian<dim-1,Vector<double >,dim> mapping(mapping_degree,dof_handler_mapping,approximate_lift);
    
    DataOut<dim-1, DoFHandler<dim-1, dim>> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(
                             solution,
                             "solution",
                             DataOut<dim-1, DoFHandler<dim-1, dim>>::type_dof_data);
    data_out.build_patches(mapping, mapping.get_degree());
    std::string filename("solution-cycle-");
    filename += static_cast<char>('0' + cycle);
    filename += "-";
    filename += static_cast<char>('0' + dim);
    filename += "d.vtk";
    std::ofstream output(filename);
    data_out.write_vtk(output);
}

template <int dim>
void Step6<dim>::run()
{
  for (unsigned int cycle = 0; cycle < 2; ++cycle)
    {
      std::cout << "Cycle " << cycle << ':' << std::endl;

      if (cycle == 0){
            //Half-Sphere
          /*
            {
                Triangulation<dim> volume_mesh;
                GridGenerator::half_hyper_ball(volume_mesh);
                std::set<types::boundary_id> boundary_ids;
                boundary_ids.insert(0);
                GridGenerator::extract_boundary_mesh(volume_mesh,
                                                     triangulation,
                                                     boundary_ids);
            }
            triangulation.set_all_manifold_ids(0);
            triangulation.set_manifold(0, SphericalManifold<dim-1, dim>());
            triangulation.refine_global(4);
           */
          
            //Sphere
            GridGenerator::hyper_sphere(triangulation);
            triangulation.refine_global(3);
            //Set manifold to be polyhedral so future refinements don't add polyhedral faces
            static FlatManifold<2,3> polyhedral_surface_description;
            triangulation.set_manifold(0, polyhedral_surface_description);
          
          
            //Torus
            /*
            GridGenerator::torus(triangulation,
                                 0.7,
                                 0.5);
            triangulation.refine_global(3);
            
            std::cout << "Surface mesh has " << triangulation.n_active_cells()
            << " cells." << std::endl;
            */
            
      }
      else{
          GEOMETRY_estimate_mark_refine(0.003);
          //triangulation.refine_global(2);
          setup_system();
          PDE_solve_estimate_mark_refine(0.0031);
          
      }
        
      std::cout << "   Number of active cells:       "
                << triangulation.n_active_cells() << std::endl;

      setup_system();

      std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
                << std::endl;
      
      //assemble_system();
      //solve();
      //output_results(cycle);
      
    }
    
}


int main()
{
  try
    {
        Step6<3> laplace_problem_2d(1,1);
      laplace_problem_2d.run();
    }

  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }

  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}

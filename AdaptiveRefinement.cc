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


#include <fstream>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/numerics/error_estimator.h>

// Finally, this is as in previous programs:
using namespace dealii;

template <int dim>
class Step6
{
public:
  Step6(const unsigned degree = 1);

  void run();

private:
  void setup_system();
  void assemble_system();
  void solve();
  void refine_grid();
  //void output_results(const unsigned int cycle) const;
  void output_results() const;
  //Triangulation<dim>        triangulation;
  //FE_Q<dim>                 fe;
  //DoFHandler<dim>           dof_handler;
  Triangulation<dim-1, dim>        triangulation;
  FE_Q<dim-1, dim>                 fe;
  DoFHandler<dim-1, dim>           dof_handler;
  MappingQ<2, 3>      mapping;

  // This is the new variable in the main class. We need an object which holds
  // a list of constraints to hold the hanging nodes and the boundary
  // conditions.
  AffineConstraints<double> constraints;

  SparseMatrix<double> system_matrix;
  SparsityPattern      sparsity_pattern;

  Vector<double> solution;
  Vector<double> system_rhs;
};


// @sect3{Nonconstant coefficients}

// The implementation of nonconstant coefficients is copied verbatim from
// step-5:
/*
template <int dim>
double coefficient(const Point<dim> &p)
{
  if (p.square() < 0.4 * 0.4)
    return 20;
  else if (p.square() > 0.8 * 0.8)
    return 20;
  else
    return 15;
}
*/
//Half-Sphere x-axis is height
/*
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
*/

//Half-Sphere x-axis is height
template <int dim>
double coefficient(const Point<dim> &p)
{
    if (p(0) < 0.4)
        return 5;
    //else if (p(2)*p(2) + p(0)*p(0) > 0.8 * 0.8)
    //    return 10;
    else
        return 15;
}

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
//Torus y is axis of symmetry
template <>
double RightHandSide<3>::value(const Point<3> &p, const unsigned int /*component*/) const
{
    return p(2);
}

template <int dim>
Step6<dim>::Step6(const unsigned degree)
  : fe(2)
  , dof_handler(triangulation)
  , mapping(degree)
{}


template <int dim>
void Step6<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  // We may now populate the AffineConstraints object with the hanging node
  // constraints. Since we will call this function in a loop we first clear
  // the current set of constraints from the last system and then compute new
  // ones:
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);


  // Now we are ready to interpolate the boundary values with indicator 0 (the
  // whole boundary) and store the resulting constraints in our
  // <code>constraints</code> object. Note that we do not to apply the
  // boundary conditions after assembly, like we did in earlier steps: instead
  // we put all constraints on our function space in the AffineConstraints
  // object. We can add constraints to the AffineConstraints object in either
  // order: if two constraints conflict then the constraint matrix either abort
  // or throw an exception via the Assert macro.
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

  // We may now, finally, initialize the sparse matrix:
  system_matrix.reinit(sparsity_pattern);
}

template <int dim>
void Step6<dim>::assemble_system()
{
  //const QGauss<dim> quadrature_formula(3);
  const QGauss<dim-1> quadrature_formula(3);
/*
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values |
                          update_gradients |
                          update_quadrature_points |
                          update_JxW_values);
*/
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

              //cell_rhs(i) += (fe_values.shape_value(i, q_index) * 1.0 *fe_values.JxW(q_index));
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


template <int dim>
void Step6<dim>::solve()
{
  SolverControl solver_control(1000, 1e-12);
  SolverCG<>    solver(solver_control);

  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);

  solver.solve(system_matrix, solution, system_rhs, preconditioner);

  constraints.distribute(solution);
}



template <int dim>
void Step6<dim>::refine_grid()
{
  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    
//Jump Estimator
  /*
  KellyErrorEstimator<dim>::estimate(dof_handler,
                                     QGauss<dim - 1>(3),
                                     std::map<types::boundary_id,
                                     const Function<dim> *>(),
                                     solution,
                                     estimated_error_per_cell);
   */
    KellyErrorEstimator<dim-1, dim>::estimate(mapping,
                                       dof_handler,
                                       QGauss<dim - 2>(3),
                                       std::map<types::boundary_id,
                                       const Function<dim> *>(),
                                       solution,
                                       estimated_error_per_cell);
    
//Bulk Estimator
    //QGauss<dim>  quad(3); //Degree 3 gauss quadrature
    
    
    //Create object containing FEM values
    /*
    FEValues<dim> fe_values (fe,
                             quad,
                             update_values |
                             update_quadrature_points |
                             update_hessians |
                             update_JxW_values);
     */
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
    
    int count = 0;
    for (; cell!=endc; ++cell,++present_cell)
    {
        //is_locally_owned checks if cell is owned by local processor for parallel computing
        //if using regular triangulation, then always returns true
        if(cell->is_locally_owned()){
            ++count;
            fe_values.reinit(cell);
            
            //Evaluate FEM solution's Laplacian at quadrature points and write them to
            // laplacians_at_q_points vector
            fe_values.get_function_laplacians(solution, laplacians_at_q_points);
            rhs.value_list(fe_values.get_quadrature_points(), rhs_at_q_points);
            
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point){
                // h*|1+Delta u|
                const double current_coefficient = coefficient<dim>(fe_values.quadrature_point(q_point));
                
                dValue = cell->diameter()/fe.degree*(rhs_at_q_points[q_point] + current_coefficient*laplacians_at_q_points[q_point]);
                
                estimated_error_per_cell(present_cell) += dValue*dValue*fe_values.JxW(q_point);
                
                
            }
        }
    }

  GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                  estimated_error_per_cell,
                                                  0.3,
                                                  0.03);

  triangulation.execute_coarsening_and_refinement();
}

/*
template <int dim>
void Step6<dim>::output_results(const unsigned int cycle) const
{
  {
    GridOut       grid_out;
    std::ofstream output("grid-" + std::to_string(cycle) + ".eps");
    grid_out.write_eps(triangulation, output);
  }

  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches();

    std::ofstream output("solution-" + std::to_string(cycle) + ".vtk");
    data_out.write_vtk(output);
  }
}
*/

template <int dim>
void Step6<dim>::output_results() const
{
    DataOut<dim-1, DoFHandler<dim-1, dim>> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(
                             solution,
                             "solution",
                             DataOut<dim-1, DoFHandler<dim-1, dim>>::type_dof_data);
    data_out.build_patches(mapping, mapping.get_degree());
    std::string filename("solution-");
    filename += static_cast<char>('0' + dim);
    filename += "d.vtk";
    std::ofstream output(filename);
    data_out.write_vtk(output);
}

template <int dim>
void Step6<dim>::run()
{
  for (unsigned int cycle = 0; cycle < 5; ++cycle)
    {
      std::cout << "Cycle " << cycle << ':' << std::endl;

      if (cycle == 0)
        {
            //Point<dim, double> a(0.0,0.0);
            //Point<dim, double> b(1.0,0.5);
            //Point<dim, double> c(0.5,1.0);
            //std::vector<Point<dim, double>> vertices = {a, b, c};
            //GridGenerator::simplex(triangulation, vertices);
            //GridGenerator::truncated_cone(triangulation);
            //GridGenerator::hyper_L(triangulation);
            //triangulation.refine_global(2);
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
            GridGenerator::torus(triangulation,
                                 0.7,
                                 0.5);
            triangulation.refine_global(3);
            
            std::cout << "Surface mesh has " << triangulation.n_active_cells()
            << " cells." << std::endl;
        }
      else
        refine_grid();


      std::cout << "   Number of active cells:       "
                << triangulation.n_active_cells() << std::endl;

      setup_system();

      std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
                << std::endl;

      assemble_system();
      solve();
      //output_results(cycle);
      
    }
    output_results();
}


int main()
{
  // The general idea behind the layout of this function is as follows: let's
  // try to run the program as we did before...
  try
    {
      Step6<3> laplace_problem_2d;
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
  // If the exception that was thrown somewhere was not an object of a class
  // derived from the standard <code>exception</code> class, then we can't do
  // anything at all. We then simply print an error message and exit.
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

  // If we got to this point, there was no exception which propagated up to
  // the main function (there may have been exceptions, but they were caught
  // somewhere in the program or the library). Therefore, the program
  // performed as was expected and we can return without error.
  return 0;
}

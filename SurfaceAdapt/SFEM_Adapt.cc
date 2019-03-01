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

#include <fstream>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/numerics/error_estimator.h>

#include "Lifts.h" //for lift definitions
#include "Shapes.h" //for Shape definitions
#include <memory>

using namespace dealii;

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
}*/


template <int dim>
double coefficient(const Point<dim> &p)
{
    return 1 + 0.*p(0); //To prevent warning about unused p;
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
    virtual double value(const Point<dim> & p, const unsigned int component = 0) const override;
    virtual Tensor<1, dim> gradient(const Point<dim> & p, const unsigned int component = 0) const override;
};


template <>
double Solution<2>::value(const Point<2> &p, const unsigned int) const
{
    return (-2. * p(0) * p(1));
}


template <>
Tensor<1, 2> Solution<2>::gradient(const Point<2> &p, const unsigned int) const
{
    Tensor<1, 2> return_value;
    return_value[0] = -2. * p(1) * (1 - 2. * p(0) * p(0));
    return_value[1] = -2. * p(0) * (1 - 2. * p(1) * p(1));
    return return_value;
}


template <>
double Solution<3>::value(const Point<3> &p, const unsigned int) const
{
    return (std::sin(numbers::PI * p(0)) * std::cos(numbers::PI * p(1)) * exp(p(2)));
}


template <>
Tensor<1, 3> Solution<3>::gradient(const Point<3> &p, const unsigned int) const
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
template <>
double RightHandSide<3>::value(const Point<3> &p, const unsigned int /*component*/) const
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
    
    return (-trace(hessian) + 2 * (gradient * normal) + (hessian * normal) * normal);
}


//Torus: y is axis of symmetry

//template <>
//double RightHandSide<3>::value(const Point<3> &p, const unsigned int /*component*/) const
/*
{
    //return p(2);
    return std::sin(numbers::PI * p(2));
}
*/


template <int dim>
class SFEMAdapt
{
public:
    //SFEMAdapt(Shape<dim> *shape_ptr, const unsigned degree = 1, const unsigned mapping_degree = 1);
    //SFEMAdapt(std::unique_ptr<Shape<dim> > shape_ptr, const unsigned degree = 1, const unsigned mapping_degree = 1);
    
     SFEMAdapt(std::unique_ptr<Shape<dim> > shape_ptr, const unsigned degree, const unsigned mapping_degree)
     : fe(degree)
     , fe_mapping(FE_Q<dim-1, dim>(mapping_degree), dim)
     , dof_handler(triangulation)
     , dof_handler_mapping(triangulation)
     , mapping_degree(mapping_degree)
     , shape(std::move(shape_ptr))
     {lift = shape->GetLift(); }
    
    void run_uniform_refinement();
    void run_adaptive_refinement();
    
private:
    void setup_system();
    void assemble_system();
    void solve();
    void PDE_estimate(double &estimated_pde_error);
    void PDE_solve_estimate_mark_refine(double PDE_tolerance);
    void GEOMETRY_estimate(double &max_estimated_geometric_error);
    void GEOMETRY_estimate_mark_refine(double GEOMETRY_tolerance);
    void output_results(const unsigned int cycle) const;
    void compute_error() const;
    
    Triangulation<dim-1, dim>        triangulation;
    FE_Q<dim-1, dim>                 fe;
    FESystem<dim-1,dim>              fe_mapping; //System for interpolating
    DoFHandler<dim-1, dim>           dof_handler, dof_handler_mapping;
    AffineConstraints<double>        constraints, mapping_constraints; //for hanging nodes
    SparseMatrix<double>             system_matrix;
    SparsityPattern                  sparsity_pattern;
    
    Vector<double>                   solution;
    Vector<double>                   system_rhs;
    
    //PDE Estimator Values Vector
    Vector<float>                    estimated_pde_error_per_cell;
    
    //Geometric Estimator Values Vector
    Vector<float>                    estimated_geometric_error_per_cell;
    Vector<double>                   approximate_lift;
    const unsigned int               mapping_degree;
    
    //Need to figure out how to properly set up Shape details
    //std::unique_ptr<Shape<dim> >                      shape;
    //Shape<dim>                      *shape;
    std::unique_ptr<Shape<dim> >     shape;
    Lift<dim> *lift;
    //Lift<dim>
    //RadialLift<dim>                  lift;
    //void& AssignPolyhedron = sphere.AssignPolyhedron;
    //Shape<dim>        shape;
    //Lift<dim>& lift = shape.lift;
    //void& AssignPolyhedron = shape.AssignPolyhedron;
};

/*
template <int dim>
SFEMAdapt(std::unique_ptr<Shape<dim> > shape_ptr, const unsigned degree, const unsigned mapping_degree)
    : fe(degree)
    , fe_mapping(FE_Q<dim-1, dim>(mapping_degree), dim)
    , dof_handler(triangulation)
    , dof_handler_mapping(triangulation)
    , mapping_degree(mapping_degree)
    //, shape(std::move(shape_ptr))
    {}
*/
//SFEMAdapt<dim>::SFEMAdapt(Shape<dim> *shape_ptr, const unsigned degree, const unsigned mapping_degree)
//SFEMAdapt(std::unique_ptr<Shape<dim>> shape_ptr, const unsigned degree, const unsigned mapping_degree)

template <int dim>
void SFEMAdapt<dim>::setup_system()
{
    // Construct the mapping first
    dof_handler_mapping.distribute_dofs(fe_mapping);
    approximate_lift.reinit(dof_handler_mapping.n_dofs());
    mapping_constraints.clear();
    DoFTools::make_hanging_node_constraints (dof_handler_mapping, mapping_constraints);
    mapping_constraints.close();
    VectorTools::interpolate(dof_handler_mapping, *lift, approximate_lift);
    mapping_constraints.distribute(approximate_lift);
    
    //Since mapping is within the scope of this method, we have to call it everywhere throughout the code. I'd rather have it as a variable of the class.
    MappingQEulerian<dim-1, Vector<double>, dim> mapping(mapping_degree, dof_handler_mapping, approximate_lift);
    
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
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ false);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
    
    //Print Sparsity pattern for viewing
    //std::ofstream out ("sparsity_pattern.svg");
    //sparsity_pattern.print_svg (out);
}


template <int dim>
void SFEMAdapt<dim>::assemble_system()
{
    //create mapping
    MappingQEulerian<dim-1, Vector<double>, dim> mapping(mapping_degree, dof_handler_mapping, approximate_lift);
    
    const QGauss<dim-1>                     quadrature_formula(3);
    
    FEValues<2,3>                           fe_values(mapping,
                                                      fe,
                                                      quadrature_formula,
                                                      update_values |
                                                      update_gradients |
                                                      update_quadrature_points |
                                                      update_JxW_values);
    
    const unsigned int                      dofs_per_cell(fe.dofs_per_cell);
    std::vector<types::global_dof_index>    local_dof_indices(dofs_per_cell);

    FullMatrix<double>                      cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>                          cell_rhs(dofs_per_cell);
    const unsigned int                      n_q_points(quadrature_formula.size());
    std::vector<double>                     rhs_values(n_q_points);
    const RightHandSide<dim>                rhs;
    

    auto cell = dof_handler.begin_active();
    auto endc = dof_handler.end();
    
    for(; cell != endc; ++cell)
    {
        cell_matrix = 0;
        cell_rhs    = 0;

        fe_values.reinit(cell);
        rhs.value_list(fe_values.get_quadrature_points(), rhs_values);
        
        for(unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        {
            const double current_coefficient = coefficient<dim>(fe_values.quadrature_point(q_index));
            for(unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for(unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    cell_matrix(i, j) += (current_coefficient * fe_values.shape_grad(i, q_index) *
                                          fe_values.shape_grad(j, q_index) * fe_values.JxW(q_index));
                }
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
void SFEMAdapt<dim>::solve()
{
    SolverControl solver_control(3000, 0.00001);
    SolverCG<>    solver(solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1); //solver very sensitive to relaxation parameter

    solver.solve(system_matrix, solution, system_rhs, preconditioner);

    constraints.distribute(solution);
}


template <int dim>
void SFEMAdapt<dim>::PDE_estimate(double &estimated_pde_error)
{
    //create mapping
    MappingQEulerian<dim-1,Vector<double >, dim> mapping(mapping_degree, dof_handler_mapping, approximate_lift);
  
    //ESTIMATE
    estimated_pde_error_per_cell.reinit(triangulation.n_active_cells());
    
    //Jump Estimator
    KellyErrorEstimator<dim-1, dim>::estimate(mapping,
                                              dof_handler,
                                              QGauss<dim - 2>(3),
                                              std::map<types::boundary_id,
                                              const Function<dim> *>(),
                                              solution,
                                              estimated_pde_error_per_cell);
    
    //Bulk Estimator
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
    std::vector<double >        rhs_at_q_points(n_q_points);
    const RightHandSide<dim>    rhs;
    double                      dValue;
    
    unsigned int                present_cell(0); // Keeps track of the number of the cell we're on
    auto                        cell(dof_handler.begin_active());
    auto                        endc(dof_handler.end());
    
    for(; cell!=endc; ++cell, ++present_cell)
    {
        fe_values.reinit(cell);
            
        //Evaluate FEM solution's Laplacian at quadrature points and write them to
        // laplacians_at_q_points vector
        fe_values.get_function_laplacians(solution, laplacians_at_q_points);
        rhs.value_list(fe_values.get_quadrature_points(), rhs_at_q_points);
            
        for(unsigned int q_point=0; q_point < n_q_points; ++q_point)
        {
            // h*|f+Delta u|
            const double current_coefficient = coefficient<dim>(fe_values.quadrature_point(q_point));
                
            dValue = cell->diameter()/fe.degree*(rhs_at_q_points[q_point] + current_coefficient*laplacians_at_q_points[q_point]);
                
            estimated_pde_error_per_cell(present_cell) += dValue*dValue*fe_values.JxW(q_point);
        }
    }
    estimated_pde_error = estimated_pde_error_per_cell.l2_norm();
}


template <int dim>
void SFEMAdapt<dim>::PDE_solve_estimate_mark_refine(double PDE_tolerance)
{
    double estimated_pde_error = 0.0;
    unsigned int count = 0;
    
    //SOLVE
    setup_system();
    assemble_system();
    solve();
    
    //ESTIMATE
    PDE_estimate(estimated_pde_error);
    
    std::cout << "Estimated Error: " << estimated_pde_error << "\n";
    output_results(count);
    
    while(estimated_pde_error > PDE_tolerance)
    {
        //MARK
        GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                          estimated_pde_error_per_cell,
                                                          0.3,
                                                          0.0);
        
        //REFINE
        triangulation.execute_coarsening_and_refinement();
        
        //SOLVE
        setup_system();
        assemble_system();
        solve();
        
        //ESTIMATE
        PDE_estimate(estimated_pde_error);
        
        //Output Info
        ++count;
        std::cout << "  Loop: " << count << "\n";
        std::cout << "      Active Cells: " << triangulation.n_active_cells() << "\n";
        std::cout << "      Estimated Error: " << estimated_pde_error << "\n";
        
        output_results(count);
        
        compute_error(); //show actual error for test case
    }
}


template <int dim>
void SFEMAdapt<dim>::GEOMETRY_estimate(double &estimated_geometric_error)
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
                                      *lift,
                                      estimated_geometric_error_per_cell,
                                      QGauss<2>(mapping_degree + 1),
                                      VectorTools::Linfty_norm);
    
    estimated_geometric_error = estimated_geometric_error_per_cell.linfty_norm();
}


template <int dim>
void SFEMAdapt<dim>::GEOMETRY_estimate_mark_refine(double GEOMETRY_tolerance)
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


template <int dim>
void SFEMAdapt<dim>::compute_error() const
{
    //create mapping
    MappingQEulerian<dim-1, Vector<double>, dim> mapping(mapping_degree, dof_handler_mapping, approximate_lift);
    
    Vector<float> difference_per_cell(triangulation.n_active_cells());
    
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      solution,
                                      Solution<dim>(),
                                      difference_per_cell,
                                      QGauss<dim-1>(2*fe.degree+1),
                                      VectorTools::H1_norm);
    
    double h1_error = VectorTools::compute_global_error(triangulation,
                                                        difference_per_cell,
                                                        VectorTools::H1_norm);
    
    std::cout << "      H1 error = " << h1_error << std::endl;
    
    difference_per_cell = 0;
    
    VectorTools::integrate_difference(mapping, dof_handler, solution,
                                      Solution<dim>(),
                                      difference_per_cell,
                                      QGauss<dim-1>(2*fe.degree+3),
                                      VectorTools::L2_norm);
    
    double L2_error = VectorTools::compute_global_error(triangulation,
                                                        difference_per_cell,
                                                        VectorTools::L2_norm);
    
    std::cout << "      L2 error = " << L2_error << std::endl;
}


template <int dim>
void SFEMAdapt<dim>::output_results(const unsigned int cycle) const
{
    //create mapping
    MappingQEulerian<dim-1,Vector<double >,dim> mapping(mapping_degree, dof_handler_mapping, approximate_lift);
    
    DataOut<dim-1, DoFHandler<dim-1, dim>> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution,
                             "solution",
                             DataOut<dim-1, DoFHandler<dim-1, dim>>::type_dof_data);
    data_out.build_patches(mapping, mapping.get_degree());
    
    std::string filename("solution-cycle-");
    filename += std::to_string(cycle);
    filename += ".vtk";
    std::ofstream output(filename);
    data_out.write_vtk(output);
}


template <int dim>
void SFEMAdapt<dim>::run_uniform_refinement()
{
    shape->AssignPolyhedron(triangulation);  //Attach Polyhedron
    
    for (unsigned int cycle = 0; cycle < 4; ++cycle)
    {
        triangulation.refine_global(1);
        setup_system();
        assemble_system();
        solve();
        
        std::cout << "Loop:                            "
        << triangulation.n_active_cells() << std::endl;
        std::cout << "   Number of active cells:       "
                  << triangulation.n_active_cells() << std::endl;

        std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
                  << std::endl;
      
        compute_error();
        output_results(cycle);
    }
}


template <int dim>
void SFEMAdapt<dim>::run_adaptive_refinement()
{
    shape->AssignPolyhedron(triangulation);  //Attach Polyhedron
    
    //std::cout << "ADAPT_SURFACE" << "\n";
    //GEOMETRY_estimate_mark_refine(0.00001);
    
    triangulation.refine_global(1);
    
    std::cout << "ADAPT_PDE" << "\n";
    PDE_solve_estimate_mark_refine(0.01);
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



//template <int dim>
//SFEMTest(std::unique_ptr<Shape<3>> shape_ptr)
//: shape(shape_ptr)
//{std::cout << "success" << std::endl; }


int main()
{
    try
    {
        
        //auto shape_ptr = std::make_unique<Sphere<3> >();
        //std::unique_ptr<Sphere<3>> shape_ptr(new Sphere);
        
        
        auto shape_ptr = std::unique_ptr<Sphere<3>>(new Sphere<3>);
        //SFEMTest<3> Test(std::move(shape_ptr), 1, 1);
        
        SFEMAdapt<3> laplace_problem_surface(std::move(shape_ptr), 1, 1);
        
        
        //Sphere<3> shape;
        //Sphere<3> *shape_ptr = &shape;
        //SFEMAdapt<3> laplace_problem_surface(shape_ptr, 1, 1);
        laplace_problem_surface.run_uniform_refinement();
        
        //SFEMAdapt<3> laplace_problem_surface_adapt(1,1);
        //laplace_problem_surface_adapt.run_adaptive_refinement();
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

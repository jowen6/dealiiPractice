//
//  main.cpp
//  
//
//  Created by Justin Owen on 3/12/19.
//

#include <stdio.h>
#include "Lifts.h" //for Lift definitions
#include "Shapes.h" //for Shape definitions
//#include "SFEMAdapt.h" //for SFEM code
#include "SFEMAdaptEigs.h" //for SFEM code
#include <deal.II/base/function.h> //for Function class
#include <deal.II/base/function_lib.h> //Function class
#include "mpi_mgr_slepc.h" //for slepc settings
#include <deal.II/base/mpi.h> //for mpi
#include <memory>


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



int main(int argc, char **argv)
{
    try
    {

        
        //auto shape_ptr = std::unique_ptr<Sphere<3>>(new Sphere<3>);
        //auto shape_ptr = std::unique_ptr<Torus<3>>(new Torus<3>);
        //laplace_problem_surface_adapt.run_adaptive_refinement();
        //RadialLift<3> lift;
        //Sphere<3> shape(lift);
        
        
        /*
        Sphere<3> shape;
        RightHandSide<3> rhs;
        Solution<3>      Solution;
        const unsigned int mapping_degree = 1;
        const unsigned int fe_degree = 1;
        std::cout << shape.GetLift() << "\n";
        
        SFEMAdapt<3> laplace_problem_surface(shape,
                                             rhs,
                                             Solution,
                                             fe_degree,
                                             mapping_degree);
        
        laplace_problem_surface.PDE_run_uniform_refinement();
        */
        
        //STEP-36
        /*
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
        AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)==1,
                    ExcMessage("This program can only be run in serial, use ./step-36"));
        */
        //double left_spectrum(1.0);
        //double right_spectrum(3.0);
        
        MPIMGRSLEPC mpi_mgr(argc, argv);
        PetscReal lower_bound = 1.0;
        PetscReal upper_bound = 3.0;
        Sphere<3> shape;
        const unsigned int mapping_degree = 1;
        const unsigned int fe_degree = 1;
        
        SFEMAdaptEigs<3> laplace_problem_surface_eigs(mpi_mgr,
                                                      shape,
                                                      lower_bound,
                                                      upper_bound,
                                                      fe_degree,
                                                      mapping_degree);
        
        laplace_problem_surface_eigs.run_uniform_refinement();
        //laplace_problem_surface_eigs.run_adaptive_refinement();
        
        /*
        SFEMAdapt<3> laplace_problem_surface(std::move(shape_ptr),
                                             rhs,
                                             Solution,
                                             1,
                                             1);
        laplace_problem_surface.PDE_run_uniform_refinement();
         */
   
        
        std::cout<< "finished" << "\n";
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

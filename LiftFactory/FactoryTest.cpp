//
//  FactoryTest.cpp
//  
//
//  Created by Justin Owen on 1/22/19.
//

#include <stdio.h>
#include "Lift.hpp"
#include <iostream>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>
//For Geometric Estimator
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_q_eulerian.h>
using namespace dealii;

int main(){
    
    LiftType type = LT_C2AlphaLift;
    //std::unique_ptr<Lift<3> > pLift = Lift<3>::Create(type);
    auto pLift = Lift<3>::Create(type);
    pLift->print_Lift();
    Vector<double> V(3);
    Point<3> P(0.5,0.5,0);
    pLift->vector_value(P, V);
    
    std::cout<< V(0) << "\n";
    std::cout<< V(1) << "\n";
    std::cout<< V(2) << "\n";
    
    
    
    //Create Triangulation
    Triangulation<2,3> triangulation;
    
    //Set triangualtion to sphere and refine mesh to make good polyhedral approximation
    GridGenerator::hyper_sphere(triangulation); //automatically attaches sphere manifold and sets all ids to 0
    triangulation.refine_global(2);
    
    //Set manifold to be polyhedral so future refinements don't add faces
    static FlatManifold<2,3> polyhedral_surface_description;
    triangulation.set_manifold(0, polyhedral_surface_description);
    triangulation.refine_global(1);
    
    
    
    //Building Approximate Lift
    unsigned int mapping_degree = 1;
    FESystem<2,3> fe_mapping;   //
    fe_mapping(FE_Q<2,3>(mapping_degree),3);
    DofHandler<2,3> dof_handler_surface_map; //mapping dof_handler for surface interpolation
    
    
    dof_handler_surface_map(*triangulation) //attach dof to triangulation
    
    dof_handler_surface_map.distribute_dofs(fe_mapping);//distribute dofs according to fe_mapping
    
    
    Vector<double> approximate_lift; //approx_lift coefficients vector
    approximate_lift.reinit(dof_handler_surface_map.n_dofs());
    
    VectorTools::interpolate(dof_handler_surface_map,*pLift,approximate_lift); //interpolate lift
    
    
    
    //Constraint Matrix for mapping
    IndexSet locally_relevant_dofs_mapping;
    ConstraintMatrix mapping_constraints;
    DoFTools::extract_locally_relevant_dofs (dof_handler_surface_map, locally_relevant_dofs_mapping);
    mapping_constraints.clear();
    mapping_constraints.reinit(locally_relevant_dofs_mapping);
    DoFTools::make_hanging_node_constraints(dof_handler_surface_map, mapping_constraints);
    mapping_constraints.distribute(approximate_lift);
    
    //MappingQEulerian works by mapping from points on the triangulation to other points via a dofHandler representing Vectors pointing from the triangulation to the new point. This is why Lift must define a Vector pointing from the triangulation to the surface rather than a Point!
    
    MappingQEulerian<2, Vector<double >, 3> mapping(mapping_degree,dof_handler_surface_map,approximate_lift);
    
    
    //Building Geometric Estimator
    Vector<float > estimator_geometry_per_cell;      // Bonito - Demlow way
    Vector<float > estimator_geometry_per_cell_BCMN; // Bonito - Cascon - Morin - Nochetto way
    
    estimator_geometry_per_cell.reinit(triangulation.n_active_cells());
    estimator_geometry_per_cell_BCMN.reinit(triangulation.n_active_cells());
    
    
    VectorTools::integrate_difference (dof_handler_surface_map, approximate_lift,
                                       lift, estimator_geometry_per_cell_BCMN,
                                       QGauss<2>(mapping_degree+1),
                                       VectorTools::W1infty_seminorm);
    
    
    VectorTools::integrate_difference (dof_handler_surface_map, approximate_lift,
                                       lift, estimator_geometry_per_cell_BD,
                                       QGauss<2>(mapping_degree+1),
                                       VectorTools::Linfty_norm);
    
    
    //Update the BD estimator with square of BCMN estimator
    for (unsigned int i =0 ; i<estimator_geometry_per_cell.size(); ++i){
        estimator_geometry_per_cell_BD(i) += estimator_geometry_per_cell_BCMN(i)*estimator_geometry_per_cell_BCMN(i);
    }
    
    
    return 0;
}

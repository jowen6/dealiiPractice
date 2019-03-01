//
//  Shapes.h
//  
//
//  Created by Justin Owen on 2/27/19.
//

#include "Lifts.h"
#include <deal.II/grid/tria.h> //Triangulation

using namespace dealii;

#ifndef Shapes_h
#define Shapes_h

//Abstract Lift Class
template <int spacedim>
class Shape
{
public:
    Shape (){};
    virtual ~Shape() {};
    virtual void AssignPolyhedron(Triangulation<spacedim-1, spacedim> &triangulation) = 0;
    virtual Lift<spacedim> *GetLift() = 0;
};


//RadialLift Sphere
template <int spacedim>
class Sphere : public Shape<spacedim>
{
public:
    Sphere() : lift(new RadialLift<spacedim>){}
    void AssignPolyhedron (Triangulation<spacedim-1, spacedim> &triangulation) override;
    RadialLift<spacedim> *GetLift() override{
        return lift;
    }
    RadialLift<spacedim> *lift;
    
};


template <int spacedim>
void Sphere<spacedim>::AssignPolyhedron(Triangulation<spacedim-1,spacedim> &triangulation)
{
    //Unit Sphere
    GridGenerator::hyper_sphere(triangulation);
    triangulation.refine_global(3);
    //Set manifold to be polyhedral so future refinements don't add polyhedral faces
    static FlatManifold<spacedim-1, spacedim> polyhedral_surface_description;
    triangulation.set_manifold(0, polyhedral_surface_description);
    
    
}

//Shapes to do

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


//Torus
/*
 GridGenerator::torus(triangulation,
 0.7,
 0.5);
 triangulation.refine_global(3);
 
 std::cout << "Surface mesh has " << triangulation.n_active_cells()
 << " cells." << std::endl;
 */
#endif /* Shapes_h */



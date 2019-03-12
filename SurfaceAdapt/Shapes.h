//
//  Shapes.h
//  
//
//  Created by Justin Owen on 2/27/19.
//

#include "Lifts.h"
#include <deal.II/grid/tria.h> //Triangulation
#include <deal.II/grid/grid_generator.h> //generate grids
#include <deal.II/grid/manifold_lib.h> //dealii SphericalManifold
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

//There is some sort of memory allocation error occuring. I think it has to do with my lift_ptr. I do not understand yet.
//RadialLift Sphere
template <int spacedim>
class Sphere : public Shape<spacedim>
{
public:
    Sphere() : lift_ptr(new RadialLift<spacedim>){}
    
    void AssignPolyhedron (Triangulation<spacedim-1, spacedim> &triangulation) override;
    
    RadialLift<spacedim> *GetLift() override{
        return lift_ptr;
    }
    
    ~Sphere(){delete lift_ptr;}
    
private:
    RadialLift<spacedim> *lift_ptr;
    //Lift<spacedim> *lift_ptr;
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


//RadialLift Half-Sphere
template <int spacedim>
class HalfSphere : public Shape<spacedim>
{
public:
    HalfSphere() : lift(new RadialLift<spacedim>){}
    
    void AssignPolyhedron (Triangulation<spacedim-1, spacedim> &triangulation) override;
    
    RadialLift<spacedim> *GetLift() override{
        return lift;
    }
    
    ~HalfSphere(){delete lift;}
    
private:
    RadialLift<spacedim> *lift;
};


template <int spacedim>
void HalfSphere<spacedim>::AssignPolyhedron(Triangulation<spacedim-1,spacedim> &triangulation)
{
    //Unit HalfSphere
    {
        Triangulation<spacedim> volume_mesh;
        GridGenerator::half_hyper_ball(volume_mesh);
        
        std::set<types::boundary_id> boundary_ids;
        boundary_ids.insert(0);
        
        GridGenerator::extract_boundary_mesh(volume_mesh,
                                             triangulation,
                                             boundary_ids);
    }
    triangulation.set_all_manifold_ids(0);
    triangulation.set_manifold(0, SphericalManifold<spacedim-1, spacedim>());
    triangulation.refine_global(3);
    //Set manifold to be polyhedral so future refinements don't add polyhedral faces
    static FlatManifold<spacedim-1, spacedim> polyhedral_surface_description;
    triangulation.set_manifold(0, polyhedral_surface_description);
}


//TorusLift Torus
template <int spacedim>
class Torus : public Shape<spacedim>
{
public:
    Torus(const double torus_circle_radius = 0.7, const double torus_inner_radius = 0.3)
    : lift(new TorusLift<spacedim>(torus_circle_radius, torus_inner_radius))
    , torus_circle_radius(torus_circle_radius)
    , torus_inner_radius(torus_inner_radius)
     {lift->print_Lift();}
    
    ~Torus(){delete lift;}
    
    void AssignPolyhedron (Triangulation<spacedim-1, spacedim> &triangulation) override;
    
    TorusLift<spacedim> *GetLift() override{
        return lift;
    }
    
private:
    TorusLift<spacedim> *lift;
    
    double torus_circle_radius;
    
    double torus_inner_radius;
};


template <int spacedim>
void Torus<spacedim>::AssignPolyhedron(Triangulation<spacedim-1,spacedim> &triangulation)
{
    //Torus
    GridGenerator::torus(triangulation,
                         torus_circle_radius,
                         torus_inner_radius);
    triangulation.refine_global(3);
    //Set manifold to be polyhedral so future refinements don't add polyhedral faces
    static FlatManifold<spacedim-1, spacedim> polyhedral_surface_description;
    triangulation.set_manifold(0, polyhedral_surface_description);
}

//Shapes to do



#endif /* Shapes_h */



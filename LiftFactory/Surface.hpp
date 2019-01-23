//
//  Surface.hpp
//  
//
//  Created by Justin Owen on 1/22/19.
//

#ifndef Surface_hpp
#define Surface_hpp

#include <stdio.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <vector>
#include "Lift.hpp"

using namespace dealii;

enum SurfaceType{
    ST_Sphere
};

//Abstract Surface Class
template <int spacedim>
class Surface{
public:
    
    Lift<spacedim>* SurfaceLift;
    virtual void Attach_Surface_To_Triangulation() = 0;
    virtual void print_Surface() = 0;
    
    static Surface* Create(SurfaceType type);
};


//Spherical surface
template <int spacedim>
class Sphere : public Surface<spacedim>{
    void Attach_Surface_To_Triangulation();
    
    void print_Lift();
};


template <int spacedim>
void Sphere::Attach_Surface_To_Triangulation();


//Surface Factory
template <int spacedim>
Surface<spacedim>* Surface<spacedim>::Create(SurfaceType type){
    if (type == ST_Sphere)
        return new Sphere<spacedim>();
    else return NULL;
}
#endif /* Surface_hpp */

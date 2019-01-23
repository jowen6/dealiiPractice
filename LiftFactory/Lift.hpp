//
//  Lift.hpp
//  
//
//  Created by Justin Owen on 1/21/19.
//

#ifndef Lift_hpp
#define Lift_hpp

#include <stdio.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <vector>
using namespace dealii;

enum LiftType{
    LT_RadialLift
};

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
    static Lift* Create(LiftType type);
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

//Define RadialLift function
template <int spacedim>
void RadialLift<spacedim>::vector_value(const Point<spacedim> &p, Vector<double> &values) const{
    double length = p.norm();
    
    for (unsigned int i=0; i<spacedim; ++i){
        values(i) = p(i)/(length-p(i));
    }
}

//Define RadialLift gradient
template <int spacedim>
void RadialLift<spacedim>::vector_gradient(const Point<spacedim> &p,
                                           std::vector<Tensor<1, spacedim, double> > &gradients) const{
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


//Define RadialLift print name
template <int spacedim>
void RadialLift<spacedim>::print_Lift(){
    std::cout<< "RadialLift" << "\n";
}


//Lift Factory
template <int spacedim>
Lift<spacedim>* Lift<spacedim>::Create(LiftType type){
    if (type == LT_RadialLift)
        return new RadialLift<spacedim>();
    else return NULL;
}

#endif /* Lift_hpp */

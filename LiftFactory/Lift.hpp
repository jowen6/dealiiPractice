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
#include <memory>
using namespace dealii;


enum LiftType{
    LT_RadialLift, LT_C2AlphaLift
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
    static std::unique_ptr<Lift<spacedim> > Create(LiftType type);
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


//Print RadialLift details
template <int spacedim>
void RadialLift<spacedim>::print_Lift(){
    std::cout<< "RadialLift given by mapping x->x/|x|" << "\n";
}
//===========================================================================


//C2AlphaLift given by max(z(x,y) = (3/4 - x^2 - y^2)^(2+alpha), 0)
template <int spacedim>
class C2AlphaLift : public Lift<spacedim>{
    void vector_value (const Point<spacedim> &p,
                       Vector<double>   &values) const;
    
    void vector_gradient(const Point<spacedim> &p,
                         std::vector<Tensor<1,spacedim,double> > &gradients) const;
    
    void print_Lift();
    
    double alpha = 2.0/5.0;
};


//Define C2AlphaLift vector pointing from point to exact surface
template <int spacedim>
void C2AlphaLift<spacedim>::vector_value(const Point<spacedim> &p, Vector<double> &values) const{
    
    values(0) = 0.0;
    values(1) = 0.0;
    values(2) = pow(std::max(0.75-p(0)*p(0)-p(1)*p(1),0.0),2.+alpha); //Maps straight up in z direction
}


//Define C2AlphaLift gradient
template <int spacedim>
void C2AlphaLift<spacedim>::vector_gradient(const Point<spacedim> &p,
                                           std::vector<Tensor<1, spacedim, double> > &gradients) const{
    // first component
    gradients[0][0] = 0.0;
    gradients[0][1] = 0.0;
    gradients[0][2] = 0.0;
    
    //second component
    gradients[1][0] = 0.0;
    gradients[1][1] = 0.0;
    gradients[1][2] = 0.0;
    
    
    double zGuts = 0.75-p(0)*p(0)-p(1)*p(1);
    
    //third component
    if (zGuts >=0 ){
        gradients[2][0] = -2.0*(2.0 + alpha)*p(0)*pow(zGuts, 1.0 + alpha);
        gradients[2][1] = -2.0*(2.0 + alpha)*p(1)*pow(zGuts, 1.0 + alpha);
        gradients[2][2] = 0.0;
    }
    else{
        gradients[2][0] = 0.0;
        gradients[2][1] = 0.0;
        gradients[2][2] = 0.0;
    }
}


//Print C2AlphaLift details
template <int spacedim>
void C2AlphaLift<spacedim>::print_Lift(){
    std::cout<< "C^(2," << alpha << ") Lift given by max(z(x,y) = (3/4 - x^2 - y^2)^(2 + " << alpha << "), 0)" << "\n";
}
//===========================================================================


//Lift Factory
template <int spacedim>
std::unique_ptr<Lift<spacedim> > Lift<spacedim>::Create(LiftType type){
    if (type == LT_RadialLift)
        return std::make_unique<RadialLift<spacedim> >(); //creates unique ptr which will auto delete once out of scope
    else if (type == LT_C2AlphaLift)
        return std::make_unique<C2AlphaLift<spacedim> >();
    else return NULL;
}

#endif /* Lift_hpp */

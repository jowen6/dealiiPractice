//
//  Lifts.h
//  
//
//  Created by Justin Owen on 2/27/19.
//
#include <deal.II/base/function_lib.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>

using namespace dealii;

#ifndef Lifts_h
#define Lifts_h


//Abstract Lift Class
template <int spacedim>
class Lift :  public Function<spacedim>
{
public:
    Lift () : Function<spacedim>(spacedim) {};
    virtual ~Lift() {};
    virtual void vector_value (const Point<spacedim> &p, Vector<double>   &values) const = 0;
    
    virtual void vector_gradient(const Point<spacedim> &p, std::vector<Tensor<1,spacedim,double> > &gradients) const = 0;
    
    virtual void print_Lift() = 0;
};


//RadialLift
template <int spacedim>
class RadialLift : public Lift<spacedim>
{
    void vector_value (const Point<spacedim> &p, Vector<double>   &values) const;
    
    void vector_gradient(const Point<spacedim> &p, std::vector<Tensor<1,spacedim,double> > &gradients) const;
    
    void print_Lift();
};


//Define RadialLift vector pointing from point to exact surface
template <int spacedim>
void RadialLift<spacedim>::vector_value(const Point<spacedim> &p, Vector<double> &values) const
{
    double length = p.norm();
    
    for (unsigned int i=0; i<spacedim; ++i)
    {
        values(i) = (p(i)/length)-p(i);
    }
}


//Define RadialLift gradient for 2D
template <>
void RadialLift<2>::vector_gradient(const Point<2> &p, std::vector<Tensor<1, 2, double> > &gradients) const
{
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
void RadialLift<3>::vector_gradient(const Point<3> &p, std::vector<Tensor<1, 3, double> > &gradients) const
{
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
void RadialLift<spacedim>::print_Lift()
{
    std::cout<< "RadialLift given by mapping x->x/|x|" << "\n";
}


#endif /* Lifts_h */

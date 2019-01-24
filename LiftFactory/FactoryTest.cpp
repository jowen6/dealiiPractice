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

using namespace dealii;

int main(){
    
    LiftType type = LT_RadialLift;
    //std::unique_ptr<Lift<3> > pLift = Lift<3>::Create(type);
    auto pLift = Lift<3>::Create(type);
    pLift->print_Lift();
    Vector<double> V(3);
    Point<3> P(1,0,0);
    pLift->vector_value(P, V);
    
    std::cout<< V(0) << "\n";
    std::cout<< V(1) << "\n";
    std::cout<< V(2) << "\n";
    
    return 0;
}

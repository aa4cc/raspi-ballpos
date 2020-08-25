#include "coope_fit.h"
#include <eigen3/Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;
// sudo apt install libeigen3-dev
// g++ coope_fit.cpp -shared -fPIC -o libcoope_fit.so

extern "C" {
Coord_t coope_fit(Coord_t *coords, int coords_l) {
  Matrix<double, Dynamic, 3> A; // = Matrix(float, coords_l, 3);
  A.resize(coords_l, 3);
  VectorXd b; // = Vector(float, coords_l);
  b.resize(coords_l);
  for (int i = 0; i < coords_l; ++i) {
    A(i, 0) = coords[i].x;
    A(i, 1) = coords[i].y;
    A(i, 2) = 1;
    b(i) = coords[i].x * coords[i].x + coords[i].y * coords[i].y;
  }
  // cout<<A<<endl;
  //   cout << A << endl;
  //   cout << b << endl;
  VectorXd sol = (A.transpose() * A).ldlt().solve(A.transpose() * b);
  // cout<<sol<<endl;
  Coord_t solution = {(float)sol(0)/2, (float)sol(1)/2};
  return solution;
}
}
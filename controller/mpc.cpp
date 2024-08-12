#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <iostream>

namespace py = pybind11;
using namespace Eigen;

MatrixXd predict_state(const MatrixXd &Ad, const MatrixXd &Bd, const VectorXd &x, const VectorXd &u) {
    return Ad * x + Bd * u;
}

double cost_function(const VectorXd &U, const MatrixXd &Ad, const MatrixXd &Bd, const MatrixXd &Q, const MatrixXd &R, const VectorXd &x0, const VectorXd &ref, int N, int nx, int nu) {
    double cost = 0.0;
    VectorXd x = x0;

    for (int i = 0; i < N; ++i) {
        VectorXd u = U.segment(i * nu, nu);
        x = predict_state(Ad, Bd, x, u);
        cost += ((x - ref).transpose() * Q * (x - ref)).value() + (u.transpose() * R * u).value();
    }
    return cost;
}

PYBIND11_MODULE(mpc_cpp, m) {
    m.def("cost_function", &cost_function, "A function that computes the cost");
}

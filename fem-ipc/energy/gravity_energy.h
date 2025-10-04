#pragma once

#include <Eigen/Core>

namespace fem_ipc {

struct GravityPotential {
    Eigen::Vector3d gravity = Eigen::Vector3d(0.0, -9.81, 0.0);

    GravityPotential() = default;
    GravityPotential(const Eigen::Vector3d& g) : gravity(g) { }

    double value(
        const Eigen::MatrixX3d& x, 
        const Eigen::VectorXd& masses
    ) const {
        double energy = 0.0;
        for (Eigen::Index i = 0; i < x.rows(); ++i) {
            energy -= masses[i] * gravity.dot(x.row(i).transpose());
        }
        return energy;
    }

    Eigen::VectorXd gradient(
        const Eigen::MatrixX3d& x,
        const Eigen::VectorXd& masses
    ) const {
        Eigen::VectorXd grad(x.rows() * 3);
        for (Eigen::Index i = 0; i < x.rows(); ++i) {
            grad.segment<3>(3 * i) = (-masses[i] * gravity).transpose();
        }
        return grad;
    }
};

} // namespace fem_ipc

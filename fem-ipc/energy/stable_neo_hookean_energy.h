#pragma once

#include <Eigen/Core>

namespace fem_ipc {

struct StableNeoHookeanEnergy {
    double mu, lambda;

    StableNeoHookeanEnergy(double mu, double lambda) : mu(mu), lambda(lambda) {}

    double E(const Eigen::Matrix3d& F);

    Eigen::Matrix3d dEdVecF(const Eigen::Matrix3d& F);

    Eigen::Matrix<double, 9, 9> ddEddVecF(const Eigen::Matrix3d& F);

    double valuePerTet(
        const Eigen::Matrix<double, 3, 4>& x,
        const Eigen::Matrix3d& Dm_inv,
        double volume);

    Eigen::Vector<double, 12> gradientPerTet(
        const Eigen::Matrix<double, 3, 4>& x,
        const Eigen::Matrix3d& Dm_inv,
        double volume);

    Eigen::Matrix<double, 12, 12> hessianPerTet(
        const Eigen::Matrix<double, 3, 4>& x,
        const Eigen::Matrix3d& Dm_inv,
        double volume);
};

} // namespace fem_ipc


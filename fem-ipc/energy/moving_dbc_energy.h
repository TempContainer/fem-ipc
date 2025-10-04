#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace fem_ipc {

struct MovingDBCEnergy {
    double DBC_stiffness = 1e3;
    double m_DBC = 1.0;

    MovingDBCEnergy() = default;
    MovingDBCEnergy(double stiffness, double m_DBC) : 
        DBC_stiffness(stiffness), m_DBC(m_DBC) {}

    double value(
        const Eigen::MatrixX3d& x_DBC,
        const Eigen::MatrixX3d& x_DBC_target
    ) const {
        double energy = 0.0;
        const Eigen::Index num_dbc = x_DBC.rows();
        for (Eigen::Index i = 0; i < num_dbc; ++i) {
            energy += 0.5 * DBC_stiffness * m_DBC * 
                (x_DBC.row(i) - x_DBC_target.row(i)).squaredNorm();
        }
        return energy;
    }

    Eigen::VectorXd gradient(
        const Eigen::MatrixX3d& x_DBC,
        const Eigen::MatrixX3d& x_DBC_target
    ) const {
        const Eigen::Index num_dbc = x_DBC.rows();
        Eigen::VectorXd grad = Eigen::VectorXd::Zero(num_dbc * 3);
        for (Eigen::Index i = 0; i < num_dbc; ++i) {
            grad.segment<3>(3 * i) += DBC_stiffness * m_DBC * 
                (x_DBC.row(i) - x_DBC_target.row(i));
        }
        return grad;
    }

    void hessian(
        const int n_DOF,
        const Eigen::MatrixX3d& x_DBC,
        std::vector<Eigen::Triplet<double>>& triplets
    ) const {
        const Eigen::Index num_dbc = x_DBC.rows();
        triplets.reserve(triplets.size() + static_cast<size_t>(num_dbc) * 3);

        for (Eigen::Index i = 0; i < num_dbc; ++i) {
            for (int d = 0; d < 3; ++d) {
                const int row = static_cast<int>(3 * (n_DOF + i) + d);
                triplets.emplace_back(row, row, DBC_stiffness * m_DBC);
            }
        }
    }
};

} // namespace fem_ipc
#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace fem_ipc {

struct Inertia {
    Inertia() = default;

    double value(
        const Eigen::MatrixX3d& x,
        const Eigen::MatrixX3d& x_tilde,
        const Eigen::VectorXd& masses
    ) const {
        double energy = 0.0;
        for (Eigen::Index i = 0; i < x.rows(); ++i) {
            const Eigen::RowVector3d diff = x.row(i) - x_tilde.row(i);
            energy += 0.5 * masses[i] * diff.squaredNorm();
        }
        return energy;
    }

    Eigen::VectorXd gradient(
        const Eigen::MatrixX3d& x,
        const Eigen::MatrixX3d& x_tilde,
        const Eigen::VectorXd& masses
    ) const {
        Eigen::VectorXd grad(x.rows() * 3);
        for (Eigen::Index i = 0; i < x.rows(); ++i) {
            grad.segment<3>(3 * i) = masses[i] * (x.row(i) - x_tilde.row(i));
        }
        return grad;
    }

    void hessian(
        const Eigen::MatrixX3d& x,
        const Eigen::VectorXd& masses,
        std::vector<Eigen::Triplet<double>>& triplets
    ) const {
        const Eigen::Index n = x.rows();
        triplets.reserve(triplets.size() + static_cast<size_t>(n) * 3);

        for (Eigen::Index i = 0; i < n; ++i) {
            for (int d = 0; d < 3; ++d) {
                const int row = static_cast<int>(3 * i + d);
                triplets.emplace_back(row, row, masses[i]);
            }
        }
    }
};

} // namespace fem_ipc

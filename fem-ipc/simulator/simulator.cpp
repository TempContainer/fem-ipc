#include "simulator.h"
#include <fem-ipc/energy/energy.h>

#include <Eigen/Sparse>
#include <spdlog/spdlog.h>

namespace fem_ipc {

Simulator::Simulator() = default;
Simulator::Simulator(const TetMesh& mesh_in) : mesh(mesh_in) {}
Simulator::~Simulator() = default;

void Simulator::initialize()
{
    mesh.computeMassAndRestVolume();
    gravityPotential = std::make_unique<GravityPotential>();
    inertia = std::make_unique<Inertia>();
    const double mu = mesh.youngs_modulus / (2 * (1 + mesh.poisson_ratio));
    const double lambda = mesh.youngs_modulus * mesh.poisson_ratio / ((1 + mesh.poisson_ratio) * (1 - 2 * mesh.poisson_ratio));
    neoHookean = std::make_unique<StableNeoHookeanEnergy>(mu, lambda);
    barrier = std::make_unique<BarrierEnergy>();
    movingDBC = std::make_unique<MovingDBCEnergy>();

    velocity = Eigen::MatrixX3d::Zero(mesh.V.rows(), 3);

    ceilPos = Eigen::Vector3d(0.0, 1.0, 0.0);
    ceilTarget = Eigen::Vector3d(0.0, -0.5, 0.0);
    ceilVelocity = Eigen::Vector3d(0.0, -1.0, 0.0);
}

void Simulator::step()
{
    Eigen::MatrixX3d x_tilde = mesh.V + dt * velocity;
    Eigen::MatrixX3d x_n = mesh.V;
    const double invDt = 1.0 / dt;
    if (ceilPos(1) > ceilTarget(1)) {
        ceilPos += dt * ceilVelocity;
        if (ceilPos(1) < ceilTarget(1)) {
            ceilPos(1) = ceilTarget(1);
        }
    }
    
    barrier->buildCollisions(mesh.V, mesh.F, mesh.V_F, mesh.E_F, mesh.E_F_restLength, 0.0, ceilPos(1));

    while (true) {
        Eigen::VectorXd dir = computeSearchDirection(x_tilde);
        const double dir_inf_norm = dir.cwiseAbs().maxCoeff();
        if (dir_inf_norm <= tol && !DBC_satisfied) {
            movingDBC->DBC_stiffness *= 2.0;
        }
        double energyLast = totalEnergyValue(x_tilde);

        double alpha = barrier->computeStepSize(
                mesh.V, mesh.F, mesh.V_F, mesh.E_F, dir, 0.0, ceilPos(1));

        Eigen::MatrixX3d x_0 = mesh.V;
        while(true) {
            mesh.V = x_0 + alpha * Eigen::Map<const Eigen::MatrixX3d>(dir.data(), mesh.V.rows(), 3);
            barrier->is_built = false;
            double energyNew = totalEnergyValue(x_tilde);
            if (energyNew > energyLast) {
                alpha *= 0.5;
            } else break;
        }

        if (dir_inf_norm < tol && DBC_satisfied) break;
    }

    velocity = (mesh.V - x_n) * invDt;
}

Eigen::VectorXd Simulator::computeSearchDirection(
    const Eigen::MatrixX3d& x_tilde
) {
    DBC_satisfied = false;
    if (std::abs(ceilPos(1) - ceilTarget(1)) / dt < tol) {
        DBC_satisfied = true;
    }

    Eigen::SparseMatrix<double> hess = totalEnergyHessian(x_tilde);
    Eigen::VectorXd grad = totalEnergyGradient(x_tilde);
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;

    solver.compute(hess);
    if (solver.info() != Eigen::Success) {
        spdlog::error("Failed to decompose the Hessian matrix!");
        std::abort();
    }

    Eigen::VectorXd dir = solver.solve(-grad);
    if (solver.info() != Eigen::Success) {
        spdlog::error("Failed to solve for the search direction!");
        std::abort();
    }

    return dir;
}

double Simulator::totalEnergyValue(
    const Eigen::MatrixX3d& x_tilde
) {
    double energy = 0.0;
    const double dt_sq = dt * dt;

    energy += inertia->value(mesh.V, x_tilde, mesh.masses);
    energy += dt_sq * gravityPotential->value(mesh.V, mesh.masses);
    energy += dt_sq * barrier->value(mesh.V, mesh.F, mesh.V_F, mesh.E_F, mesh.E_F_restLength, 0.0, ceilPos(1));
    const Eigen::Matrix<double, Eigen::Dynamic, 3> ceilPosMat = ceilPos.transpose();
    const Eigen::Matrix<double, Eigen::Dynamic, 3> ceilTargetMat = ceilTarget.transpose();
    energy += movingDBC->value(ceilPosMat, ceilTargetMat);

    for (int i = 0; i < mesh.T.rows(); ++i) {
        Eigen::Matrix<double, 3, 4> tet_x;
        tet_x.col(0) = mesh.V.row(mesh.T(i, 0));
        tet_x.col(1) = mesh.V.row(mesh.T(i, 1));
        tet_x.col(2) = mesh.V.row(mesh.T(i, 2));
        tet_x.col(3) = mesh.V.row(mesh.T(i, 3));

        energy += dt_sq * neoHookean->valuePerTet(tet_x, mesh.inv_Dm[i], mesh.volumes(i));
    }
    return energy;
}

Eigen::VectorXd Simulator::totalEnergyGradient(
    const Eigen::MatrixX3d& x_tilde
) {
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(3 * (mesh.V.rows() + 1));
    const double dt_sq = dt * dt;

    grad.head(3 * mesh.V.rows()) += inertia->gradient(mesh.V, x_tilde, mesh.masses);
    grad.head(3 * mesh.V.rows()) += dt_sq * gravityPotential->gradient(mesh.V, mesh.masses);
    grad += dt_sq * barrier->gradient(mesh.V, mesh.F, mesh.V_F, mesh.E_F, 0.0, ceilPos(1));
    const Eigen::Matrix<double, Eigen::Dynamic, 3> ceilPosMat = ceilPos.transpose();
    const Eigen::Matrix<double, Eigen::Dynamic, 3> ceilTargetMat = ceilTarget.transpose();
    if (!DBC_satisfied) {
        const auto dbc_grad = movingDBC->gradient(ceilPosMat, ceilTargetMat);
        grad.tail<3>() += dbc_grad;
    }

    for (int i = 0; i < mesh.T.rows(); ++i) {
        Eigen::Matrix<double, 3, 4> tet_x;
        tet_x.col(0) = mesh.V.row(mesh.T(i, 0));
        tet_x.col(1) = mesh.V.row(mesh.T(i, 1));
        tet_x.col(2) = mesh.V.row(mesh.T(i, 2));
        tet_x.col(3) = mesh.V.row(mesh.T(i, 3));

        auto tet_grad = dt_sq * neoHookean->gradientPerTet(tet_x, mesh.inv_Dm[i], mesh.volumes(i));

        for (int j = 0; j < 4; ++j) {
            grad.segment<3>(3 * mesh.T(i, j)) += tet_grad.segment<3>(3 * j);
        }
    }

    return grad;
}

Eigen::SparseMatrix<double> Simulator::totalEnergyHessian(
    const Eigen::MatrixX3d& x_tilde
) {
    std::vector<Eigen::Triplet<double>> triplets;
    const double dt_sq = dt * dt;

    inertia->hessian(mesh.V, mesh.masses, triplets);
    barrier->hessian(mesh.V, mesh.F, mesh.V_F, mesh.E_F, 0.0, ceilPos(1), dt_sq, triplets);
    if (!DBC_satisfied) {
    const Eigen::Matrix<double, Eigen::Dynamic, 3> ceilPosMat = ceilPos.transpose();
    movingDBC->hessian(mesh.V.rows(), ceilPosMat, triplets);
    } else {
        triplets.emplace_back(3 * mesh.V.rows(), 3 * mesh.V.rows(), 1.0);
        triplets.emplace_back(3 * mesh.V.rows() + 1, 3 * mesh.V.rows() + 1, 1.0);
        triplets.emplace_back(3 * mesh.V.rows() + 2, 3 * mesh.V.rows() + 2, 1.0);
    }

    for (int i = 0; i < mesh.T.rows(); ++i) {
        Eigen::Matrix<double, 3, 4> tet_x;
        tet_x.col(0) = mesh.V.row(mesh.T(i, 0));
        tet_x.col(1) = mesh.V.row(mesh.T(i, 1));
        tet_x.col(2) = mesh.V.row(mesh.T(i, 2));
        tet_x.col(3) = mesh.V.row(mesh.T(i, 3));

        auto neoHookeanHess = dt_sq * neoHookean->hessianPerTet(tet_x, mesh.inv_Dm[i], mesh.volumes(i));

        for (int a = 0; a < 4; ++a) {
            int va = mesh.T(i, a);
            for (int b = 0; b < 4; ++b) {
                int vb = mesh.T(i, b);
                for (int d1 = 0; d1 < 3; ++d1) {
                    for (int d2 = 0; d2 < 3; ++d2) {
                        triplets.emplace_back(
                            3 * va + d1, 3 * vb + d2,
                            neoHookeanHess(3 * a + d1, 3 * b + d2));
                    }
                }
            }
        }
    }

    Eigen::SparseMatrix<double> hess(3 * (mesh.V.rows() + 1), 3 * (mesh.V.rows() + 1));
    hess.setFromTriplets(triplets.begin(), triplets.end());
    return hess;
}

} // namespace fem_ipc

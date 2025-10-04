#pragma once

#include <fem-ipc/mesh/tet_mesh.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace fem_ipc {

struct GravityPotential;
struct Inertia;
struct StableNeoHookeanEnergy;
struct BarrierEnergy;
struct MovingDBCEnergy;

struct Simulator {
    double dt = 0.01;           // timestep size
    int maxIters = 20;          // max iterations per step
    double tol = 1e-3;          // convergence tolerance
    bool DBC_satisfied = false;

    TetMesh mesh;
    Eigen::MatrixX3d velocity;

    Eigen::Vector3d ceilPos;
    Eigen::Vector3d ceilTarget;
    Eigen::Vector3d ceilVelocity;

    // energy
    std::unique_ptr<GravityPotential> gravityPotential;
    std::unique_ptr<Inertia> inertia;
    std::unique_ptr<StableNeoHookeanEnergy> neoHookean;
    std::unique_ptr<BarrierEnergy> barrier;
    std::unique_ptr<MovingDBCEnergy> movingDBC;

    Simulator();
    Simulator(const TetMesh& mesh_in);
    ~Simulator();

    void initialize();

    void step();

    Eigen::VectorXd computeSearchDirection(
        const Eigen::MatrixX3d& x_tilde
    );

    double totalEnergyValue(
        const Eigen::MatrixX3d& x_tilde
    );

    Eigen::VectorXd totalEnergyGradient(
        const Eigen::MatrixX3d& x_tilde
    );

    Eigen::SparseMatrix<double> totalEnergyHessian(
        const Eigen::MatrixX3d& x_tilde
    );
};

} // namespace fem_ipc

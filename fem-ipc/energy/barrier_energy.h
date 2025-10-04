#pragma once

#include <Eigen/Sparse>

namespace fem_ipc {

struct BarrierEnergy {
    double dhat = 1e-2; // barrier active threshold
    double kappa = 1e5; // barrier stiffness
    bool is_built = false;

    std::vector<class EdgeEdgeCollision> edgeCollisions;
    std::vector<class VertexFaceCollision> vertexFaceCollisions;
    std::vector<int> floorCollisions;
    std::vector<int> DBC_Collisions;

    BarrierEnergy();
    BarrierEnergy(double dhat_in, double kappa_in);
    ~BarrierEnergy();

    void buildCollisions(
        const Eigen::MatrixX3d& V, const Eigen::MatrixX3i& F,
        const Eigen::VectorXi& V_F, const Eigen::MatrixX2i& E_F,
        const Eigen::VectorXd& E_F_restLength,
        const double floorHeight,
        const double ceilHeight
    );

    double computeStepSize(
        const Eigen::MatrixX3d& V, const Eigen::MatrixX3i& F,
        const Eigen::VectorXi& V_F, const Eigen::MatrixX2i& E_F,
        const Eigen::VectorXd& searchDir,
        const double floorHeight,
        const double ceilHeight
    );

    double value(
        const Eigen::MatrixX3d& V, const Eigen::MatrixX3i& F,
        const Eigen::VectorXi& V_F, const Eigen::MatrixX2i& E_F,
        const Eigen::VectorXd& E_F_restLength,
        const double floorHeight,
        const double ceilHeight
    );

    Eigen::VectorXd gradient(
        const Eigen::MatrixX3d& V, const Eigen::MatrixX3i& F,
        const Eigen::VectorXi& V_F, const Eigen::MatrixX2i& E_F,
        const double floorHeight,
        const double ceilHeight
    );

    void hessian(
        const Eigen::MatrixX3d& V, const Eigen::MatrixX3i& F,
        const Eigen::VectorXi& V_F, const Eigen::MatrixX2i& E_F,
        const double floorHeight,
        const double ceilHeight,
        const double dt_sq,
        std::vector<Eigen::Triplet<double>>& triplets
    );
};

} // namespace fem_ipc
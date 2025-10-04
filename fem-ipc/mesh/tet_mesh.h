#pragma once

#include <Eigen/Dense>

namespace fem_ipc {

class TetMesh {
public:
    Eigen::MatrixX3d V;   // vertex positions
    Eigen::MatrixX4i T;   // tetrahedra
    Eigen::MatrixX3i F;   // surface faces
    Eigen::VectorXi V_F;  // vertex indices of surface faces
    Eigen::MatrixX2i E_F; // edge indices of surface faces
    Eigen::VectorXd E_F_restLength; // rest lengths of surface edges

    Eigen::VectorXd masses; // per-vertex masses
    std::vector<Eigen::Matrix3d> inv_Dm; // per-tet inverse rest edge matrices
    Eigen::VectorXd volumes; // per-tet rest volumes
    double rho = 1000.0; // density
    double youngs_modulus = 1e5; // Young's modulus
    double poisson_ratio = 0.4; // Poisson's ratio

    TetMesh() = default;

    TetMesh(const Eigen::MatrixX3d& V_in, const Eigen::MatrixX4i& T_in, const Eigen::MatrixX3i& F_in)
        : V(V_in), T(T_in), F(F_in) { }

    size_t numVertices() const { return V.rows(); }
    size_t numTets() const { return T.rows(); }
    size_t numFaces() const { return F.rows(); }

    void computeSurfaceVertexAndEdgeIndices();

    void computeMassAndRestVolume();
};

} // namespace fem_ipc
#include "barrier_energy.h"
#include "fem_utils.h"
#include <fem-ipc/geometry/edge_edge_collision.h>
#include <fem-ipc/geometry/vertex_face_collision.h>
#include <spdlog/spdlog.h>

namespace fem_ipc {

static double barrier(double d, double dhat) {
    if (d >= dhat) return 0.0;
    const double d_minus_dhat = d - dhat;
    return -d_minus_dhat * d_minus_dhat * std::log(d / dhat);
}

static double d_barrier(double d, double dhat) {
    if (d >= dhat) { return 0.0; }
    return (dhat - d) * (2 * std::log(d / dhat) - dhat / d + 1);
}

static double d2_barrier(double d, double dhat) {
    if (d >= dhat) { return 0.0; }
    const double dhat_over_d = dhat / d;
    return (dhat_over_d + 2) * dhat_over_d - 2 * std::log(d / dhat) - 3;
}

static double barrier_sq(const double dist_squared, const double dhat, const double dmin = 0) {
    return barrier(dist_squared - dmin * dmin, 2 * dmin * dhat + dhat * dhat);
}

static double d_barrier_sq(const double dist_squared, const double dhat, const double dmin = 0) {
    return d_barrier(dist_squared - dmin * dmin, 2 * dmin * dhat + dhat * dhat);
}

static double d2_barrier_sq(const double dist_squared, const double dhat, const double dmin = 0) {
    return d2_barrier(dist_squared - dmin * dmin, 2 * dmin * dhat + dhat * dhat);
}

BarrierEnergy::BarrierEnergy() = default;

BarrierEnergy::BarrierEnergy(double dhat_in, double kappa_in)
    : dhat(dhat_in), kappa(kappa_in), is_built(false) {}

BarrierEnergy::~BarrierEnergy() = default;

void BarrierEnergy::buildCollisions(
    const Eigen::MatrixX3d& V, const Eigen::MatrixX3i& F,
    const Eigen::VectorXi& V_F, const Eigen::MatrixX2i& E_F,
    const Eigen::VectorXd& E_F_restLength,
    const double floorHeight,
    const double ceilHeight
) {
    vertexFaceCollisions.clear();
    floorCollisions.clear();
    DBC_Collisions.clear();
    for (int i = 0; i < V_F.size(); ++i) {
        Eigen::Vector3d v = V.row(V_F[i]);
        // vertex-face
        for (int j = 0; j < F.rows(); ++j) {
            int t0 = F(j, 0), t1 = F(j, 1), t2 = F(j, 2);
            if (t0 != V_F[i] && t1 != V_F[i] && t2 != V_F[i]) {
                double dist_sq = point_triangle_distance(
                    v, V.row(t0), V.row(t1), V.row(t2),
                    PointTriangleDistType::AUTO);
                if (dist_sq < dhat * dhat) {
                    vertexFaceCollisions.emplace_back(j, i);
                }
            }
        }
        // vertex-floor
        if (v.y() - floorHeight < dhat) {
            floorCollisions.push_back(i);
        }
        // vertex-ceil
        if (ceilHeight - v.y() < dhat) {
            DBC_Collisions.push_back(i);
        }
    }
    
    // edge-edge
    edgeCollisions.clear();
    for (int i = 0; i < E_F.rows(); ++i) {
        int ea0 = E_F(i, 0), ea1 = E_F(i, 1);
        for (int j = 0; j < E_F.rows(); ++j) {
            int eb0 = E_F(j, 0), eb1 = E_F(j, 1);
            if (ea0 != eb0 && ea0 != eb1 && ea1 != eb0 && ea1 != eb1) {
                double dist_sq = edge_edge_distance(
                    V.row(ea0), V.row(ea1), V.row(eb0), V.row(eb1),
                    EdgeEdgeDistType::AUTO);
                    
                if (dist_sq < dhat * dhat) {
                    edgeCollisions.emplace_back(i, j);
                    edgeCollisions.back().eps = 1e-3 * E_F_restLength[i] * E_F_restLength[j];
                }
            }
        }
    }
        
    is_built = true;
}

double BarrierEnergy::computeStepSize(
    const Eigen::MatrixX3d& V, const Eigen::MatrixX3i& F,
    const Eigen::VectorXi& V_F, const Eigen::MatrixX2i& E_F,
    const Eigen::VectorXd& searchDir,
    const double floorHeight,
    const double ceilHeight
) {
    const Eigen::MatrixX3d V1 = V + searchDir.head(V.rows() * 3).reshaped(V.rows(), 3);
    double toi = 1.0;

    for (int i = 0; i < V_F.size(); ++i) {
        int vi = V_F[i];
        // floor
        if (searchDir(3 * vi + 1) < 0) {
            toi = std::min(toi, 0.9 * (V(vi, 1) - floorHeight) / -searchDir(3 * vi + 1));
        }
        // ceil
        double relVelocity = searchDir(3 * vi + 1) - searchDir(3 * V.rows() + 1);
        if (relVelocity > 0) {
            toi = std::min(toi, 0.9 * (ceilHeight - V(vi, 1)) / relVelocity);
        }
        // vertex-face
        for (int j = 0; j < F.rows(); ++j) {
            int t0 = F(j, 0), t1 = F(j, 1), t2 = F(j, 2);
            if (t0 != vi && t1 != vi && t2 != vi) {
                Eigen::Matrix<double, 3, 4> pos0, pos1;
                pos0.col(0) = V.row(vi);
                pos0.col(1) = V.row(t0);
                pos0.col(2) = V.row(t1);
                pos0.col(3) = V.row(t2);
                pos1.col(0) = V1.row(vi);
                pos1.col(1) = V1.row(t0);
                pos1.col(2) = V1.row(t1);
                pos1.col(3) = V1.row(t2);

                VertexFaceCollision vf(j, vi);
                if (vf.isBroadIntersect(pos0, pos1, dhat / 2.0)) {
                    toi = std::min(
                        toi, vf.compute_accd_timestep(pos0, pos1, toi));
                }
            }
        }
    }

    // edge-edge
    for (int i = 0; i < E_F.rows(); ++i) {
        int ea0 = E_F(i, 0), ea1 = E_F(i, 1);
        for (int j = 0; j < E_F.rows(); ++j) {
            int eb0 = E_F(j, 0), eb1 = E_F(j, 1);
            if (ea0 != eb0 && ea0 != eb1 && ea1 != eb0 && ea1 != eb1) {
                Eigen::Matrix<double, 3, 4> pos0, pos1;
                pos0.col(0) = V.row(ea0);
                pos0.col(1) = V.row(ea1);
                pos0.col(2) = V.row(eb0);
                pos0.col(3) = V.row(eb1);
                pos1.col(0) = V1.row(ea0);
                pos1.col(1) = V1.row(ea1);
                pos1.col(2) = V1.row(eb0);
                pos1.col(3) = V1.row(eb1);

                EdgeEdgeCollision ee(i, j);
                if (ee.isBroadIntersect(pos0, pos1, dhat / 2.0)) {
                    toi = std::min(
                        toi, ee.compute_accd_timestep(pos0, pos1, toi));
                }
            }
        }
    }

    return toi;
}

double BarrierEnergy::value(
    const Eigen::MatrixX3d& V, const Eigen::MatrixX3i& F,
    const Eigen::VectorXi& V_F, const Eigen::MatrixX2i& E_F,
    const Eigen::VectorXd& E_F_restLength,
    const double floorHeight,
    const double ceilHeight
) {
    if (!is_built) {
        buildCollisions(V, F, V_F, E_F, E_F_restLength, floorHeight, ceilHeight);
    }
    double ee_energy = 0.0, vf_energy = 0.0, bc_energy = 0.0;

    for (const auto& ee : edgeCollisions) {
        Eigen::Matrix<double, 3, 4> pos;
        pos.col(0) = V.row(E_F(ee.edge0_idx, 0));
        pos.col(1) = V.row(E_F(ee.edge0_idx, 1));
        pos.col(2) = V.row(E_F(ee.edge1_idx, 0));
        pos.col(3) = V.row(E_F(ee.edge1_idx, 1));
        const double dist_sq = ee.distance(pos);
        ee_energy += ee.mollifier(pos, ee.eps) * barrier_sq(dist_sq, dhat);
    }

    for (const auto& vf : vertexFaceCollisions) {
        Eigen::Matrix<double, 3, 4> pos;
        pos.col(0) = V.row(V_F(vf.vertex_idx));
        pos.col(1) = V.row(F(vf.face_idx, 0));
        pos.col(2) = V.row(F(vf.face_idx, 1));
        pos.col(3) = V.row(F(vf.face_idx, 2));
        const double dist_sq = vf.distance(pos);
        vf_energy += barrier_sq(dist_sq, dhat);
    }

    for (const auto& f : floorCollisions) {
        double d = V(V_F[f], 1) - floorHeight;
        bc_energy += barrier_sq(d * d, dhat);
    }

    for (const auto& f : DBC_Collisions) {
        double d = ceilHeight - V(V_F[f], 1);
        bc_energy += barrier_sq(d * d, dhat);
    }

    return kappa * (ee_energy + vf_energy + bc_energy);
}

Eigen::VectorXd BarrierEnergy::gradient(
    const Eigen::MatrixX3d& V, const Eigen::MatrixX3i& F,
    const Eigen::VectorXi& V_F, const Eigen::MatrixX2i& E_F,
    const double floorHeight,
    const double ceilHeight
) {
    if (!is_built) {
        spdlog::error("buildCollisions() must be called beforehand!");
        std::abort();
    }
    // DOF of moving ceil
    const int num_vertices = static_cast<int>(V.rows());
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(3 * (num_vertices + 1));

    for (const auto& ee : edgeCollisions) {
        Eigen::Matrix<double, 3, 4> pos;
        pos.col(0) = V.row(E_F(ee.edge0_idx, 0));
        pos.col(1) = V.row(E_F(ee.edge0_idx, 1));
        pos.col(2) = V.row(E_F(ee.edge1_idx, 0));
        pos.col(3) = V.row(E_F(ee.edge1_idx, 1));

        const double dist_sq = ee.distance(pos);
        const Eigen::Matrix<double, 3, 4> d_grad = ee.distance_grad(pos);
        const double f = barrier_sq(dist_sq, dhat);
        const double f_grad = d_barrier_sq(dist_sq, dhat);
        const double m = ee.mollifier(pos, ee.eps);
        const Eigen::Matrix<double, 3, 4> m_grad = ee.mollifier_grad(pos, ee.eps);
        const auto ee_grad = f * m_grad + m * f_grad * d_grad;

        for (int i = 0; i < 4; ++i) {
            int vertex_idx;
            if (i < 2) vertex_idx = E_F(ee.edge0_idx, i);
            else vertex_idx = E_F(ee.edge1_idx, i - 2);
            grad.segment<3>(3 * vertex_idx) += kappa * ee_grad.col(i);
        }
    }

    for (const auto& vf : vertexFaceCollisions) {
        Eigen::Matrix<double, 3, 4> pos;
        pos.col(0) = V.row(V_F(vf.vertex_idx));
        pos.col(1) = V.row(F(vf.face_idx, 0));
        pos.col(2) = V.row(F(vf.face_idx, 1));
        pos.col(3) = V.row(F(vf.face_idx, 2));

        const double dist_sq = vf.distance(pos);
        const Eigen::Matrix<double, 3, 4> d_grad = vf.distance_grad(pos);
        const double f_grad = d_barrier_sq(dist_sq, dhat);
        const auto vf_grad = f_grad * d_grad;

        for (int i = 0; i < 4; ++i) {
            int vertex_idx;
            if (i == 0) vertex_idx = V_F(vf.vertex_idx);
            else vertex_idx = F(vf.face_idx, i - 1);
            grad.segment<3>(3 * vertex_idx) += kappa * vf_grad.col(i);
        }
    }

    for (const auto& f : floorCollisions) {
        double d = V(V_F[f], 1) - floorHeight;
        const double f_grad = d_barrier_sq(d * d, dhat);
        const double bc_grad = 2 * d * f_grad;
        grad(3 * V_F[f] + 1) += kappa * bc_grad;
    }

    for (const auto& f : DBC_Collisions) {
        double d = ceilHeight - V(V_F[f], 1);
        const double f_grad = d_barrier_sq(d * d, dhat);
        const double bc_grad = -2 * d * f_grad;
        grad(3 * V_F[f] + 1) += kappa * bc_grad;
        // moving ceil
        grad(3 * num_vertices + 1) -= kappa * bc_grad;
    }

    return grad;
}

void BarrierEnergy::hessian(
    const Eigen::MatrixX3d& V, const Eigen::MatrixX3i& F,
    const Eigen::VectorXi& V_F, const Eigen::MatrixX2i& E_F,
    const double floorHeight,
    const double ceilHeight,
    const double dt_sq,
    std::vector<Eigen::Triplet<double>>& triplets
) {
    if (!is_built) {
        spdlog::error("buildCollisions() must be called beforehand!");
        std::abort();
    }

    for (const auto& ee : edgeCollisions) {
        Eigen::Matrix<double, 3, 4> pos;
        pos.col(0) = V.row(E_F(ee.edge0_idx, 0));
        pos.col(1) = V.row(E_F(ee.edge0_idx, 1));
        pos.col(2) = V.row(E_F(ee.edge1_idx, 0));
        pos.col(3) = V.row(E_F(ee.edge1_idx, 1));

        const double dist_sq = ee.distance(pos);
        const Eigen::Matrix<double, 3, 4> d_grad = ee.distance_grad(pos);
        const Eigen::Matrix<double, 12, 12> d_hess = ee.distance_hess(pos);
        const double f = barrier_sq(dist_sq, dhat);
        const double f_grad = d_barrier_sq(dist_sq, dhat);
        const double f_hess = d2_barrier_sq(dist_sq, dhat);
        const double m = ee.mollifier(pos, ee.eps);
        const Eigen::Matrix<double, 3, 4> m_grad = ee.mollifier_grad(pos, ee.eps);
        const Eigen::Matrix<double, 12, 12> m_hess = ee.mollifier_hess(pos, ee.eps);

        Eigen::MatrixXd ee_hess =
            f * m_hess + 
            f_grad * (
                d_grad.reshaped() * m_grad.reshaped().transpose() +
                m_grad.reshaped() * d_grad.reshaped().transpose()) + 
            m * f_hess * d_grad.reshaped() * d_grad.reshaped().transpose() +
            m * f_grad * d_hess;
        utils::makeSPD(ee_hess);

        for (int i = 0; i < 4; ++i) {
            int vi;
            if (i < 2) vi = E_F(ee.edge0_idx, i);
            else vi = E_F(ee.edge1_idx, i - 2);
            for (int j = 0; j < 4; ++j) {
                int vj;
                if (j < 2) vj = E_F(ee.edge0_idx, j);
                else vj = E_F(ee.edge1_idx, j - 2);
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        triplets.emplace_back(
                            3 * vi + a, 3 * vj + b,
                            dt_sq * kappa * ee_hess(3 * i + a, 3 * j + b));
                    }
                }
            }
        }
    }

    for (const auto& vf : vertexFaceCollisions) {
        Eigen::Matrix<double, 3, 4> pos;
        pos.col(0) = V.row(V_F(vf.vertex_idx));
        pos.col(1) = V.row(F(vf.face_idx, 0));
        pos.col(2) = V.row(F(vf.face_idx, 1));
        pos.col(3) = V.row(F(vf.face_idx, 2));

        const double dist_sq = vf.distance(pos);
        const Eigen::Matrix<double, 3, 4> d_grad = vf.distance_grad(pos);
        const Eigen::Matrix<double, 12, 12> d_hess = vf.distance_hess(pos);
        const double f_grad = d_barrier_sq(dist_sq, dhat);
        const double f_hess = d2_barrier_sq(dist_sq, dhat);

        Eigen::MatrixXd vf_hess =
            f_hess * d_grad.reshaped() * d_grad.reshaped().transpose() +
            f_grad * d_hess;
        utils::makeSPD(vf_hess);

        for (int i = 0; i < 4; ++i) {
            int vi;
            if (i == 0) vi = V_F(vf.vertex_idx);
            else vi = F(vf.face_idx, i - 1);
            for (int j = 0; j < 4; ++j) {
                int vj;
                if (j == 0) vj = V_F(vf.vertex_idx);
                else vj = F(vf.face_idx, j - 1);
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        triplets.emplace_back(
                            3 * vi + a, 3 * vj + b,
                            dt_sq * kappa * vf_hess(3 * i + a, 3 * j + b));
                    }
                }
            }
        }
    }

    // floor
    for (const auto& f : floorCollisions) {
        double d = V(V_F[f], 1) - floorHeight;
        const double f_grad = d_barrier_sq(d * d, dhat);
        const double f_hess = d2_barrier_sq(d * d, dhat);
        const double bc_hess = 4 * d * d * f_hess + 2 * f_grad;
        triplets.emplace_back(
            3 * V_F[f] + 1, 3 * V_F[f] + 1,
            dt_sq * kappa * bc_hess);
    }

    // ceil
    for (const auto& f : DBC_Collisions) {
        double d = ceilHeight - V(V_F[f], 1);
        const double f_grad = d_barrier_sq(d * d, dhat);
        const double f_hess = d2_barrier_sq(d * d, dhat);
        const double bc_hess = 4 * d * d * f_hess + 2 * f_grad;
        // vertex
        triplets.emplace_back(
            3 * V_F[f] + 1, 3 * V_F[f] + 1,
            dt_sq * kappa * bc_hess);
        // moving ceil
        const int ceil_dof = 3 * static_cast<int>(V.rows()) + 1;
        triplets.emplace_back(
            ceil_dof, ceil_dof,
            dt_sq * kappa * bc_hess);
        // cross term
        triplets.emplace_back(
            3 * V_F[f] + 1, ceil_dof,
            -dt_sq * kappa * bc_hess);
        triplets.emplace_back(
            ceil_dof, 3 * V_F[f] + 1,
            -dt_sq * kappa * bc_hess);
    }
}

} // namespace fem_ipc
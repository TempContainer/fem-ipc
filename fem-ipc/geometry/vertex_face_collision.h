#pragma once

#include "distance.h"
#include "accd.h"

namespace fem_ipc {

class VertexFaceCollision {
  public:
    int vertex_idx, face_idx;
    VertexFaceCollision(int face_id, int vertex_id)
        : vertex_idx(vertex_id), face_idx(face_id) {}

    double distance(const Eigen::Matrix<double, 3, 4> &position) const {
        return point_triangle_distance(
            position.col(0), position.col(1), position.col(2), position.col(3),
            PointTriangleDistType::AUTO);
    }

    Eigen::Matrix<double, 3, 4> distance_grad(const Eigen::Matrix<double, 3, 4> &position) const {
        return point_triangle_distance_gradient(
            position.col(0), position.col(1), position.col(2), position.col(3),
            PointTriangleDistType::AUTO);
    }

    Eigen::Matrix<double, 12, 12> distance_hess(const Eigen::Matrix<double, 3, 4> &position) const {
        return point_triangle_distance_hessian(
            position.col(0), position.col(1), position.col(2), position.col(3),
            PointTriangleDistType::AUTO);
    }

    bool isBroadIntersect(
        const Eigen::Matrix<double, 3, 4> &pos0, const Eigen::Matrix<double, 3, 4> &pos1,
        const double half_dHat
    ) const {
        Eigen::Vector3d vertex_min = pos0.col(0);
        vertex_min = vertex_min.cwiseMin(pos1.col(0));
        Eigen::Vector3d vertex_max = pos0.col(0);
        vertex_max = vertex_max.cwiseMax(pos1.col(0));

        Eigen::Vector3d face_min = pos0.col(1);
        face_min = face_min.cwiseMin(pos0.col(2));
        face_min = face_min.cwiseMin(pos0.col(3));
        face_min = face_min.cwiseMin(pos1.col(1));
        face_min = face_min.cwiseMin(pos1.col(2));
        face_min = face_min.cwiseMin(pos1.col(3));
        Eigen::Vector3d face_max = pos0.col(1);
        face_max = face_max.cwiseMax(pos0.col(2));
        face_max = face_max.cwiseMax(pos0.col(3));
        face_max = face_max.cwiseMax(pos1.col(1));
        face_max = face_max.cwiseMax(pos1.col(2));
        face_max = face_max.cwiseMax(pos1.col(3));

        vertex_min.array() -= half_dHat;
        vertex_max.array() += half_dHat;
        face_min.array() -= half_dHat;
        face_max.array() += half_dHat;

        return (vertex_min.array() <= face_max.array()).all()
            && (face_min.array() <= vertex_max.array()).all();
    }

    double compute_accd_timestep(
        const Eigen::Matrix<double, 3, 4> &pos0, const Eigen::Matrix<double, 3, 4> &pos1,
        const double t_ccd_fullstep, const double thickness = 0.0, const int max_iteration = 10000
    ) const {
        double t_ccd_addictive = 0.0;
        const Eigen::Vector3d v = pos0.col(0), t0 = pos0.col(1), t1 = pos0.col(2), t2 = pos0.col(3);
        const double init_dist =
            point_triangle_distance(v, t0, t1, t2, PointTriangleDistType::AUTO);
        if ((pos0 - pos1).squaredNorm() == 0.0) {
            if (init_dist > thickness) { return t_ccd_fullstep; }
            printf("initial distance is below dmin, toi = 0!\n");
            return 0.0;
        }

        if (!vertex_face_accd(
                pos0, pos1, thickness, t_ccd_fullstep, t_ccd_addictive, max_iteration, 0.9)) {
            return t_ccd_fullstep;
        }
        return t_ccd_addictive;
    }

};

} // namespace fem_ipc

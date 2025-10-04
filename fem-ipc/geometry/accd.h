#pragma once

#include "distance.h"

namespace fem_ipc {

// N is the number of vertices in the first geometry primitive
template <int N>
bool addictive_ccd(
    Eigen::Matrix<double, 3, 4> pos0, Eigen::Matrix<double, 3, 4> pos_delta, const double thickness,
    const std::function<double(const Eigen::Matrix<double, 3, 4> &)> &distance_function, const double t_ccd_fullstep,
    double &t_ccd_addictive, const int max_iteration,
    const double conservative_rescaling = 0.9
) {
    Eigen::Vector3d pos_delta_average = Eigen::Vector3d::Zero();
    for (int i = 0; i < 4; i++) { pos_delta_average += pos_delta.col(i); }
    pos_delta_average *= 0.25;

    pos_delta.colwise() -= pos_delta_average;

    double first_primative_relative_motion_max = 0.0, second_primative_relative_motion_max = 0.0;
    for (int i = 0; i < N; i++) {
        first_primative_relative_motion_max =
            std::max(first_primative_relative_motion_max, pos_delta.col(i).norm());
    }
    for (int i = N; i < 4; i++) {
        second_primative_relative_motion_max =
            std::max(second_primative_relative_motion_max, pos_delta.col(i).norm());
    }
    double relative_motion_norm =
        first_primative_relative_motion_max + second_primative_relative_motion_max;

    // no relative motion, no collision, full unit step is valid
    if (relative_motion_norm == 0.0) { return false; }

    double dist_squrare = distance_function(pos0);
    // conservative rescaling factor
    const double s = conservative_rescaling;
    double min_seperation =
        s * (dist_squrare - thickness * thickness) / (std::sqrt(dist_squrare) + thickness);

    t_ccd_addictive = 0.0;
    double t_lower_bound = (1 - s) * (dist_squrare - thickness * thickness)
                           / (relative_motion_norm * (std::sqrt(dist_squrare) + thickness));

    Eigen::Matrix<double, 3, 4> pos = pos0;
    int iter = 0;
    while (true) {
        iter += 1;
        for (int i = 0; i < 4; i++) {
            pos.col(i) = pos0.col(i) + t_lower_bound * pos_delta.col(i);
        }
        dist_squrare = distance_function(pos);
        if (t_ccd_addictive > 0.0
            && (dist_squrare - thickness * thickness) / (std::sqrt(dist_squrare) + thickness)
                   < min_seperation) {
            break;
        }

        t_ccd_addictive = t_ccd_addictive + t_lower_bound;
        if (t_ccd_addictive > t_ccd_fullstep) { return false; }

        t_lower_bound = 0.9 * (dist_squrare - thickness * thickness)
                        / (relative_motion_norm * (std::sqrt(dist_squrare) + thickness));
        if (iter > max_iteration) { return false; }
    }
    return true;
}

static bool edge_edge_accd(
    const Eigen::Matrix<double, 3, 4> &pos0, const Eigen::Matrix<double, 3, 4> &pos1, const double thickness,
    const double t_ccd_fullstep, double &t_ccd_addictive, const int max_iteration,
    const double conservative_rescaling = 0.9
) {
    const Eigen::Matrix<double, 3, 4> pos_delta = pos1 - pos0;
    auto dist_function = [](const Eigen::Matrix<double, 3, 4> &position) {
        return edge_edge_distance(
            position.col(0), position.col(1), position.col(2), position.col(3),
            EdgeEdgeDistType::AUTO);
    };
    return addictive_ccd<2>(
        pos0, pos_delta, thickness, dist_function, t_ccd_fullstep, t_ccd_addictive, max_iteration,
        conservative_rescaling);
}

static bool vertex_face_accd(
    const Eigen::Matrix<double, 3, 4> &pos0, const Eigen::Matrix<double, 3, 4> &pos1, const double thickness,
    const double t_ccd_fullstep, double &t_ccd_addictive, const int max_iteration,
    const double conservative_rescaling = 0.9
) {
    const Eigen::Matrix<double, 3, 4> pos_delta = pos1 - pos0;
    auto dist_function = [](const Eigen::Matrix<double, 3, 4> &position) {
        return point_triangle_distance(
            position.col(0), position.col(1), position.col(2), position.col(3),
            PointTriangleDistType::AUTO);
    };
    return addictive_ccd<1>(
        pos0, pos_delta, thickness, dist_function, t_ccd_fullstep, t_ccd_addictive, max_iteration,
        conservative_rescaling);
}

} // namespace fem_ipc

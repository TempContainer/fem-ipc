#pragma once

#include <Eigen/Core>
#include <string>

namespace fem_ipc {

// Read a Gmsh v2 ASCII mesh (.msh)
// Inputs:
//  - path: path to .msh file; if relative, the loader will try:
//      1) as-is relative to CWD
//      2) models/<filename> under CWD
//      3) PROJECT_ROOT/<path> if PROJECT_ROOT is defined
//      4) PROJECT_ROOT/models/<filename>
// Outputs:
//  - V: (#V x 3) double vertex positions
//  - T: (#T x 4) int tetrahedra (1-based in file -> 0-based indices here)
//  - F: (#F x 3) int surface triangles (optional; empty if none in file)
void readMESH(const std::string& path,
              Eigen::MatrixX3d& V,
              Eigen::MatrixX4i& T,
              Eigen::MatrixX3i& F);

} // namespace fem_ipc
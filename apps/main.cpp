#include <polyscope/polyscope.h>
#include <polyscope/volume_mesh.h>
#include <polyscope/surface_mesh.h>
#include <igl/boundary_facets.h>
#include <igl/writeOBJ.h>
#include <fem-ipc/fem-ipc.h>
#include <spdlog/spdlog.h>

using namespace fem_ipc;

int main()
{
    polyscope::init();

    TetMesh mesh;
    readMESH(std::string(PROJECT_ROOT) + std::string("/models/bunny.msh"), mesh.V, mesh.T, mesh.F);
    spdlog::info("Loaded mesh with {} vertices, {} tets, and {} faces", mesh.V.rows(), mesh.T.rows(), mesh.F.rows());
    igl::boundary_facets(mesh.T, mesh.F);
    mesh.computeSurfaceVertexAndEdgeIndices();
    spdlog::info("Computed boundary facets: {} faces", mesh.F.rows());
    spdlog::info("Surface has {} vertices and {} edges", mesh.V_F.size(), mesh.E_F.rows());

    Simulator simulator(mesh);
    simulator.initialize();
    for (int i = 0; i < 200; ++i) {
        simulator.step();
        spdlog::info("Step {}: ceil at {}, energy = {}", i, simulator.ceilPos(1), simulator.totalEnergyValue(simulator.mesh.V));
        std::string filename = "bunny_" + std::to_string(i) + ".obj";
        igl::writeOBJ(filename, simulator.mesh.V, simulator.mesh.F);
    }

    polyscope::registerSurfaceMesh("bunny", mesh.V, mesh.F);
    polyscope::show();

    return 0;
}
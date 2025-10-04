#include "tet_mesh.h"

#include <fem-ipc/energy/fem_utils.h>
#include <unordered_set>

namespace fem_ipc {

void TetMesh::computeSurfaceVertexAndEdgeIndices()
{
    const int num_faces = static_cast<int>(F.rows());
    V_F.resize(0);
    E_F.resize(0, 2);
    if (num_faces == 0) return;

    std::unordered_set<int> vset;
    vset.reserve(num_faces * 3);
    std::vector<int> vlist;
    vlist.reserve(num_faces * 3);

    auto add_vertex = [&](int v) {
        if (vset.insert(v).second) vlist.push_back(v);
    };

    auto edge_key = [](int a, int b) -> std::uint64_t {
        if (a > b) std::swap(a, b);
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32)
             |  static_cast<std::uint32_t>(b);
    };

    std::unordered_set<std::uint64_t> eset;
    eset.reserve(num_faces * 3);
    std::vector<Eigen::Vector2i> elist;
    elist.reserve(num_faces * 3);

    for (int f = 0; f < num_faces; ++f) {
        const int a = F(f, 0), b = F(f, 1), c = F(f, 2);

        add_vertex(a); add_vertex(b); add_vertex(c);

        const std::pair<int,int> edges[3] = { {a,b}, {b,c}, {c,a} };
        for (int i = 0; i < 3; ++i) {
            int u = edges[i].first, v = edges[i].second;
            const std::uint64_t k = edge_key(u, v);
            if (eset.insert(k).second) {
                if (u > v) std::swap(u, v);
                elist.emplace_back(u, v);
            }
        }
    }

    V_F.resize(static_cast<int>(vlist.size()));
    for (int i = 0; i < static_cast<int>(vlist.size()); ++i) V_F[i] = vlist[i];

    E_F.resize(static_cast<int>(elist.size()), 2);
    E_F_restLength.resize(static_cast<int>(elist.size()));
    for (int i = 0; i < static_cast<int>(elist.size()); ++i) {
        E_F.row(i) = elist[i].transpose();
        E_F_restLength[i] = (V.row(E_F(i,0)) - V.row(E_F(i,1))).squaredNorm();
    }
}

void TetMesh::computeMassAndRestVolume()
{
    const int num_tets = static_cast<int>(T.rows());
    volumes.resize(num_tets);
    inv_Dm.resize(num_tets);
    masses.resize(V.rows());
    masses.setZero();

    for (int e = 0; e < num_tets; ++e) {
        Eigen::Matrix<double, 3, 4> X;
        for (int local = 0; local < 4; ++local) {
            const int vertex = T(e, local);
            X.col(local) = V.row(vertex).transpose();
        }

        Eigen::Matrix3d Dm = utils::Ds(X);

        volumes[e] = std::abs(Dm.determinant()) / 6.0;
        inv_Dm[e] = Dm.inverse();
        const double mass = rho * volumes[e] / 4.0;
        for (int local = 0; local < 4; ++local) {
            const int vertex = T(e, local);
            masses[vertex] += mass;
        }
    }
}

} // namespace fem_ipc
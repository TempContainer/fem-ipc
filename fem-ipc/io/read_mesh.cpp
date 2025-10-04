#include "read_mesh.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>

namespace fem_ipc {

namespace fs = std::filesystem;

void readMESH(const std::string& path, Eigen::MatrixX3d& V, Eigen::MatrixX4i& T, Eigen::MatrixX3i& F) {
    V.resize(0, 3);
    T.resize(0, 4);
    F.resize(0, 3);

    fs::path file(path);
    if (file.empty()) {
        throw std::runtime_error("readMESH: cannot find file: " + path);
    }

    std::ifstream fin(file);
    if (!fin) {
        throw std::runtime_error("readMESH: failed to open: " + file.string());
    }

    std::string line;

    // Expect $MeshFormat ... $EndMeshFormat
    if (!std::getline(fin, line)) throw std::runtime_error("readMESH: unexpected EOF");
    if (line.rfind("$MeshFormat", 0) != 0) {
        throw std::runtime_error("readMESH: only supports Gmsh v2 ASCII ($MeshFormat section)");
    }
    if (!std::getline(fin, line)) throw std::runtime_error("readMESH: bad MeshFormat");
    // Format line like: "2.2 0 8" (version, file-type(0 ascii), data-size)
    // We ignore details but could validate ascii
    if (!std::getline(fin, line)) throw std::runtime_error("readMESH: missing EndMeshFormat");
    if (line.rfind("$EndMeshFormat", 0) != 0) throw std::runtime_error("readMESH: malformed MeshFormat section");

    // Find $Nodes
    while (std::getline(fin, line)) {
        if (line.rfind("$Nodes", 0) == 0) break;
    }
    if (!fin) throw std::runtime_error("readMESH: missing $Nodes section");

    // number of nodes
    size_t num_nodes = 0;
    {
        if (!std::getline(fin, line)) throw std::runtime_error("readMESH: missing node count");
        std::istringstream iss(line);
        iss >> num_nodes;
        if (!iss) throw std::runtime_error("readMESH: invalid node count");
    }

    std::vector<Eigen::Vector3d> verts;
    verts.reserve(num_nodes);

    // nodes lines: id x y z (id is 1-based)
    for (size_t i = 0; i < num_nodes; ++i) {
        if (!std::getline(fin, line)) throw std::runtime_error("readMESH: unexpected EOF in nodes");
        std::istringstream iss(line);
        int id; double x, y, z;
        iss >> id >> x >> y >> z;
        if (!iss) throw std::runtime_error("readMESH: invalid node line");
        // push in order – gmsh node ids are not guaranteed contiguous, so we map after
        verts.emplace_back(x, y, z);
    }
    // End nodes
    if (!std::getline(fin, line) || line.rfind("$EndNodes", 0) != 0) {
        // Some files might continue directly; try to find $EndNodes
        if (!fin) throw std::runtime_error("readMESH: missing $EndNodes");
        // Skip until $EndNodes is found
        while (line.rfind("$EndNodes", 0) != 0 && std::getline(fin, line)) {}
        if (line.rfind("$EndNodes", 0) != 0) throw std::runtime_error("readMESH: missing $EndNodes");
    }

    // $Elements
    while (std::getline(fin, line)) {
        if (line.rfind("$Elements", 0) == 0) break;
    }
    if (!fin) throw std::runtime_error("readMESH: missing $Elements section");

    size_t num_elements = 0;
    {
        if (!std::getline(fin, line)) throw std::runtime_error("readMESH: missing element count");
        std::istringstream iss(line);
        iss >> num_elements;
        if (!iss) throw std::runtime_error("readMESH: invalid element count");
    }

    // We'll collect tets (type 4) and triangles (type 2)
    std::vector<Eigen::Vector4i> tets;
    std::vector<Eigen::Vector3i> tris;
    tets.reserve(num_elements / 2);
    tris.reserve(num_elements / 2);

    // To handle non-contiguous node ids, map encountered ids to 0..(n-1).
    // But many meshes list nodes in increasing order, so we can assume ids are 1..N
    // For safety, we store last id seen and adjust indices by -1; if out of range, we'll check.
    const int nV = static_cast<int>(num_nodes);

    for (size_t i = 0; i < num_elements; ++i) {
        if (!std::getline(fin, line)) throw std::runtime_error("readMESH: unexpected EOF in elements");
        if (line.empty()) { --i; continue; }
        std::istringstream iss(line);
        int eid, etype, ntags;
        iss >> eid >> etype >> ntags;
        if (!iss) throw std::runtime_error("readMESH: invalid element header");
        // skip tags
        for (int t = 0; t < ntags; ++t) { int tmp; iss >> tmp; }
        if (etype == 2) { // triangle, 3 nodes
            int a,b,c; iss >> a >> b >> c;
            if (!iss) throw std::runtime_error("readMESH: bad triangle record");
            // convert to 0-based
            a--; b--; c--;
            if (a<0||b<0||c<0||a>=nV||b>=nV||c>=nV) throw std::runtime_error("readMESH: triangle index out of range");
            tris.emplace_back(a,b,c);
        } else if (etype == 4) { // tetra, 4 nodes
            int a,b,c,d; iss >> a >> b >> c >> d;
            if (!iss) throw std::runtime_error("readMESH: bad tetra record");
            a--; b--; c--; d--;
            if (a<0||b<0||c<0||d<0||a>=nV||b>=nV||c>=nV||d>=nV) throw std::runtime_error("readMESH: tetra index out of range");
            tets.emplace_back(a,b,c,d);
        } else {
            // ignore other element types
        }
    }

    // optional $EndElements
    if (std::getline(fin, line)) {
        if (line.rfind("$EndElements", 0) != 0) {
            // try to find and ignore the rest
        }
    }

    // Fill V
    V.resize(static_cast<int>(verts.size()), 3);
    for (int i = 0; i < (int)verts.size(); ++i) {
        V.row(i) = verts[i];
    }
    // Fill T
    T.resize(static_cast<int>(tets.size()), 4);
    for (int i = 0; i < (int)tets.size(); ++i) {
        T.row(i) = tets[i];
    }
    // Fill F
    F.resize(static_cast<int>(tris.size()), 3);
    for (int i = 0; i < (int)tris.size(); ++i) {
        F.row(i) = tris[i];
    }
}

} // namespace fem_ipc
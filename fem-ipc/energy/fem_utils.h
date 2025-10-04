#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace fem_ipc {

namespace utils {

using Matrix9x12 = Eigen::Matrix<double, 9, 12>;

inline Eigen::Vector<double, 9> flatten(const Eigen::Matrix3d& A) noexcept
{
    Eigen::Vector<double, 9> column;

    Eigen::Index index = 0;
    for(Eigen::Index j = 0; j < A.cols(); j++)
        for(Eigen::Index i = 0; i < A.rows(); i++, index++)
            column[index] = A(i, j);

    return column;
}

inline void makeSPD(Eigen::MatrixXd& mat)
{
    Eigen::EigenSolver<Eigen::MatrixXd> es(mat);

    Eigen::VectorXd eigenvalues = es.eigenvalues().real();

    for (Eigen::Index i = 0; i < eigenvalues.size(); ++i)
        if (eigenvalues[i] < 0)
            eigenvalues[i] = 0;

    Eigen::MatrixXd V = es.eigenvectors().real();
    Eigen::MatrixXd Vt = V.transpose();
    Eigen::MatrixXd lambda = eigenvalues.asDiagonal();

    mat = V * lambda * Vt;
}

inline Eigen::Matrix3d Ds(const Eigen::Matrix<double, 3, 4>& x)
{
    Eigen::Matrix3d Ds;
    Ds.col(0) = x.col(1) - x.col(0);
    Ds.col(1) = x.col(2) - x.col(0);
    Ds.col(2) = x.col(3) - x.col(0);
    return Ds;
}

inline Matrix9x12 dFdx(const Eigen::Matrix3d& DmInv)
{
    const double m = DmInv(0, 0);
    const double n = DmInv(0, 1);
    const double o = DmInv(0, 2);
    const double p = DmInv(1, 0);
    const double q = DmInv(1, 1);
    const double r = DmInv(1, 2);
    const double s = DmInv(2, 0);
    const double t = DmInv(2, 1);
    const double u = DmInv(2, 2);

    const double t1 = -m - p - s;
    const double t2 = -n - q - t;
    const double t3 = -o - r - u;

    Matrix9x12 PFPu = Matrix9x12::Zero();
    PFPu(0, 0)      = t1;
    PFPu(0, 3)      = m;
    PFPu(0, 6)      = p;
    PFPu(0, 9)      = s;
    PFPu(1, 1)      = t1;
    PFPu(1, 4)      = m;
    PFPu(1, 7)      = p;
    PFPu(1, 10)     = s;
    PFPu(2, 2)      = t1;
    PFPu(2, 5)      = m;
    PFPu(2, 8)      = p;
    PFPu(2, 11)     = s;
    PFPu(3, 0)      = t2;
    PFPu(3, 3)      = n;
    PFPu(3, 6)      = q;
    PFPu(3, 9)      = t;
    PFPu(4, 1)      = t2;
    PFPu(4, 4)      = n;
    PFPu(4, 7)      = q;
    PFPu(4, 10)     = t;
    PFPu(5, 2)      = t2;
    PFPu(5, 5)      = n;
    PFPu(5, 8)      = q;
    PFPu(5, 11)     = t;
    PFPu(6, 0)      = t3;
    PFPu(6, 3)      = o;
    PFPu(6, 6)      = r;
    PFPu(6, 9)      = u;
    PFPu(7, 1)      = t3;
    PFPu(7, 4)      = o;
    PFPu(7, 7)      = r;
    PFPu(7, 10)     = u;
    PFPu(8, 2)      = t3;
    PFPu(8, 5)      = o;
    PFPu(8, 8)      = r;
    PFPu(8, 11)     = u;

    return PFPu;
}

inline Eigen::Matrix3d F(const Eigen::Matrix<double, 3, 4>& x, const Eigen::Matrix3d& DmInv)
{
    return Ds(x) * DmInv;
}

} // namespace utils

} // namespace fem_ipc
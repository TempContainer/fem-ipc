#include "stable_neo_hookean_energy.h"
#include "fem_utils.h"

#include <Eigen/LU>
#include <Eigen/Geometry>
#include <cmath>

namespace fem_ipc {

double StableNeoHookeanEnergy::E(const Eigen::Matrix3d& F)
{
	const double J = F.determinant();
    const double Ic = F.squaredNorm();
    const double alpha = 1 + 0.75 * mu / lambda;
    return 0.5 * lambda * (J - alpha) * (J - alpha) + 0.5 * mu * (Ic - 3) - 0.5 * mu * log(Ic + 1);
}

Eigen::Matrix3d StableNeoHookeanEnergy::dEdVecF(const Eigen::Matrix3d& F)
{
	const double J  = F.determinant();
    const double Ic = F.squaredNorm();
    Eigen::Matrix3d pJpF;

    pJpF(0, 0) = F(1, 1) * F(2, 2) - F(1, 2) * F(2, 1);
    pJpF(0, 1) = F(1, 2) * F(2, 0) - F(1, 0) * F(2, 2);
    pJpF(0, 2) = F(1, 0) * F(2, 1) - F(1, 1) * F(2, 0);

    pJpF(1, 0) = F(2, 1) * F(0, 2) - F(2, 2) * F(0, 1);
    pJpF(1, 1) = F(2, 2) * F(0, 0) - F(2, 0) * F(0, 2);
    pJpF(1, 2) = F(2, 0) * F(0, 1) - F(2, 1) * F(0, 0);

    pJpF(2, 0) = F(0, 1) * F(1, 2) - F(1, 1) * F(0, 2);
    pJpF(2, 1) = F(0, 2) * F(1, 0) - F(0, 0) * F(1, 2);
    pJpF(2, 2) = F(0, 0) * F(1, 1) - F(0, 1) * F(1, 0);

    return mu * (1 - 1 / (Ic + 1)) * F + (lambda * (J - 1 - 0.75 * mu / lambda)) * pJpF;
}

Eigen::Matrix<double, 9, 9> StableNeoHookeanEnergy::ddEddVecF(const Eigen::Matrix3d& F)
{
    const double J  = F.determinant();
    const double Ic = F.squaredNorm();
    Eigen::Matrix<double, 9, 9> H1 = 2 * Eigen::Matrix<double, 9, 9>::Identity();
    Eigen::Matrix<double, 9, 1> g1;
    g1.block<3, 1>(0, 0) = 2 * F.col(0);
    g1.block<3, 1>(3, 0) = 2 * F.col(1);
    g1.block<3, 1>(6, 0) = 2 * F.col(2);
    Eigen::Matrix<double, 9, 1> gJ;
    gJ.block<3, 1>(0, 0) = F.col(1).cross(F.col(2));
    gJ.block<3, 1>(3, 0) = F.col(2).cross(F.col(0));
    gJ.block<3, 1>(6, 0) = F.col(0).cross(F.col(1));
    Eigen::Matrix<double, 3, 3> f0hat;
    f0hat << 
		0, -F(2, 0), F(1, 0),
		F(2, 0), 0, -F(0, 0),
		-F(1, 0), F(0, 0), 0;
    Eigen::Matrix<double, 3, 3> f1hat;
    f1hat << 
		0, -F(2, 1), F(1, 1),
		F(2, 1), 0, -F(0, 1),
		-F(1, 1), F(0, 1), 0;
    Eigen::Matrix<double, 3, 3> f2hat;
    f2hat << 
		0, -F(2, 2), F(1, 2),
		F(2, 2), 0, -F(0, 2),
		-F(1, 2), F(0, 2), 0;
    Eigen::Matrix<double, 9, 9> HJ;
    HJ.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Zero();
    HJ.block<3, 3>(0, 3) = -f2hat;
    HJ.block<3, 3>(0, 6) = f1hat;
    HJ.block<3, 3>(3, 0) = f2hat;
    HJ.block<3, 3>(3, 3) = Eigen::Matrix<double, 3, 3>::Zero();
    HJ.block<3, 3>(3, 6) = -f0hat;
    HJ.block<3, 3>(6, 0) = -f1hat;
    HJ.block<3, 3>(6, 3) = f0hat;
    HJ.block<3, 3>(6, 6) = Eigen::Matrix<double, 3, 3>::Zero();
    return (Ic * mu) / (2 * (Ic + 1)) * H1 + 
		lambda * (J - 1 - (3 * mu) / (4.0 * lambda)) * HJ + 
		(mu / (2 * (Ic + 1) * (Ic + 1))) * g1 * g1.transpose() + 
		lambda * gJ * gJ.transpose();
}

double StableNeoHookeanEnergy::valuePerTet(
	const Eigen::Matrix<double, 3, 4>& x,
	const Eigen::Matrix3d& Dm_inv,
	double volume
) {
	const Eigen::Matrix3d F = utils::F(x, Dm_inv);
	return volume * E(F);
}

Eigen::Vector<double, 12> StableNeoHookeanEnergy::gradientPerTet(
	const Eigen::Matrix<double, 3, 4>& x,
	const Eigen::Matrix3d& Dm_inv,
	double volume
) {
	const Eigen::Matrix3d F = utils::F(x, Dm_inv);
	const Eigen::Matrix<double, 9, 12> dFdx = utils::dFdx(Dm_inv);
	const Eigen::Matrix3d P = dEdVecF(F);
	Eigen::Vector<double, 9> vecP = utils::flatten(P);
	vecP *= volume;
	return dFdx.transpose() * vecP;
}

Eigen::Matrix<double, 12, 12> StableNeoHookeanEnergy::hessianPerTet(
	const Eigen::Matrix<double, 3, 4>& x,
	const Eigen::Matrix3d& Dm_inv,
	double volume 
) {
	const Eigen::Matrix3d F = utils::F(x, Dm_inv);
	const Eigen::Matrix<double, 9, 12> dFdx = utils::dFdx(Dm_inv);
	Eigen::MatrixXd d2EdF2 = ddEddVecF(F);
	d2EdF2 *= volume;
	utils::makeSPD(d2EdF2);
	return dFdx.transpose() * d2EdF2 * dFdx;
}

} // namespace fem_ipc
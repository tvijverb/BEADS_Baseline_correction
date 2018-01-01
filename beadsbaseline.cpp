#include "beadsbaseline.h"

beadsBaseline::beadsBaseline()
{

}

VectorXd beadsBaseline::H(std::vector<Eigen::SparseMatrix<double>> AB, Eigen::VectorXd x)
{
    // anonimous functions of beads
//    phi = @(x) abs(x) - EPS1 * log(abs(x) + EPS1);
	VectorXd xz(AB.at(0).cols());
	SparseQR<SparseMatrix<double>, COLAMDOrdering<double> > qr(AB.at(0));

	return AB.at(1)*(qr.solve(x));

}

Eigen::VectorXd beadsBaseline::wfun(Eigen::VectorXd x, double EPS1)
{
//    wfun = @(x) 1./( abs(x) + EPS1);
	Eigen::VectorXd x2 = Eigen::VectorXd::Constant(x.rows(), x.cols(), 0.00001) + x.cwiseAbs();

	
	VectorXd x3 = Eigen::VectorXd::Ones(x2.rows(), x2.cols()).array() / x2.array();
    return x3;
}

Eigen::SparseMatrix<double> spdiags(const Eigen::MatrixXd& A, std::vector<int> diagPos)
{
	// Input matrixXd with size (x,N) where:
	// x is the number of bands required
	// N is the data length
	// Input diagPos where:
	// vector of diagonal placement such as for instance {-1,0,1} for symmetric off-diagonal placement

	int nCol = A.cols();
	int nRow = A.rows();
	int nBands = diagPos.size();

	//Eigen::MatrixXd B(nCol, nCol);
	SparseMatrix<double> mat(nCol, nCol);
	//B.setZero();

	for (int i = 0; i < nBands; i++)
	{
		for (int j = 0; j < nCol; j++)
		{
			if ((j + diagPos.at(i)) >= 0 && j + diagPos.at(i) < mat.cols())
			{
				//B(j, j + diagPos.at(i)) = A(i, j);
				mat.insert(j, j + diagPos.at(i)) = A(i, j);
			}
		}
	}
	mat.pruned();
	mat.makeCompressed();
	return mat;
}

void removeRow(Eigen::SparseMatrix<double>& matrix, unsigned int rowToRemove)
{
	unsigned int numRows = matrix.rows() - 1;
	unsigned int numCols = matrix.cols();

	if (rowToRemove < numRows)
		matrix.block(rowToRemove, 0, numRows - rowToRemove, numCols) = matrix.block(rowToRemove + 1, 0, numRows - rowToRemove, numCols);

	matrix.conservativeResize(numRows, numCols);
}

void removeColumn(Eigen::SparseMatrix<double>& matrix, unsigned int colToRemove)
{
	unsigned int numRows = matrix.rows();
	unsigned int numCols = matrix.cols() - 1;

	if (colToRemove < numCols)
		matrix.block(0, colToRemove, numRows, numCols - colToRemove) = matrix.block(0, colToRemove + 1, numRows, numCols - colToRemove);

	matrix.conservativeResize(numRows, numCols);
}

std::vector<std::vector<double>> beadsBaseline::beads(std::vector<double> y, int d, double fc, double positivityBias, double lam0, double lam1, double lam2)
{
	std::vector<double> a(y.begin(), y.end());
	Eigen::VectorXd y1 = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(a.data(), a.size());
	Eigen::VectorXd x = y1;

	int N = y.size();
	int Iter = 29;

	Eigen::VectorXd cost = Eigen::VectorXd::Zero(N, 1);

	std::vector<Eigen::SparseMatrix<double>> AB = BAfilt(d, fc, N);

	MatrixXd myOne = Eigen::MatrixXd::Ones(1, N);

	Eigen::MatrixXd AX(2, N);
	AX.row(0) = Eigen::MatrixXd::Constant(1, N, -1);
	AX.row(1) = myOne;

	std::vector<int> diagPos{ 0,1 };

	Eigen::SparseMatrix<double> D1 = spdiags(AX, diagPos);
	removeRow(D1, N);

	Eigen::MatrixXd AX2(3, N);
	AX2.row(1) = Eigen::MatrixXd::Constant(1, N, -2);
	AX2.row(0) = myOne;
	AX2.row(2) = myOne;

	diagPos.push_back(2);

	Eigen::SparseMatrix<double> D2 = spdiags(AX2, diagPos);

	removeRow(D2, N);
	removeRow(D2, N - 1);

	Eigen::SparseMatrix<double> D((D1.rows() + D2.rows()), D1.cols());
	D.setZero();

	// Fill D with triples from the other matrices
	std::vector<Triplet<double> > tripletList;
	for (int k = 0; k < D1.outerSize(); ++k)
	{
		for (SparseMatrix<double>::InnerIterator it(D1, k); it; ++it)
		{
			tripletList.push_back(Triplet<double>(it.row(), it.col(), it.value()));
		}
	}
	for (int k = 0; k < D2.outerSize(); ++k)
	{
		for (SparseMatrix<double>::InnerIterator it(D2, k); it; ++it)
		{
			tripletList.push_back(Triplet<double>(it.row()+D1.rows(), it.col(), it.value()));
		}
	}
	D.setFromTriplets(tripletList.begin(), tripletList.end());
	D.pruned();
	D.makeCompressed();

	SparseMatrix<double> BTB = AB.at(1).transpose() * AB.at(1);
	BTB.pruned();
	BTB.makeCompressed();

	Eigen::MatrixXd w1 = Eigen::MatrixXd::Constant(N - 1, 1, lam1);
	Eigen::MatrixXd w2 = Eigen::MatrixXd::Constant(N - 2, 1, lam2);
	Eigen::MatrixXd w(w1.rows() + w2.rows(), w1.cols());
	w << w1, w2;

	Eigen::MatrixXd b = Eigen::MatrixXd::Constant(N, 1, (1 - positivityBias) / 2);
	
	SparseQR<SparseMatrix<double>, COLAMDOrdering<SparseMatrix<double>::Index>> QR(AB.at(0));

	Eigen::VectorXd d1 = BTB * QR.solve(y1) - lam0 * AB.at(0).transpose() * b;

	Eigen::VectorXd gamma = Eigen::VectorXd::Ones(N, 1);

	std::vector<int> placeDiag{ 0 };

	for (int i = 0; i < Iter; i++)
	{
		//Eigen::MatrixXd prod = D*x;
		Eigen::SparseMatrix<double> lambda = spdiags((w.cwiseProduct(wfun(D*x, 0.000001))).transpose(), placeDiag);

		VectorXd k = (x.array().abs() > Eigen::VectorXd::Constant(x.rows(), x.cols(), 0.00001).array()).cast<double>();
		VectorXd x2 = x.cwiseAbs();
		for (int i = 0; i < k.size();  i++)
		{
			if (k(i, 0) == 1)
			{
				gamma(i, 0) = ((1 + positivityBias) / 4) / x2(i);
			}
			if (k(i, 0) == 0)
			{
				gamma(i, 0) = ((1 + positivityBias) / 4) / 0.000001;
			}
		}
		Eigen::SparseMatrix<double> Gamma = spdiags(gamma.transpose(), placeDiag);

		Gamma.pruned();
		Gamma.makeCompressed();

		MatrixXd lam0c = MatrixXd::Constant(Gamma.rows(), Gamma.cols(),2*lam0);

		SparseMatrix<double> M = lam0c.cwiseProduct(Gamma) + D.transpose() * lambda * D;
		M.pruned();
		M.makeCompressed();

		//VectorXd xz2(AB.at(0).cols());

		SparseQR<SparseMatrix<double>, COLAMDOrdering<SparseMatrix<double>::Index>> QR2(BTB + (AB.at(0).transpose() * M * AB.at(0)) );
		//xz2 = QR2.solve(d1);
		
		x = AB.at(0) * QR2.solve(d1);
	}
	VectorXd f = y1 - x - H(AB, y1 - x);

	std::vector<double> vec1(f.data(), f.data() + f.size());
	std::vector<double> vec2(x.data(), x.data() + x.size());


	std::vector<std::vector<double>> pb{ vec2,vec1 };
    return pb;
}

template<typename T>
std::vector<T>
conv_valid(std::vector<T> const &f, std::vector<T> const &g) {
  int const nf = f.size();
  int const ng = g.size();

  int const n = nf + ng - 1;
  std::vector<T> out;
  for(int i = 0; i < n; i++) {
	  out.push_back(0);
	  for (int j = 0; j <= i; j++)
	  {
		  if ((i - j) < ng && j < nf)
			  out[i] += f[j] * g[(i - j)];
	  }
		
  }
  return out;
}

std::vector<Eigen::SparseMatrix<double>> beadsBaseline::BAfilt(int d,double fc,int N)
{
    std::vector<double> b1 {1,-1};
	std::vector<double> b{ -1,1 };
    std::vector<double> v1 {-1,2,-1};
	std::vector<double> v2{ 1,2,1 };

    double omc,t;
    std::vector<double> a {1};

    for(int i = 0; i < d-1; i++)
    {
        b1 = conv_valid(b1, v1);
    }
    b = conv_valid(b1, b);

    omc = 2*M_PI*fc;
    t = std::pow((1-cos(omc))/(1+cos(omc)),d);

    for(int i = 0; i < d; i++)
    {
        a = conv_valid(a, v2);
    }

    for(int i = 0; i < a.size(); i++)
    {
        a[i] = a[i] * t + b[i];
    }

    Eigen::VectorXd v3 = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(a.data(), a.size());
    Eigen::VectorXd v4 = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(b.data(), b.size());

    Eigen::VectorXd a2 = Eigen::VectorXd::Ones(N,1);

    Eigen::MatrixXd AX = v3*a2.transpose();
    Eigen::MatrixXd BX = v4*a2.transpose();

	std::vector<int> diagPos{ -1,0,1 };

	Eigen::SparseMatrix<double> A2 = spdiags(AX,diagPos);
	Eigen::SparseMatrix<double> B2 = spdiags(BX,diagPos);

    std::vector<Eigen::SparseMatrix<double>> S;
    S.push_back(A2);
    S.push_back(B2);
    return S;
}

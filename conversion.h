#include <armadillo>
#include <boost/foreach.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

//void arma2vector() {
//
//	int nRow = 4;
//	int nCol = 5;
//	arma::mat A = arma::randu<arma::mat>(nRow, nCol);
//	std::cout << " arma::mat A: " << A << std::endl;
//	arma::mat tmpA = A.t();
//
//	std::vector<std::vector<double> > Vec;
//	Vec.resize(nRow);
//
//	for (int i = 0; i < nRow; i++) {
//		// initialize tmpVec by input A data pointer
//		std::vector<double> tmpVec(tmpA.colptr(i), tmpA.colptr(i) + nCol);
//		Vec[i] = tmpVec;
//	}
//
//	std::cout << " A to tmpA conversion: " << std::endl;
//	BOOST_FOREACH(std::vector<double> tmpVec, Vec) {
//		BOOST_FOREACH(double d, tmpVec) {
//			std::cout << d << " ";
//		}
//		std::cout << std::endl;
//	}
//}
//
//void mat2arma() {
//	cv::Mat img(2, 2, CV_32FC1);
//	cv::randu(img, cv::Scalar(0), cv::Scalar(255));
//
//	std::cout << "img Test" << img << std::endl;
//	cv::Mat imgt(img.t());
//
//	// mat to armadillo
//	arma::fmat armaConv(imgt.ptr<float>(), 2, 2);
//	std::cout << "armaConv" << std::endl << armaConv << std::endl;
//}
//
//void arma2mat() {
//	arma::fmat arma2matData = arma::randu<arma::fmat>(5, 6);
//	std::cout << "arma2matData: " << std::endl << arma2matData << std::endl;
//
//	cv::Mat cvMatConvTmp(6, 5, CV_32FC1, arma2matData.memptr());
//	cv::Mat cvMatConv(cvMatConvTmp.t());
//	std::cout << "cvMatConv: " << std::endl << cvMatConv << std::endl;
//}
//
//template<typename T>
//cv::Mat_<T> vec2mat(std::vector<std::vector<T> > &inVec) {
//	int rows = static_cast<int>(inVec.size());
//	int cols = static_cast<int>(inVec[0].size());
//
//	cv::Mat_<T> resmat(rows, cols);
//	for (int i = 0; i < rows; i++) {
//		resmat.row(i) = cv::Mat(inVec[i]).t();
//	}
//	return resmat;
//}

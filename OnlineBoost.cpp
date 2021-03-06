// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License 
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#include "OnlineBoost.h"
#include "Sample.h"

using namespace arma;

/*
 * Initialize weak classifier
 */
void ClfWeak::init(int id, float lRate, Ftr *ftr) {
	_lRate = lRate;
	_ind = id;
	_ftr = ftr;
	_mu0 = 0;
	_mu1 = 0;
	_sig0 = 1;
	_sig1 = 1;
	_lRate = 0.85f;
	_trained = false;
}

/*
 * Initialize strong classifier
 */
void ClfStrong::init(ClfParams *params) {
	_params = params;
	_numsamples = 0;

	// generate _numFeat random features
	_ftrs = Ftr::generateAll(_params->_ftrParams, _params->_numFeat);
	_weakclf.resize(_params->_numFeat);

	// initialize _numFeat weak classifiers using these random filters
	for (int k = 0; k < _params->_numFeat; k++) {
		_weakclf[k] = new ClfWeak();
		_weakclf[k]->init(k, _params->_lRate, _ftrs[k]);
	}
}

/*
 * Given positive and negative SampleSet as inputs,
 * update _selectors and _weakclf of strong classifier
 */
void ClfStrong::update(SampleSet &posx, SampleSet &negx) {
	int numneg = negx.size();
	int numpos = posx.size();

	fprintf(stderr,
			"update strong classifier using %d positive, %d negative samples\n",
			numpos, numneg);

	if (!posx.ftrsComputed())
		Ftr::computeAll(posx, _ftrs);
	if (!negx.ftrsComputed())
		Ftr::computeAll(negx, _ftrs);

	_selectors.clear();

	fvec Hpos(numpos);
	fvec Hneg(numneg);
	fvec L(_params->_numFeat);
	fmat hpos(numpos, _params->_numFeat);
	fmat hneg(numneg, _params->_numFeat);
	float Lpos, Lneg;

	fprintf(stderr, "update weak classifiers \n");
	for (int j = 0; j < _params->_numFeat; j++) {
		_weakclf[j]->update(posx, negx);
	}

	// pick the best features
	for (int s = 0; s < _params->_numSel; s++) {
		for (int j = 0; j < (int) _weakclf.size(); j++) {
			hpos.col(j) = _weakclf[j]->classify(posx);
			hneg.col(j) = _weakclf[j]->classify(negx);

			Lpos = -trunc_log(
					1 - prod(1 - 1 / (1 + trunc_exp(-(Hpos + hpos.col(j))))))
					/ numpos;
			Lneg = -trunc_log(
					1 - prod(1 - 1 / (1 + trunc_exp(-(Hneg + hneg.col(j))))))
					/ numneg;
			L(j) = Lpos + Lneg;
		}

		uvec indices = sort_index(L);
		uint best;

		BOOST_FOREACH(uint k, indices) {
			if (find(_selectors.begin(), _selectors.end(), k)
					== _selectors.end()) {
				_selectors.push_back(k);
				best = k;
				break;
			}
		}

		Hpos = Hpos + hpos.col(best);
		Hneg = Hneg + hneg.col(best);
	}

	return;
}

// run weak classifiers in _selectors, then combine weak classifier scores
arma::fvec ClfStrong::classify(SampleSet &x, bool logR) {
	int numsamples = x.size();
	fprintf(stderr, "numsamples = %d\n", numsamples);
	fprintf(stderr, "selector size = %d\n", _selectors.size());

	if (!x.ftrsComputed()) {
		Ftr::computeAll(x, _ftrs);
	}

	fmat hjxi = zeros<fmat>(numsamples, _ftrs.size());
//	fvec hj;

	int rowInd = 0;
	BOOST_FOREACH(int j, _selectors) {
		hjxi.col(rowInd) = _weakclf[j]->classify(x);
		rowInd++;
	}

	arma::fvec H = arma::sum(hjxi, 1);

	return H;
}

inline fvec ClfWeak::classify(SampleSet &x) {
	fvec fv = x._ftrVals.col(_ind);
	fvec dev1 = fv - _mu1;
	fvec dev0 = fv - _mu0;
	fvec hj = -square(dev1) / (2 * _sig1) - 0.5 * logf(_sig1)
			+ square(dev0) / (2 * _sig0) - 0.5 * logf(_sig0);

	return hj;
}

void ClfWeak::update(SampleSet &posx, SampleSet &negx) {
	float posmu = 0.0, negmu = 0.0;

	arma::fvec pfv = posx._ftrVals.col(_ind);
	arma::fvec nfv = negx._ftrVals.col(_ind);

	if (_trained) {
		if (posx.size() > 0) {
			_mu1 = _lRate * _mu1 + (1 - _lRate) * arma::mean(pfv);
			_sig1 = _lRate * _sig1
					+ (1 - _lRate) * arma::mean(square(pfv - _mu1));
		}
		if (negx.size() > 0) {
			_mu0 = _lRate * _mu0 + (1 - _lRate) * arma::mean(nfv);
			_sig0 = _lRate * _sig0
					+ (1 - _lRate) * arma::mean(square(nfv - _mu0));
		}
	} else {
		_trained = true;
		if (posx.size() > 0) {
			_mu1 = arma::mean(pfv);
			_sig1 = arma::var(pfv) + 1e-9f;
		}

		if (negx.size() > 0) {
			_mu0 = arma::mean(nfv);
			_sig0 = arma::var(nfv) + 1e-9f;
		}
	}

	_n0 = logf(1e-5 * (1.0f / pow(_sig0, 0.5f)));
	_n1 = logf(1e-5 * (1.0f / pow(_sig1, 0.5f)));
	_e1 = -1.0f / (2.0f * _sig1 + 1e-99);
	_e0 = -1.0f / (2.0f * _sig0 + 1e-99);
}

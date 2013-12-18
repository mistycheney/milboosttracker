// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License 
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#ifndef ONLINEBOOST_H
#define ONLINEBOOST_H

#include "Public.h"
#include "ImageFtr.h"
#include "conversion.h"

class ClfWeak {
public:
	void init(int id, float lRate, Ftr *ftr);
	void update(SampleSet &posx, SampleSet &negx);
	arma::fvec classify(SampleSet &x);

	float _mu0, _mu1, _sig0, _sig1;
	float _n1, _n0;
	float _e1, _e0;
	bool _trained;
	Ftr *_ftr;
	vecFtr *_ftrs;
	int _ind;
	float _lRate;
};

class ClfParams {
public:
	ClfParams() {
		_lRate = 0.85f;
		_numSel = 50;
		_numFeat = 250;
	}
	;

	FtrParams *_ftrParams;
	float _lRate; // learning rate for weak learners;
	int _numFeat;
	int _numSel;
};

class ClfStrong {
public:
	ClfParams *_params;
	vecFtr _ftrs;
	vectori _selectors;
	vector<ClfWeak*> _weakclf;
	uint _numsamples;

public:
	int nFtrs() {
		return _ftrs.size();
	}
	;

	void init(ClfParams *params);
	void update(SampleSet &posx, SampleSet &negx);
	arma::fvec classify(SampleSet &x, bool logR = true);

};

inline void ClfWeak::update(SampleSet &posx, SampleSet &negx) {
	float posmu = 0.0, negmu = 0.0;

	arma::fvec pfv = posx._ftrVals.col(_ind);
	arma::fvec nfv = negx._ftrVals.col(_ind);

	if (posx.size() > 0)
		posmu = arma::mean(pfv);
	if (negx.size() > 0)
		negmu = arma::mean(nfv);

	if (_trained) {
		if (posx.size() > 0) {
			_mu1 = _lRate * _mu1 + (1 - _lRate) * posmu;
			_sig1 = _lRate * _sig1
					+ (1 - _lRate) * arma::mean(square(pfv - _mu1));
		}
		if (negx.size() > 0) {
			_mu0 = _lRate * _mu0 + (1 - _lRate) * negmu;
			_sig0 = _lRate * _sig0
					+ (1 - _lRate) * arma::mean(square(nfv - _mu0));
		}
	} else {
		_trained = true;
		if (posx.size() > 0) {
			_mu1 = posmu;
			_sig1 = arma::var(pfv) + 1e-9f;
		}

		if (negx.size() > 0) {
			_mu0 = negmu;
			_sig0 = arma::var(nfv) + 1e-9f;
		}
	}

	_n0 = logf(1e-5 * (1.0f / pow(_sig0, 0.5f)));
	_n1 = logf(1e-5 * (1.0f / pow(_sig1, 0.5f)));
	_e1 = -1.0f / (2.0f * _sig1 + 1e-99);
	_e0 = -1.0f / (2.0f * _sig0 + 1e-99);
}


#endif

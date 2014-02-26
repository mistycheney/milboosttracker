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
//	vecFtr *_ftrs;
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
	void init(ClfParams *params);
	void update(SampleSet &posx, SampleSet &negx);
	arma::fvec classify(SampleSet &x, bool logR = true);

};

#endif

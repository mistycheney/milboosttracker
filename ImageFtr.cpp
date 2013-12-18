// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License 
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#include "ImageFtr.h"
#include "Sample.h"

/*
 * Compute filter values for all samples in SampleSet.
 * Directly modify _ftrVals in input SampleSet.
 */
void Ftr::computeAll(SampleSet &samples, const vecFtr &ftrs) {

	int numftrs = ftrs.size();
	int numsamples = samples.size();
	if (numsamples == 0)
		return;

	fprintf(stderr, "compute %d feature values for all %d samples \n", numftrs, numsamples);

	samples._ftrVals.zeros(numsamples, numftrs);
//	#pragma omp parallel for
	for (int j = 0; j < numftrs; j++) {
//		cerr << ftrs[f]->_rects << endl;
//		cerr << ftrs[f]->_weights << endl;
		//		#pragma omp parallel for
		for (int i = 0; i < numsamples; i++) {
			samples._ftrVals(i, j) = ftrs[j]->compute(samples[i]);
			;
		}
	}
}

vecFtr Ftr::generateAll(FtrParams *params, uint num) {
	vecFtr ftrs;
	ftrs.resize(num);
	for (uint k = 0; k < num; k++) {
		ftrs[k] = new Ftr();
		ftrs[k]->generate(params);
	}
	return ftrs;
}

/*
 * Generate a filter. A filter is a random number of random boxes.
 */
void Ftr::generate(FtrParams *p) {
	_width = p->_width;
	_height = p->_height;
	int numrects = randint(p->_minNumRect, p->_maxNumRect);
	_rects.resize(numrects);
	_weights.resize(numrects);

	for (int k = 0; k < numrects; k++) {
		_weights[k] = (randfloat() * 2 - 1) / 1000;
//		int wt = (randfloat()*16);
//		_weights[k] = (float)(wt - 8);
		_rects[k].x = randint(0, (uint) (p->_width - 3));
		_rects[k].y = randint(0, (uint) (p->_height - 3));
		_rects[k].width = randint(1, (p->_width - _rects[k].x - 2));
		_rects[k].height = randint(1, (p->_height - _rects[k].y - 2));
	}
}

float Ftr::compute(const Sample &sample) const {

	Rect r;
	float sum = 0.0f;
	int roi_sum = 0;

	//#pragma omp parallel for
	for (int k = 0; k < (int) _rects.size(); k++) {
		r = _rects[k];
		roi_sum = sample._imgII->at<int>(r.x, r.y)
				+ sample._imgII->at<int>(r.x + r.width, r.y + r.height)
				- sample._imgII->at<int>(r.x + r.width, r.y)
				- sample._imgII->at<int>(r.x, r.y + r.height);
		sum += _weights[k] * (float) roi_sum;
	}

	return sum;
}


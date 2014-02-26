// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License 
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#ifndef H_SAMPLE
#define H_SAMPLE

#include "Public.h"
#include <boost/foreach.hpp>

class Sample {
public:
	Sample() {
		_img = NULL;
		_imgII = NULL;
		_row = _col = _height = _width = 0;
		_weight = 1.0f;
	}

//	Sample& operator=(const Sample &a) {
//		_img = a._img;
//		_imgII = a._imgII;
//		_row = a._row;
//		_col = a._col;
//		_width = a._width;
//		_height = a._height;
//		_weight = a._weight;
//
//		return (*this);
//	}

public:
	Mat *_img;
	Mat *_imgII;
	int _row, _col, _width, _height;
	float _weight;

};

class SampleSet {
public:
	SampleSet() {

	}

	SampleSet(const Sample &s) {
		_samples.push_back(s);
	}

	int size() const {
		return _samples.size();
	}
//	void push_back(const Sample &s) {
//		_samples.push_back(s);
//	}

	Sample operator[](const int sample) const {
		return _samples[sample];
	}

	inline bool ftrsComputed() const {
		return (!_ftrVals.empty() && !_samples.empty());
	}

	void clear() {

		_ftrVals.clear();

		BOOST_FOREACH (Sample s, _samples) {
			delete s._imgII;
			delete s._img;
		}

		vector<Sample>().swap(_samples);
	}

	// densly sample the image in a donut shaped region: will take points inside circle of radius inrad,
	// but outside of the circle of radius outrad.  when outrad=0 (default), then just samples points inside a circle
	void sampleImage(Mat *img, int x, int y, int w, int h, float inrad,
			float outrad = 0, int maxnum = 100, int flag=0);

	arma::fmat _ftrVals;

private:
	vector<Sample> _samples;
};

#endif

// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License 
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#ifndef H_IMGFTR
#define H_IMGFTR

//#include "Matrix.h"
#include "Public.h"
#include "Sample.h"

class Ftr;
typedef vector<Ftr*> vecFtr;

class FtrParams {
public:
	int _width, _height;
	int _maxNumRect, _minNumRect;
};

class Ftr {
public:
	Ftr() {
		_width = 0;
		_height = 0;
	}
	;
	float compute(const Sample &sample) const;
	void generate(FtrParams *params);
	Mat visualize() const;

	static void computeAll(SampleSet &samples, const vecFtr &ftrs);
	static vecFtr generateAll(FtrParams *params, uint num);
	uint _width, _height;
	vectorf _weights;
	vector<Rect> _rects;
};

#endif

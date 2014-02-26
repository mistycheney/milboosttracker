// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License 
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#include "Sample.h"
#include <boost/format.hpp>

void SampleSet::sampleImage(Mat *img, int x, int y, int w, int h, float inrad,
		float outrad, int maxnum, int flag) {

	fprintf(stderr,"sample around x=%d y=%d, w=%d, h=%d\n",x,y,w,h);

	float inradsq = inrad * inrad;
	float outradsq = outrad * outrad;
	int dist;

	uint minrow = max(0, (int) y - (int) inrad);
	uint maxrow = min(img->rows, (int) y + (int) inrad);
	uint mincol = max(0, (int) x - (int) inrad);
	uint maxcol = min(img->cols, (int) x + (int) inrad);

	fprintf(stderr,"inrad=%f outrad=%f, minrow=%d maxrow=%d mincol=%d maxcol=%d\n",inrad,outrad,minrow,maxrow,mincol,maxcol);

	_samples.resize((maxrow - minrow + 1) * (maxcol - mincol + 1));
	int i = 0;

	float prob = ((float) (maxnum)) / _samples.size();

	Mat roi;

	//#pragma omp parallel for
	for (int r = minrow; r <= (int) maxrow; r++) {
		for (int c = mincol; c <= (int) maxcol; c++) {
			dist = (y - r) * (y - r) + (x - c) * (x - c);
			if (randfloat() < prob && dist < inradsq && dist >= outradsq) {
//				_samples[i]._img = img;
				_samples[i]._img = new Mat();
				roi = (*img)(Rect(c-w/2,r-h/2,w,h));
				*(_samples[i]._img) = roi;
				_samples[i]._col = c;
				_samples[i]._row = r;
				_samples[i]._height = h;
				_samples[i]._width = w;

				if (_samples[i]._imgII == NULL) {
					_samples[i]._imgII = new Mat();
					integral(*(_samples[i]._img), *(_samples[i]._imgII));
				}

				if (flag>0) {
					if (flag==1) {
						imwrite((boost::format("pos/sample_pos_%d.jpg")%i).str(), roi);
					} else {
						imwrite((boost::format("neg/sample_neg_%d.jpg")%i).str(), roi);
					}
				} else {
					imwrite((boost::format("sample/sample_%d.jpg")%i).str(), roi);
				}

				i++;
			}
		}
	}
	_samples.resize(min(i, maxnum));
	fprintf(stderr, "sampled %d patches\n", _samples.size());
}

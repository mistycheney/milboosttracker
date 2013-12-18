// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License 
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#include "OnlineBoost.h"
#include "Tracker.h"
#include "Public.h"
#include "Sample.h"

bool Tracker::init(Mat *img, TrackerParams *p, ClfParams *clfparams) {
	SampleSet posx, negx;

	_clf = new ClfStrong();
	_clf->init(clfparams);

	_lost = 0;
	_x = p->_initX;
	_y = p->_initY;
	_w = p->_initW;
	_h = p->_initH;
	_scale = p->_initScale;

	fprintf(stderr, "Initializing Tracker..\n");

	// sample positives and negatives from first frame

	posx.sampleImage(img, _x, _y, _w, _h, p->_init_postrainrad);
	negx.sampleImage(img, _x, _y, _w, _h, 2.0f * p->_srchwinsz,
			(1.5f * p->_init_postrainrad), p->_init_negnumtrain);
	if (posx.size() < 1 || negx.size() < 1) {
		fprintf(stderr, "samples not enough.");
		return false;
	}

	// train
	_clf->update(posx, negx);

	fprintf(stderr, "after _clf->update..\n");

	posx.clear();
	negx.clear();

	fprintf(stderr, "after sampleset clear..\n");

	_trparams = p;
	_clfparams = clfparams;

	return true;
}


double Tracker::update_location(Mat *img) {
	fprintf(stderr, "begin update_location\n");

	static SampleSet detectx;
	static arma::fvec confidence;
	double resp;

	// detectx selects the location in which to calculate the classifier;
	// considers all locations within a window around current state;
	// _srchwinsz is the radius of this window
	detectx.sampleImage(img, _x, _y, _w, _h, (float) _trparams->_srchwinsz, 0,
			100);
	fprintf(stderr, "sampleImage\n");

	// run current classifier (_clf) on search window

	struct timespec tp;
	long time_nsec;
	clock_gettime(CLOCK_REALTIME, &tp);
	time_nsec = tp.tv_nsec;

	confidence = _clf->classify(detectx, _trparams->_useLogR);

	clock_gettime(CLOCK_REALTIME, &tp);
	cout << "time " << tp.tv_nsec - time_nsec << endl;

	fprintf(stderr, "classify done\n");

	// find best location
//	{
//		using namespace arma;
//		fvec pr = fvec(confidence);
//		uvec h1 = hist(pr, linspace<fvec>(-2,2,11));
//		cout << h1 << endl;
//	}

	unsigned int bestind;
	resp = confidence.max(bestind);
	_x = (float) detectx[bestind]._col;
	_y = (float) detectx[bestind]._row;

	// clean up
	detectx.clear();
	return resp;
}

void Tracker::update_classifier(Mat *img) {
	fprintf(stderr, "begin update_classifier\n");

	static SampleSet posx, negx;

	// train location clf (negx are randomly selected from image, posx is just the current tracker location)
	negx.sampleImage(img, _x, _y, _w, _h, (1.5f * _trparams->_srchwinsz),
			_trparams->_posradtrain + 5, _trparams->_negnumtrain);
	posx.sampleImage(img, _x, _y, _w, _h, _trparams->_posradtrain, 0,
			_trparams->_posmaxtrain);
	_clf->update(posx, negx);

	// clean up
	posx.clear();
	negx.clear();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
TrackerParams::TrackerParams() {
	_srchwinsz = 30;
	_negsamplestrat = 1;
	_boxcolor.resize(3);
	_boxcolor[0] = 204;
	_boxcolor[1] = 25;
	_boxcolor[2] = 204;
	_lineWidth = 2;
	_negnumtrain = 15;
	_posradtrain = 1;
	_posmaxtrain = 100000;
	_init_negnumtrain = 1000;
	_init_postrainrad = 3;
	_initX = 0;
	_initY = 0;
	_initW = 0;
	_initH = 0;
	_initScale = 1.0;
	_debugv = false;
	_useLogR = true;
	_disp = true;
	_initWithFace = true;
}


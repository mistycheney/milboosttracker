//#include "Matrix.h"
#include "ImageFtr.h"
#include "Tracker.h"
#include "Public.h"
#include <string.h>
#include <algorithm>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <boost/filesystem.hpp>
#include <boost/format.hpp>

#define NUM_TRACKERS 1
#define NUM_SEL_FEATS 50
#define NUM_FEATS 250

using namespace cv;
using namespace boost::filesystem;

GET_TIME_INIT(2);

float distCenters(Tracker *t0, Tracker *t1) {
	float centerX0 = (t0->_x + t0->_w / 2.0) * t0->_scale;
	float centerY0 = (t0->_y + t0->_h / 2.0) * t0->_scale;
	float centerX1 = (t1->_x + t1->_w / 2.0) * t1->_scale;
	float centerY1 = (t1->_y + t1->_h / 2.0) * t1->_scale;
	return (centerX0 - centerX1) * (centerX0 - centerX1)
			+ (centerY0 - centerY1) * (centerY0 - centerY1);
}

bool trackersAgree(Tracker *tr[], int numTrackers) {
	if (numTrackers < 3)
		return true;

	float distThresh = (0.8 * tr[0]->_w) * (0.8 * tr[0]->_w);
	float dist01 = distCenters(tr[0], tr[1]);
	float dist02 = distCenters(tr[0], tr[2]);
	float dist12 = distCenters(tr[1], tr[2]);
	if (dist01 > distThresh || dist02 > distThresh || dist12 > distThresh) {
		if (dist01 < dist02) {
			if (dist01 < dist12) {
				float centerX = ((tr[0]->_x * tr[0]->_scale)
						+ (tr[1]->_x * tr[1]->_scale)) / 2.0;
				float centerY = ((tr[0]->_y * tr[0]->_scale)
						+ (tr[1]->_y * tr[1]->_scale)) / 2.0;
				tr[2]->_x = centerX / tr[2]->_scale;
				tr[2]->_y = centerY / tr[2]->_scale;
			} else {
				float centerX = ((tr[1]->_x * tr[1]->_scale)
						+ (tr[2]->_x * tr[2]->_scale)) / 2.0;
				float centerY = ((tr[1]->_y * tr[1]->_scale)
						+ (tr[2]->_y * tr[2]->_scale)) / 2.0;
				tr[0]->_x = centerX / tr[0]->_scale;
				tr[0]->_y = centerY / tr[0]->_scale;
			}
		} else {
			if (dist02 < dist12) {
				float centerX = ((tr[0]->_x * tr[0]->_scale)
						+ (tr[2]->_x * tr[2]->_scale)) / 2.0;
				float centerY = ((tr[0]->_y * tr[0]->_scale)
						+ (tr[2]->_y * tr[2]->_scale)) / 2.0;
				tr[1]->_x = centerX / tr[1]->_scale;
				tr[1]->_y = centerY / tr[1]->_scale;
			} else {
				float centerX = ((tr[1]->_x * tr[1]->_scale)
						+ (tr[2]->_x * tr[2]->_scale)) / 2.0;
				float centerY = ((tr[1]->_y * tr[1]->_scale)
						+ (tr[2]->_y * tr[2]->_scale)) / 2.0;
				tr[0]->_x = centerX / tr[0]->_scale;
				tr[0]->_y = centerY / tr[0]->_scale;
			}
		}
		return false;
	} else {
		return true;
	}
}

void initParams(TrackerParams *trparams[], FtrParams *ftrparams[],
		ClfParams *clfparams[], float scale, int x, int y, int w, int h,
		int searchRad, int numTrackers, int numFeats, int numSelFeats) {

	for (int i = 0; i < numTrackers; i++) {
		ftrparams[i]->_minNumRect = 2;
		ftrparams[i]->_maxNumRect = 6;
		ftrparams[i]->_width = w / scale;
		ftrparams[i]->_height = h / scale;

		clfparams[i]->_numSel = numSelFeats;
		clfparams[i]->_numFeat = numFeats;
		clfparams[i]->_ftrParams = ftrparams[i];

		trparams[i]->_init_negnumtrain = 65;
		trparams[i]->_init_postrainrad = 3.0f;
		trparams[i]->_srchwinsz = searchRad;
		trparams[i]->_negnumtrain = 65;
		trparams[i]->_posradtrain = 4.0f;
		trparams[i]->_initScale = scale;
		trparams[i]->_initX = x / scale;
		trparams[i]->_initY = y / scale;
		trparams[i]->_initW = w / scale;
		trparams[i]->_initH = h / scale;
	}

	if (numTrackers > 0) {
		trparams[0]->_boxcolor[0] = 25;
		trparams[0]->_boxcolor[1] = 25;
		trparams[0]->_boxcolor[2] = 204;
	}

	if (numTrackers > 1) {
		trparams[1]->_boxcolor[0] = 25;
		trparams[1]->_boxcolor[1] = 204;
		trparams[1]->_boxcolor[2] = 25;
	}
}

//bool withinRad(vectori *xPos, vectori *yPos, int rad, int numFrames) {
//	int xdist, ydist;
//	int zdist = rad * rad;
//	if (xPos->size() >= numFrames) {
//		for (int i = xPos->size() - 2; i >= 0 && i >= xPos->size() - numFrames;
//				i--) {
//			xdist = ((*xPos)[i + 1] - (*xPos)[i])
//					* ((*xPos)[i + 1] - (*xPos)[i]);
//			ydist = ((*yPos)[i + 1] - (*yPos)[i])
//					* ((*yPos)[i + 1] - (*yPos)[i]);
//			if (xdist + ydist > zdist)
//				return false;
//		}
//	}
//	return true;
//}

typedef struct {
	Rect* bb;
	bool* dragging;
} bb_done_struct;

void mouseHandler(int event, int x, int y, int, void* data) {
	bb_done_struct* bb_done = reinterpret_cast<bb_done_struct*>(data);
	Rect* bb = bb_done->bb;
	if (event == EVENT_LBUTTONDOWN) {
		if (!*(bb_done->dragging)) {
			bb->x = x;
			bb->y = y;
			bb->width = 0;
			bb->height = 0;
			*(bb_done->dragging) = true;
		} else {
			*(bb_done->dragging) = false;
		}
	} else if (*(bb_done->dragging) && event == cv::EVENT_MOUSEMOVE) {
		bb->width = x - bb->x;
		bb->height = y - bb->y;
	}
}

int getBBFromUser(Mat *img, Rect *bb) {
	bool done = false;
	bool dragging = false;

	Mat imgrgb, vis;
	cvtColor(*img, imgrgb, cv::COLOR_GRAY2BGR);

	bb_done_struct bb_done;
	bb_done.bb = bb;
	bb_done.dragging = &dragging;
	setMouseCallback("Figure 1", mouseHandler, &bb_done);

	while (!done) {

		imgrgb.copyTo(vis);

		if (!done) {
			rectangle(vis, Point(bb->x, bb->y),
					Point(bb->x + bb->width, bb->y + bb->height),
					Scalar(255, 0, 0));
		}

		char key = waitKey(1);
		if (key == 'q') {
			return 0;
		}
		if ((key == '\n') || (key == '\r')) {
			done = true;
		}

		imshow("Figure 1", vis);
	}

	printf("initialize %d, %d, %d, %d\n", bb->x, bb->y, bb->width, bb->height);

	setMouseCallback("Figure 1", NULL, NULL);
	return 0;
}

void demo(char* images_path, int start_id, char* name_format) {
	Tracker *tr[NUM_TRACKERS];
	TrackerParams *trparams[NUM_TRACKERS];
	ClfParams *clfparams[NUM_TRACKERS];
	FtrParams *ftrparams[NUM_TRACKERS];
	Mat f[NUM_TRACKERS];
	Mat frame, vis, frameRGB, frame_small;
	int initX, initY, initW, initH;
	double ttime = 0.0;

	namedWindow("Figure 1");

	// print usage
	printf("Commands:\n");
	printf("\tPress 'q' to QUIT\n");
	printf("\tPress 'r' to RE-INITIALIZE\n\n");

	// Tracker and parameters
	for (int i = 0; i < NUM_TRACKERS; i++) {
		ftrparams[i] = new FtrParams();
		clfparams[i] = new ClfParams();
		trparams[i] = new TrackerParams();
		tr[i] = new Tracker();
	}

	//	const string filepath = "/home/yuncong/SidLetterTests/g1.mov";
	//	VideoCapture cap(filepath);
	//	if (!cap.isOpened()) {
	//		throw "Cannot open file."; // check if we succeeded
	//	}
	//	cap.read(frameRGB);

	path sequencePath(images_path);

	int ind = start_id;
	string ifn =
			(sequencePath / path((boost::format("%d.jpg") % ind).str())).string();
        frameRGB = imread(ifn, 1);
	cvtColor(frameRGB, frame, COLOR_BGR2GRAY);
//	resize(frame, frame_small, Size(frame.cols / 4, frame.rows / 4));

	Rect bb;
	getBBFromUser(&frame, &bb);

	initX = bb.x + bb.width / 2;
	initY = bb.y + bb.height / 2;
	initW = bb.width;
	initH = bb.height;

	initParams(trparams, ftrparams, clfparams, 1.0, initX, initY, initW, initH,
			25, NUM_TRACKERS, NUM_FEATS, NUM_SEL_FEATS);

	printf("initialize trackers \n");

	for (int t = 0; t < NUM_TRACKERS; t++) {
		tr[t]->init(&frame, trparams[t], clfparams[t]);
	}
	char q = waitKey();

	GET_TIME_VAL(0);

	for (;; ind++) {

		ifn =
				(sequencePath / path((boost::format(name_format) % ind).str())).string();
		frameRGB = imread(ifn, 1);
		cvtColor(frameRGB, frame, COLOR_BGR2GRAY);
//		resize(frame, frame_small, Size(frame.cols / 4, frame.rows / 4));

		fprintf(stderr, "Frame %d\n", ind++);

		// Update location
		for (int t = 0; t < NUM_TRACKERS; t++) {
			resize(frame, f[t],
					Size(frame.cols / tr[t]->_scale,
							frame.rows / tr[t]->_scale));

			tr[t]->update_location(&f[t]);  // grab tracker confidence
		}

		fprintf(stderr, "location updated\n");

		// Check if all trackers agree
		if (trackersAgree(tr, NUM_TRACKERS)) {
			for (int t = 0; t < NUM_TRACKERS; t++)
				tr[t]->update_classifier(&f[t]);
		}

		fprintf(stderr, "classifier updated\n");

		// Draw locations
		cvtColor(frame, vis, COLOR_GRAY2BGR);
		for (int t = 0; t < NUM_TRACKERS; t++) {

			fprintf(stderr, "%d,%d,%d,%d\n", tr[t]->_x, tr[t]->_y, tr[t]->_w,
					tr[t]->_h);

			rectangle(vis,
					Point((tr[t]->_x - tr[t]->_w / 2) * tr[t]->_scale,
							(tr[t]->_y - tr[t]->_h / 2) * tr[t]->_scale),
					Point((tr[t]->_x + tr[t]->_w / 2) * tr[t]->_scale,
							(tr[t]->_y + tr[t]->_h / 2) * tr[t]->_scale),
					Scalar(trparams[t]->_boxcolor[0], trparams[t]->_boxcolor[1],
							trparams[t]->_boxcolor[2]));
		}
		imshow("Figure 1", vis);

		char q = waitKey();
		if (q == 'q')
			break;
		else if (q == 'r') {
			getBBFromUser(&frame, &bb);

			for (int t = 0; t < NUM_TRACKERS; t++) {
				trparams[t]->_initX = bb.x + bb.width / 2;
				trparams[t]->_initY = bb.y + bb.height / 2;
				trparams[t]->_initW = bb.width;
				trparams[t]->_initH = bb.height;

				tr[t]->init(&frame, trparams[t], clfparams[t]);
			}
			waitKey();
		}
	}
}

int main(int argc, char * argv[]) {
	if (argc != 4) {
            cout << "usage: "<< argv[0] << " <images path> <start_frame> <name_format>\n";
        } else {
            demo(argv[1], atoi(argv[2]), argv[3]);
        }
}


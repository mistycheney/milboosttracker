// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License 
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

// Some of the vector functions and the StopWatch class are based off code by Piotr Dollar (http://vision.ucsd.edu/~pdollar/)

#ifndef H_PUBLIC
#define H_PUBLIC

#define CLOCKS_PER_SEC (2.6 * 1000000000)
#define _CRT_SECURE_NO_WARNINGS

#include <typeinfo>

#include <iostream>
#include <fstream>
#include <cmath>
#include <new>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cassert>
#include <algorithm> 
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <list>
#include <math.h>
#include <sys/time.h>
//#include "ipp.h"

//#pragma comment(lib,"ippi.lib")
//#pragma comment(lib,"ippm.lib")
//#pragma comment(lib,"ippcore.lib")
//#pragma comment(lib,"ippcv.lib")

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <armadillo>

//#include "opencv/highgui.h"
//#include "opencv/cv.h"

//#pragma comment(lib,"cxcore.lib")
//#pragma comment(lib,"highgui.lib")
//#pragma comment(lib,"cvhaartraining.lib")
//#pragma comment(lib,"ml.lib")
//#pragma comment(lib,"cvaux.lib")
//#pragma comment(lib,"cxts.lib")

//#include "omp.h"
#include "timer.h"

using namespace std;
using namespace cv;

typedef unsigned char  uchar;
typedef unsigned short ushort;
typedef unsigned int   uint;
typedef unsigned long  ulong;

typedef vector<float>	vectorf;
typedef vector<double>	vectord;
typedef vector<int>		vectori;
typedef vector<long>	vectorl;
typedef vector<uchar>	vectoru;
typedef vector<string>	vectorString;
typedef vector<bool>	vectorb;

#define	PI	3.1415926535897931
#define PIINV 0.636619772367581
#define INF 1e99
#define INFf 1e50f
#define EPS 1e-99;
#define EPSf 1e-50f
#define ERASELINE "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b"

#define  sign(s)	((s > 0 ) ? 1 : ((s<0) ? -1 : 0))
#define  round(v)   ((int) (v+0.5))

//static CvRNG rng_state = cvRNG((int)time(NULL));
static RNG rng( 0xFFFFFFFF );
//static CvRNG rng_state = cvRNG(1);

//////////////////////////////////////////////////////////////////////////////////////////////////////
// Debug timing stuff
#ifdef DTIME
	#define NUMTIMES 30
	extern struct timeval gTv;
	extern double gTime[NUMTIMES];
	extern double gTTime[10];

	#define INITIME()		for (int rof=0; rof <= NUMTIMES; rof++) gTime[rof] = 0.0;
	#define STRTIME(tt)		gettimeofday(&gTv, NULL); \
							gTTime[tt] = gTv.tv_sec + gTv.tv_usec/1000000.0;
	#define ELPTIMEC(t, tt)	gettimeofday(&gTv, NULL); \
							gTime[t] = gTime[t] + (gTv.tv_sec + gTv.tv_usec/1000000.0) - gTTime[tt]; \
							gTTime[tt] = gTv.tv_sec + gTv.tv_usec/1000000.0;
	#define ELPTIME(t, tt)	gettimeofday(&gTv, NULL); \
							gTime[t] = (gTv.tv_sec + gTv.tv_usec/1000000.0) - gTTime[tt]; \
							gTTime[tt] = gTv.tv_sec + gTv.tv_usec/1000000.0;
	#define TIMEMS(t)		(gTime[t]*1000)
#else
	#define INITIME()
	#define STRTIME(tt)
	#define ELPTIMEC(t, tt)
	#define ELPTIME(t, tt)
	#define TIMEMS(t)
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////
// random generator stuff
//void				randinitalize( const int init );
int					randint( const int min=0, const int max=5 );
vectori				randintvec( const int min=0, const int max=5, const uint num=100 );
vectorf				randfloatvec( const uint num=100 );
float				randfloat();
float				randgaus(const float mean, const float std);
vectorf				randgausvec(const float mean, const float std, const int num=100);
vectori				sampleDisc(const vectorf &weights, const uint num=100);

inline float		sigmoidf(float x)
{
	return 1.0f/(1.0f+expf(-x));
}
inline double		sigmoid(double x)
{
	return 1.0/(1.0+exp(-x));
}

inline vectorf		sigmoid(vectorf x)
{
	vectorf r(x.size());
	for( uint k=0; k<r.size(); k++ )
		r[k] = sigmoid(x[k]);
	return r;

}

inline double		oneminussigmoid(double x)
{
	if (x < 0) {
		return 1.0/(1.0+exp(x));
	}
	else {
		double ep = exp(-x);
		return ep/(1.0+ep);
	}
}

inline int			force_between(int i, int mn, int mx)
{
	return min(max(i,mn),mx);
}

string				int2str( int i, int ndigits );
//////////////////////////////////////////////////////////////////////////////////////////////////////
// vector functions
template<class T> class				SortableElement
{
public:
	T _val; int _ind;
	SortableElement() {};
	SortableElement( T val, int ind ) { _val=val; _ind=ind; }
	bool operator< (const SortableElement &b ) const { return (_val > b._val ); };
};

template<class T> class				SortableElementRev
{
public:
	T _val; int _ind;
	SortableElementRev() {};
	SortableElementRev( T val, int ind ) { _val=val; _ind=ind; }
	bool operator< (const SortableElementRev &b ) const { return ((float)_val < (float)b._val ); };
};

template<class T> void sort_order( vector<T> &v, vectori &order )
{
	uint n=(uint)v.size();
	vector< SortableElement<T> > v2; 
	v2.resize(n); 
	order.clear(); order.resize(n);
	for( uint i=0; i<n; i++ ) {
		v2[i]._ind = i;
		v2[i]._val = v[i];
	}
	std::sort( v2.begin(), v2.end() );
	for( uint i=0; i<n; i++ ) {
		order[i] = v2[i]._ind;
		v[i] = v2[i]._val;
	}
};

template<class T> void sort_order_asc( vector<T> &v, vectori &order )
{
	uint n=(uint)v.size();
	vector< SortableElementRev<T> > v2; 
	v2.resize(n); 
	order.clear(); order.resize(n);
	for( uint i=0; i<n; i++ ) {
		v2[i]._ind = i;
		v2[i]._val = v[i];
	}
	std::sort( v2.begin(), v2.end() );
	for( uint i=0; i<n; i++ ) {
		order[i] = v2[i]._ind;
		v[i] = v2[i]._val;
	}
};

template<class T> void resizeVec( vector< vector<T> > &v, int sz1, int sz2, T val=0)
{
	v.resize(sz1);
	for( int k=0; k<sz1; k++ )
		v[k].resize(sz2,val);
};



template<class T> inline uint		min_idx( const vector<T> &v )
{
	return (uint)(distance(v.begin(), min_element(v.begin(),v.end())));
}
template<class T> inline uint		max_idx( const vector<T> &v )
{
	return (uint)(distance(v.begin(), max_element(v.begin(),v.end())));
}

template<class T> inline void		normalizeVec( vector<T> &v )
{
	T sum = 0;
	for( uint k=0; k<v.size(); k++ ) sum+=v[k];
	for( uint k=0; k<v.size(); k++ ) v[k]/=sum;
}


template<class T> ostream&			operator<<(ostream& os, const vector<T>& v)
{  //display vector
	os << "[ " ;
	for (size_t i=0; i<v.size(); i++)
		os << v[i] << " ";
	os << "]";
	return os;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////
// error functions
inline void							abortError( const int line, const char *file, const char *msg=NULL) 
{
	if( msg==NULL )
		fprintf(stderr, "%s %d: ERROR\n", file, line );
	else
		fprintf(stderr, "%s %d: ERROR: %s\n", file, line, msg );
	exit(0);
}



//////////////////////////////////////////////////////////////////////////////////////////////////////
// Stop Watch
class								StopWatch
{
public:
	StopWatch() { Reset(); }
	StopWatch(bool start) { Reset(); if(start) Start(); }
	
	inline void Reset(bool restart=false) { 
		totaltime=0; 
		running=false; 
		if(restart) Start();
	}

	inline double Elapsed(bool restart=false) { 
		if(running) Stop();
		if(restart) Start();
		return totaltime; 
	}

	inline char* ElapsedStr(bool restart=false) { 
		if(running) Stop();
		if( totaltime < 60.0f )
			sprintf( totaltimeStr, "%5.2fs", totaltime );
		else if( totaltime < 3600.0f )
			sprintf( totaltimeStr, "%5.2fm", totaltime/60.0f );
		else 
			sprintf( totaltimeStr, "%5.2fh", totaltime/3600.0f );
		if(restart) Start();
		return totaltimeStr; 
	}

	inline void Start() {
		assert(!running); 
		running=true;
		sttime = clock();
	}

	inline void Stop() {
		totaltime += ((double) (clock() - sttime)) / CLOCKS_PER_SEC;
		assert(running);
		running=false;
	}

protected:
	bool running;
	clock_t sttime;
	double totaltime;
	char totaltimeStr[100];
};


#endif

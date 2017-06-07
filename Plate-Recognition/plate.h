#ifndef _PLATE_H_
#define _PLATE_H_


////////////////////////////////////////////////////////////////////////////////
// includes
////////////////////////////////////////////////////////////////////////////////
#include<opencv2\opencv.hpp>
#include<iostream>
#include<string>
#include<map>
#include<set>
#include<queue>
#include<functional>
#include<algorithm>
#include<cstring>

using namespace std;
using namespace cv;


////////////////////////////////////////////////////////////////////////////////
// macros
////////////////////////////////////////////////////////////////////////////////
#define SCR_WIDTH 12
#define SCR_HEIGHT 24
#define SCR_OUTPUTSIZE 34
#define SCR_RATIO 2

#define CNR_WIDTH 16
#define CNR_HEIGHT 28
#define CNR_OUTPUTSIZE 31
#define CNR_RATIO 1.75

#define PLT_WIDTH 136
#define PLT_HEIGHT 36
#define PLT_RATIO (36.0 / 136)


////////////////////////////////////////////////////////////////////////////////
// global variables
////////////////////////////////////////////////////////////////////////////////
extern map<int, char> scr_code2label;
extern Mat scr_mat_w, scr_mat_b;
extern float scr_w[SCR_WIDTH * SCR_HEIGHT * SCR_OUTPUTSIZE], scr_b[SCR_OUTPUTSIZE];

extern map<int, string> cnr_code2label;
extern Mat cnr_mat_w, cnr_mat_b;
extern float cnr_w[CNR_WIDTH * CNR_HEIGHT * CNR_OUTPUTSIZE], cnr_b[CNR_OUTPUTSIZE];


////////////////////////////////////////////////////////////////////////////////
// functions
////////////////////////////////////////////////////////////////////////////////

// initialize global data, including 2 maps and 4 matrices
void init_global_data();

// find the index that have the greatest number in given 1-d array
int max_index(const Mat &res);

// print binary image in command lines
void print_bin_image(const Mat &src);


////////////////////////////////////////////////////////////////////////////////
// class definitions
////////////////////////////////////////////////////////////////////////////////

//
// The definition of class plate_t
//
class plate_t {
public:
	string province;
	char city;
	char code[5];
	bool valid;

	// constructor
	plate_t();

	// overload of operator << for output stream
	friend ostream &operator<<(ostream &o, const plate_t &p);
};

//
// The definition of class block_t
class block {
public:
	int xmin, xmax, ymin, ymax;

	// constructor
	block(int x1, int x2, int y1, int y2);

	// number of columns
	int width() const;

	// number of rows
	int height() const;

	// number of size
	int size() const;

	// compare by size
	friend bool operator>(const block &a, const block &b);

	// compare by position
	friend bool pos_less(const block &a, const block &b);
};

#endif

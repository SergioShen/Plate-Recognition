#include<opencv2\opencv.hpp>
#include<iostream>
#include<string>
#include<map>
#include<set>
#include<queue>
#include<functional>
#include<algorithm>
#include<cstring>

using namespace cv;
using namespace std;

#define SCR_WIDTH 12
#define SCR_HEIGHT 24
#define SCR_OUTPUTSIZE 34
#define SCR_RATIO 2

// global variables
map<int, char> scr_code2label;
Mat scr_mat_w;
Mat scr_mat_b;
float scr_w[SCR_WIDTH * SCR_HEIGHT * SCR_OUTPUTSIZE], scr_b[SCR_OUTPUTSIZE];

void init_global_data() {
	// initialize scr_code2label map
	scr_code2label[0] = '0'; scr_code2label[1] = '1';
	scr_code2label[2] = '2'; scr_code2label[3] = '3';
	scr_code2label[4] = '4'; scr_code2label[5] = '5';
	scr_code2label[6] = '6'; scr_code2label[7] = '7';
	scr_code2label[8] = '8'; scr_code2label[9] = '9';
	scr_code2label[10] = 'A'; scr_code2label[11] = 'B';
	scr_code2label[12] = 'C'; scr_code2label[13] = 'D';
	scr_code2label[14] = 'E'; scr_code2label[15] = 'F';
	scr_code2label[16] = 'G'; scr_code2label[17] = 'H';
	scr_code2label[18] = 'J'; scr_code2label[19] = 'K';
	scr_code2label[20] = 'L'; scr_code2label[21] = 'M';
	scr_code2label[22] = 'N'; scr_code2label[23] = 'P';
	scr_code2label[24] = 'Q'; scr_code2label[25] = 'R';
	scr_code2label[26] = 'S'; scr_code2label[27] = 'T';
	scr_code2label[28] = 'U'; scr_code2label[29] = 'V';
	scr_code2label[30] = 'W'; scr_code2label[31] = 'X';
	scr_code2label[32] = 'Y'; scr_code2label[33] = 'Z';


	// initialize matrices
	ifstream scr_file_w("scr_array_w", ios::in | ios::binary),
				scr_file_b("scr_array_b", ios::in | ios::binary);

	scr_file_w.read((char *)scr_w, sizeof(scr_w));
	scr_file_b.read((char *)scr_b, sizeof(scr_b));


	scr_mat_w = Mat(SCR_WIDTH * SCR_HEIGHT, SCR_OUTPUTSIZE, CV_32FC1, scr_w);
	scr_mat_b = Mat(1, SCR_OUTPUTSIZE, CV_32FC1, scr_b);
}

Mat scr_image_pre_process(const Mat &src) {
	Mat temp, temp1, temp2, temp3, temp4, result;
	int width, height, x, y;

	// convert to gray
	cvtColor(src, temp, CV_BGR2GRAY);

	// resize
	resize(temp1, temp2, Size(SCR_WIDTH, SCR_HEIGHT));

	// convert to float32
	temp2.convertTo(temp3, CV_32FC1);

	// convert to binary value
	threshold(temp3, temp4, mean(temp3)[0], 1.0, CV_THRESH_BINARY);

	// reshape as an 1d-vector
	result = temp4.reshape(1, 1);

	return result;
}

int max_index(const Mat &res) {
	float max = res.at<float>(0, 0);
	int max_index = 0;
	for (int i = 1; i < res.cols; i++)
		if (res.at<float>(0, i) > max) {
			max = res.at<float>(0, i);
			max_index = i;
		}
	return max_index;
}

char scr_recognition(const Mat &pred_image) {
	return scr_code2label[max_index(pred_image * scr_mat_w + scr_mat_b)];
}

int scr_main(int argc, char **argv) {
	bool show = false;
	init_global_data();
	Mat image, mat_image;

	image = imread("test_images\\K.bmp");

	//
	// image convert
	//
	mat_image = scr_image_pre_process(image);

	//
	// compute
	//
	cout << scr_recognition(mat_image) << endl;

	return 0;
}

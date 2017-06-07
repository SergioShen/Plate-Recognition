#include"plate.h"

map<int, char> scr_code2label;
Mat scr_mat_w, scr_mat_b;
float scr_w[SCR_WIDTH * SCR_HEIGHT * SCR_OUTPUTSIZE], scr_b[SCR_OUTPUTSIZE];

map<int, string> cnr_code2label;
Mat cnr_mat_w, cnr_mat_b;
float cnr_w[CNR_WIDTH * CNR_HEIGHT * CNR_OUTPUTSIZE], cnr_b[CNR_OUTPUTSIZE];

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

	// initialize cnr_code2label map
	cnr_code2label[0] = "皖"; cnr_code2label[1] = "京";
	cnr_code2label[2] = "渝"; cnr_code2label[3] = "闽";
	cnr_code2label[4] = "甘"; cnr_code2label[5] = "粤";
	cnr_code2label[6] = "桂"; cnr_code2label[7] = "贵";
	cnr_code2label[8] = "琼"; cnr_code2label[9] = "冀";
	cnr_code2label[10] = "黑"; cnr_code2label[11] = "豫";
	cnr_code2label[12] = "鄂"; cnr_code2label[13] = "湘";
	cnr_code2label[14] = "苏"; cnr_code2label[15] = "赣";
	cnr_code2label[16] = "吉"; cnr_code2label[17] = "辽";
	cnr_code2label[18] = "内"; cnr_code2label[19] = "宁";
	cnr_code2label[20] = "青"; cnr_code2label[21] = "鲁";
	cnr_code2label[22] = "沪"; cnr_code2label[23] = "陕";
	cnr_code2label[24] = "晋"; cnr_code2label[25] = "川";
	cnr_code2label[26] = "津"; cnr_code2label[27] = "新";
	cnr_code2label[28] = "藏"; cnr_code2label[29] = "云";
	cnr_code2label[30] = "浙";

	// initialize matrices
	ifstream scr_file_w("scr_array_w", ios::in | ios::binary),
		scr_file_b("scr_array_b", ios::in | ios::binary);
	ifstream cnr_file_w("cnr_array_w", ios::in | ios::binary),
		cnr_file_b("cnr_array_b", ios::in | ios::binary);

	scr_file_w.read((char *)scr_w, sizeof(scr_w));
	scr_file_b.read((char *)scr_b, sizeof(scr_b));

	cnr_file_w.read((char *)cnr_w, sizeof(cnr_w));
	cnr_file_b.read((char *)cnr_b, sizeof(cnr_b));

	scr_mat_w = Mat(SCR_WIDTH * SCR_HEIGHT, SCR_OUTPUTSIZE, CV_32FC1, scr_w);
	scr_mat_b = Mat(1, SCR_OUTPUTSIZE, CV_32FC1, scr_b);

	cnr_mat_w = Mat(CNR_WIDTH * CNR_HEIGHT, CNR_OUTPUTSIZE, CV_32FC1, cnr_w);
	cnr_mat_b = Mat(1, CNR_OUTPUTSIZE, CV_32FC1, cnr_b);
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

void print_bin_image(const Mat & src) {
	cout << " ";
	for (int j = 0; j < src.cols; j++)
		cout << j % 10;
	cout << endl;
	for (int i = 0; i < src.rows; i++) {
		cout << i % 10;
		for (int j = 0; j < src.cols; j++) {
			if (src.at<int>(i, j) > 0)
				cout << "#";
			else
				cout << " ";
		}
		cout << endl;
	}
}

ostream & operator<<(ostream & o, const plate_t & p)
{
	if (p.valid) {
		cout << p.province << p.city;
		for (int i = 0; i < 5; i++)
			cout << p.code[i];
		cout << endl;
	}
	else {
		cout << "Plate recognition fail." << endl;
	}
	return o;
}

bool operator>(const block & a, const block & b) {
	if (a.size() > b.size())
		return true;
	else if (a.size() == b.size())
		return a.ymin > b.ymin;
	else
		return false;
}

bool pos_less(const block & a, const block & b) {
	return a.ymin < b.ymin;
}

block::block(int x1, int x2, int y1, int y2) { }

int block::width() const {
	return ymax - ymin + 1;
}

int block::height() const {
	return xmax - xmin + 1;
}

int block::size() const {
	return width() * height();
}

plate_t::plate_t() { }

#include "recognition.h"

int scr_recognition(const Mat & pred_image) {
	return max_index(pred_image * scr_mat_w + scr_mat_b);
}

int cnr_recognition(const Mat & pred_image) {
	return max_index(pred_image * cnr_mat_w + cnr_mat_b);
}

Mat plate_image_pre_process(const Mat & src) {
	Mat gray, std, std_bin, result;

	// convert to gray
	cvtColor(src, gray, CV_BGR2GRAY);

	// resize
	resize(gray, std, Size(PLT_WIDTH, PLT_HEIGHT));

	// convert to binary image
	threshold(std, std_bin, mean(std)[0], 1, CV_THRESH_BINARY);

	// convert to single channel
	std_bin.convertTo(result, CV_32SC1);
	return result;
}

Mat scr_image_pre_process(const Mat & src) {
	Mat float_image, std_float, result;

	// convert to float32
	src.convertTo(float_image, CV_32FC1);

	// resize
	resize(float_image, std_float, Size(SCR_WIDTH, SCR_HEIGHT));

	// reshape as an 1d-vector
	result = std_float.reshape(1, 1);

	return result;
}

Mat cnr_image_pre_process(const Mat & src) {
	Mat float_image, std_float, result;

	// convert to float32
	src.convertTo(float_image, CV_32FC1);

	// resize
	resize(float_image, std_float, Size(CNR_WIDTH, CNR_HEIGHT));

	// reshape as an 1d-vector
	result = std_float.reshape(1, 1);

	return result;
}

plate_t plate_rlt_cut_recognition(const Mat &src) {
	plate_t result;
	
	Mat full_cut_image = cut_edge(src);

	int row_sum[PLT_HEIGHT], col_sum[PLT_WIDTH];

	// calculate new image's row & col sum
	memset(row_sum, 0, sizeof(row_sum));
	memset(col_sum, 0, sizeof(col_sum));
	for (int i = 0; i < full_cut_image.rows; i++) {
		for (int j = 0; j < full_cut_image.cols; j++) {
			row_sum[i] += full_cut_image.at<int>(i, j);
			col_sum[j] += full_cut_image.at<int>(i, j);
		}
	}

	// detach single chars
	int currcode = 0;
	int j1 = 0;
	while (j1 < full_cut_image.cols) {
		if (currcode >= 7)
			break;

		while (col_sum[j1] < 5)
			j1++;
		int j2 = j1;
		while (col_sum[j2] >= 5)
			j2++;

		// check the f**king point
		if (j2 - j1 < 8) {
			int sum = 0;
			for (int j = j1; j < j2; j++)
				sum += col_sum[j];
			if (sum < 25) { // regard as that f**king point, ignore it
				j1 = j2;
				continue;
			}
		}

		// standardlize the size by ratio
		int left = j1, right = j2;
		int std_width;
		if (currcode == 0)
			std_width = full_cut_image.rows / CNR_RATIO;
		else
			std_width = full_cut_image.rows / SCR_RATIO;

		while (right - left < std_width) {
			left--;
			right++;
		}

		// safety check
		if (left < 0)
			left = 0;
		if (right > full_cut_image.cols)
			right = full_cut_image.cols;

#ifdef DEBUG
		cout << "Block" << currcode << ": " << left << " " << right << endl;
		print_bin_image(full_cut_image(Range(0, full_cut_image.rows), Range(left, right)));
#endif

		if (currcode == 0) {
			result.province = cnr_code2label[
				cnr_recognition(
					cnr_image_pre_process(
						full_cut_image(Range(0, full_cut_image.rows), Range(left, right))))];
		}
		else if (currcode == 1) {
			result.city = scr_code2label[
				scr_recognition(
					scr_image_pre_process(
						full_cut_image(Range(0, full_cut_image.rows), Range(left, right))))];
		}
		else {
			result.code[currcode - 2] = scr_code2label[
				scr_recognition(
					scr_image_pre_process(
						full_cut_image(Range(0, full_cut_image.rows), Range(left, right))))];
		}
		currcode++;
		j1 = j2;
	}
	if (currcode == 7)
		result.valid = true;

	return result;
}

Mat cut_edge(const Mat & src) {
	int row_sum[PLT_HEIGHT], col_sum[PLT_WIDTH];
	memset(row_sum, 0, sizeof(row_sum));
	memset(col_sum, 0, sizeof(col_sum));

#ifdef DEBUG
	print_bin_image(src);
#endif

	// calculate row sum
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			row_sum[i] += src.at<int>(i, j);
		}
	}

	// cut up & down
	int up = PLT_HEIGHT / 2, down = PLT_HEIGHT / 2;
	while (up > 0 && row_sum[up] >= 7)
		up--;
	up++;
	while (down < PLT_HEIGHT - 1 && row_sum[down] >= 7)
		down++;

	// cut safety check
	if (down - up < PLT_HEIGHT / 2) {
		up = 0;
		down = PLT_HEIGHT;
	}

	Mat udcut_image = src(Range(up, down), Range(0, PLT_WIDTH));

#ifdef DEBUG
	print_bin_image(udcut_image);
#endif

	// calculate col sum
	for (int i = 0; i < udcut_image.rows; i++) {
		for (int j = 0; j < udcut_image.cols; j++) {
			col_sum[j] += udcut_image.at<int>(i, j);
		}
	}

	// cut left & right
	int block_cnt = 0;
	for (int j = 0; j < udcut_image.cols; j++) {
		if (col_sum[j] >= 5) {
			block_cnt++;
			while (j < udcut_image.cols - 1 && col_sum[j] >= 5)
				j++;
		}
	}

	int left = 0, right = udcut_image.cols;
	if (block_cnt > 8) { // need to cut left and right bound
		while (col_sum[left] < 5)
			left++;
		while (col_sum[left] >= 5)
			left++;
		while (col_sum[left] < 5)
			left++;

		right--;
		while (col_sum[right] < 5)
			right--;
		while (col_sum[right] >= 5)
			right--;
		while (col_sum[right] < 5)
			right--;
		right++;

		// cut safety check
		if (left >= 8)
			left = 0;
		if (udcut_image.cols - right >= 8)
			right = udcut_image.cols;
	}
	Mat dual_cut_image = udcut_image(Range(0, udcut_image.rows), Range(left, right));

#ifdef DEBUG
	print_bin_image(dual_cut_image);
#endif

	memset(row_sum, 0, sizeof(row_sum));

	// second cut of up & down
	// calculate row sum
	for (int i = 0; i < dual_cut_image.rows; i++) {
		for (int j = 0; j < dual_cut_image.cols; j++) {
			row_sum[i] += dual_cut_image.at<int>(i, j);
		}
	}

	// cut up & down
	up = dual_cut_image.rows / 2; down = dual_cut_image.rows / 2;
	while (up > 0 && row_sum[up] >= 7)
		up--;
	up++;
	while (down < dual_cut_image.rows - 1 && row_sum[down] >= 7)
		down++;

	// cut safety check
	if (down - up < dual_cut_image.rows / 2) {
		up = 0;
		down = dual_cut_image.rows;
	}

	Mat full_cut_image = dual_cut_image(Range(up, down), Range(0, dual_cut_image.cols));

#ifdef DEBUG
	print_bin_image(full_cut_image);
#endif

	return full_cut_image;
}


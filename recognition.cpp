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

plate_t plate_dfs_cut_recognition(const Mat & src) {
	plate_t result;
	bool **visited;

#ifdef DEBUG
	print_bin_image(src);
#endif

	// build @visited array
	visited = new bool*[src.rows];
	for (int i = 0; i < src.rows; i++) {
		visited[i] = new bool[src.cols];
		memset(visited[i], 0, src.cols * sizeof(bool));
	}

	// dfs to find all the blocks
	vector<block> pq;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<int>(i, j) > 0 && !visited[i][j]) {
				block new_block(i, i, j, j);
				dfs(src, i, j, visited, new_block);

				// filter invalid blocks roughly
				if (new_block.ymin < 12 || new_block.width() < new_block.height())
					pq.push_back(new_block);
			}
		}
	}

	// safety check
	if (pq.size() < 7)
		return result;

	// collect 7 bigest blocks
	vector<block> blks;
	bool have_edge = false;
	block edge(0, 0, 0, 0);

	sort(pq.begin(), pq.end(), greater<block>());
	vector<block>::iterator it = pq.begin();
	for (int i = 0; i < 7; i++) {
		if (it == pq.end())
			return result;

		block temp = *it; it++;

		// edge identify
		if (temp.size() > src.cols * src.rows / 4) {
			have_edge = true;
			edge = temp;
			i--;
			continue;
		}
		blks.push_back(temp);
	}

	// sort by position
	sort(blks.begin(), blks.end(), pos_less);
	blks[0].ymax = max(blks[1].ymin - 2, blks[0].ymax);
	blks[0].ymin = 0;

	// recognition
	for (int i = 0; i < 7; i++) {
		if (i == 0) {
			int xmin = blks[i].xmin, xmax = blks[i].xmax, ymin = blks[i].ymin, ymax = blks[i].ymax;
			int *row_sum = new int[src.rows], *col_sum = new int[src.cols];
			memset(row_sum, 0, sizeof(int) * src.rows);
			memset(col_sum, 0, sizeof(int) * src.cols);
			for (int i = 0; i < src.rows; i++) {
				for (int j = 0; j < src.cols; j++) {
					row_sum[i] += src.at<int>(i, j);
					col_sum[j] += src.at<int>(i, j);
				}
			}
			while (xmin > 0 && 4 * row_sum[xmin - 1] > row_sum[xmin])
				xmin--;
			while (xmax < src.rows - 1 && 4 * row_sum[xmax + 1] > row_sum[xmax])
				xmax++;
			if (have_edge) {
				if (edge.ymin <= 2) {
					ymin = edge.ymin;
					while (ymin < ymax / 2 && 4 * col_sum[ymin + 1] > col_sum[ymin])
						ymin++;
					ymin += 2;
				}
			}

#ifdef DEBUG
			print_bin_image(src(Range(xmin, xmax + 1), Range(ymin, ymax + 1)));
#endif

			result.province = cnr_code2label[
				cnr_recognition(
					cnr_image_pre_process(
						src(Range(xmin, xmax + 1), Range(ymin, ymax + 1))))];
		}
		else {
			// fix the '1' case: extend to 1:2
			if ((float)blks[i].height() / blks[i].width() > 2.0) {
				int mid = (blks[i].ymin + blks[i].ymax) / 2;
				blks[i].ymin = mid - blks[i].height() / 4;
				if (blks[i].ymin < 0)
					blks[i].ymin = 0;
				blks[i].ymax = mid + blks[i].height() / 4;
				if (blks[i].ymax >= src.cols)
					blks[i].ymax = src.cols - 1;
			}

#ifdef DEBUG
			print_bin_image(src(Range(blks[i].xmin, blks[i].xmax + 1), Range(blks[i].ymin, blks[i].ymax + 1)));
#endif

			if (i == 1) {
				result.city = scr_code2label[
					scr_recognition(
						scr_image_pre_process(
							src(Range(blks[i].xmin, blks[i].xmax + 1), Range(blks[i].ymin, blks[i].ymax + 1))))];
			}
			else {
				result.code[i - 2] = scr_code2label[
					scr_recognition(
						scr_image_pre_process(
							src(Range(blks[i].xmin, blks[i].xmax + 1), Range(blks[i].ymin, blks[i].ymax + 1))))];
			}
		}
	}
	result.valid = true;
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

plate_t plate_edge_cut_dfs_cut_recognition(const Mat & src) {
	Mat full_cut_image = cut_edge(src);
	return plate_dfs_cut_recognition(full_cut_image);
}

void dfs(const Mat & src, int x, int y, bool **visited, block & curr_block) {
	if (x < curr_block.xmin)
		curr_block.xmin = x;
	if (x > curr_block.xmax)
		curr_block.xmax = x;
	if (y < curr_block.ymin)
		curr_block.ymin = y;
	if (y > curr_block.ymax)
		curr_block.ymax = y;
	visited[x][y] = true;

	int direction[4][2] = { 0,1,0,-1,1,0,-1,0 };

	for (int i = 0; i < 4; i++) {
		int nx = x + direction[i][0], ny = y + direction[i][1];
		if (nx >= 0 && nx < src.rows && ny >= 0 && ny < src.cols) {
			if (src.at<int>(nx, ny) > 0 && !visited[nx][ny])
				dfs(src, nx, ny, visited, curr_block);
		}
	}
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

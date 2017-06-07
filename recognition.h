#ifndef _RECOGNITION_H_
#define _RECOGNITION_H_

////////////////////////////////////////////////////////////////////////////////
// includes
////////////////////////////////////////////////////////////////////////////////
#include"plate.h"

////////////////////////////////////////////////////////////////////////////////
// functions
////////////////////////////////////////////////////////////////////////////////

// find the max index of scr result
int scr_recognition(const Mat &pred_image);

// find the max index of cnr result
int cnr_recognition(const Mat &pred_image);

// convert to standard, one-channel, binary image
Mat plate_image_pre_process(const Mat &src);

// convert to standard, float-valued, i-d array
Mat scr_image_pre_process(const Mat &src);

// convert to standard, float-valued, i-d array
Mat cnr_image_pre_process(const Mat &src);

// cut by depth-first-search to find joint-blocks
plate_t plate_dfs_cut_recognition(const Mat &src);

// cut by analysing the values of every row and cloumn
plate_t plate_rlt_cut_recognition(const Mat &src);

// cut by cuting edge and df-search
plate_t plate_edge_cut_dfs_cut_recognition(const Mat &src);

// depth-first-search function
void dfs(const Mat &src, int x, int y, bool **visited, block &curr_block);

// cut edge surrounds the plate
Mat cut_edge(const Mat &src);

#endif

/**
 * Basic Structures.
 * 2020/11/20
 */
#ifndef STRUCT_H
#define STRUCT_H

#include <iostream>
#include <vector>
#include <array>
#include <string>

typedef std::vector<std::vector<std::array<float, 5>>> BatchBox;
typedef std::pair<std::vector<std::vector<std::array<float, 5>>>, std::vector<std::vector<std::vector<float>>>> TrackRes;

struct Bbox{
	float xmin;
	float ymin;
	float xmax;
	float ymax;
	float score;
	int cid;
	int w;
	int h;
	int fea_index;
};

struct SCropResizePara
{
	unsigned char *dstArray;
	unsigned char *dstTmp;
	unsigned char* cropImage;
	unsigned char* dstImage;
	unsigned char* srcImage;
	float xScaleS_R, yScaleS_R; //the size ratio between src image and resized image.
	short srcXoffset=0, srcYoffset=0;// srcXoffset do not * channels, simily to srcYoffset.
	unsigned short srcWidthBites, srcHeight=0, srcWidth=0, srcChennal=3;
	unsigned short dstWidth, dstHeight, dstWriteWidth, dstChennal=3;
	unsigned short cropWidth, cropHeight;
	//srcWidthBites may be bigger than srcWidth*3,similay to dstWriteWidth may be bigger than dstWidth
};

#endif  // STRUCT_H

/**
 * Basic Structures.
 * 2020/11/20
 */
#ifndef STRUCTS_H
#define STRUCTS_H

#include <iostream>
#include <vector>
#include <array>
#include <string>

typedef std::vector<std::vector<std::array<float, 6>>> BatchBox;
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

#endif  // STRUCTS_H

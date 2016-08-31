/* Copyright 2014 Google Inc.

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
*/

#ifndef BIRDWRAP_H
#define BIRDWRAP_H


#include <stdexcept>
#include <cassert>
#include <algorithm>
#include <list>
#include <iostream>
#include <queue>
#include <string>

extern "C" {
	typedef struct Point
	{
		int x;
		int y;
	} Point ;
	void processScreenShot(unsigned char* input,
	                       int* output,
	                       int width, int height);

	void preprocessDataForNN(int* input,
	                         float* output,
	                         int width, int height);


	int findSlingshotCenter(int* scene,
	                        int width,
	                        int height);

	int getCurrScore(int const* input, int* output, int width, int height);
	int getEndScore(unsigned char const* input, int width, int height);
}


// struct Point
// {
// 	int x;
// 	int y;
// };


class Rectangle
{
public:
	int x;
	int y;
	int w;
	int h;

	Rectangle();
	Rectangle(int a, int b, int c, int d);
	void setBounds(int a, int b, int c, int d);
	void add(Point p);
	void add(Rectangle r);
	void setSize(int a, int b);
	int getSize();
	bool intersects(Rectangle r);
};


extern int* components;
extern int numComponents;
extern Rectangle* boxes;
extern int* compColors;
extern int nWidth;
extern int nHeight;


int* histogram(Rectangle r, int width, int height);

Point findSlingshotCenterImpl(int* scene,
                              int width,
                              int height);

void findConnectedComponents(int* image,
                             int width, int height);
void findBoundingBoxes(int* image, int width, int height,
                       int numComp);

int getScoreInGame(int const* screenshot, int* output, int width, int height);
int countToDigitInGame(int count);
int getScoreEndGame(unsigned char const* screenshot, int width, int height);
int countToDigitEndGame(int count);

#endif  // BIRDWRAP_H

// Local Variables:
// mode: c++
// End:

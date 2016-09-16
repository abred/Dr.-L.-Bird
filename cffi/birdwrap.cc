#include "birdwrap.h"


Rectangle::Rectangle() : x(0), y(0), w(0), h(0) {};
Rectangle::Rectangle(int a, int b, int c, int d) :
  x(a), y(b), w(c), h(d)
{};

void Rectangle::setBounds(int a, int b, int c, int d)
{
  x = a;
  y = b;
  w = c;
  h = d;
};

void Rectangle::add(Point p)
{
  if (p.x < x) {
    w = w + (x - p.x);
    x = p.x;
  }
  else if (p.x > x + w) {
    w = w + (p.x - (x+w));
  }

  if (p.y < y) {
    h = h + (y - p.y);
    y = p.y;
  }
  else if (p.y > y + h) {
    h = h + (p.y - (y+h));
  }
};

void Rectangle::add(Rectangle r)
{
  add({r.x, r.y});
  add({r.x+w, r.y+h});
}

void Rectangle::setSize(int a, int b)
{
  w = a;
  h = b;
};

int Rectangle::getSize()
{
  return w * h;
}

bool Rectangle::intersects(Rectangle r)
{
  return !(x   > r.x + r.w ||
           r.x > x   + w   ||
           y   > r.y + r.h ||
           r.y > y   + h);
};


int* histogram(Rectangle r, int width, int height) {
  int* h = new int[512] {0};

  for (int y = r.y; y < r.y + r.h; y++) {
    if ((y < 0) || (y >= height))
      continue;
    for (int x = r.x; x < r.x + r.w; x++) {
      if ((x < 0) || (x >= width))
        continue;
      h[compColors[components[y * width + x]]] += 1;
    }
  }

  return h;
}


void processScreenShotImpl(unsigned char* input,
                        int* output,
                        int width, int height) {
  // extract width and height
  if ((height != 480) && (width != 840)) {
    std::cerr << "Expecting 840x480 image" << std::endl;
  }

  // quantize to 3-bit color
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      unsigned char r = input[3 * (y * width + x) + 0];
      unsigned char g = input[3 * (y * width + x) + 1];
      unsigned char b = input[3 * (y * width + x) + 2];
      int ri = (int)(r & 0x000000e0) <<  1;
      int gi = (int)(g & 0x000000e0) >>  2;
      int bi = (int)(b & 0x000000e0) >>  5;
      output[y * width + x] = ri | gi | bi;
    }
  }

  findConnectedComponents(output, width, height);
}


void preprocessDataForNNImpl(int* input,
                             float* output,
                             int width, int height) {
  // extract width and height
  if ((height != 480) && (width != 840)) {
    std::cerr << "Expecting 840x480 image" << std::endl;
  }

  // remove 32 zero lines
  // remove 8 zero columns
  // -> for dropout layers (half size)
  // (output already zeroed)
  for (int y = 0; y < height-32; y++) {
    for (int x = 0; x < width-8; x++) {
      // output[y * (width-8) + x] = (float(input[(y+32) * width + (x+4)]) - 512.0f) / 256.0f;
      output[y * (width-8) + x] = float(input[(y+32) * width + (x+4)]) / 512.0f;

    }
  }

  for (int y = 0; y < 448; y++) {
    for (int x = 0; x < 832; x++) {
      if(output[y * 832 + x] > 1.0 || output[y * 832 + x] < 0.0)
        std::cerr << "pixel should be between 0 and 1: " << output[y * 832 + x] << std::endl;
    }
  }


}





extern "C" {
void processScreenShot(unsigned char* input,
                       int* output,
                       int width, int height) {
  processScreenShotImpl(input, output, width, height);
}

void preprocessDataForNN(int* input,
                         float* output,
                         int width, int height) {
  preprocessDataForNNImpl(input, output, width, height);
}

int findSlingshotCenter(int* scene,
                        int width,
                        int height) {
  Point p = findSlingshotCenterImpl(scene, width, height);
  return p.y * width + p.x;
}

int getCurrScore(int const* input, int* output, int width, int height) {
  return getScoreInGame(input, output, width, height);
}

int getEndScore(unsigned char const* input, int width, int height) {
    return getScoreEndGame(input, width, height);
  }

}

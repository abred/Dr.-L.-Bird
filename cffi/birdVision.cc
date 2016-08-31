#include "birdwrap.h"

#include <stdio.h>
#include <vector>

int* components = nullptr;
int numComponents = 0;
Rectangle* boxes = nullptr;
int* compColors = nullptr;
int nWidth = 840;
int nHeight = 480;

#include <tesseract/baseapi.h>


Point findSlingshotCenterImpl(int* scene,
                              int width,
                              int height) {
  Rectangle obj;

  bool ignorePixel[height][width];

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      ignorePixel[i][j] = false;
    }
  }

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if ((scene[i*width + j] != 345) || ignorePixel[i][j])
        continue;
      obj.x = j; obj.y = i; obj.w = 0; obj.h = 0;
      std::list<Point> l;

      l.push_back({j, i});
      ignorePixel[i][j] = true;
      while (true) {
        if (l.empty())
          break;
        Point p = l.back();
        l.pop_back();
        // check if the colours of the adjacent points of p is
        // belong to slingshot

        //check underneath pixel
        if (p.y < height - 1)
          if ((   scene[(p.y + 1) * width + p.x] == 345
               || scene[(p.y + 1) * width + p.x] == 418
               || scene[(p.y + 1) * width + p.x] == 273
               || scene[(p.y + 1) * width + p.x] == 281
               || scene[(p.y + 1) * width + p.x] == 209
               || scene[(p.y + 1) * width + p.x] == 346
               || scene[(p.y + 1) * width + p.x] == 354
               || scene[(p.y + 1) * width + p.x] == 282
               || scene[(p.y + 1) * width + p.x] == 351)
              && !ignorePixel[p.y + 1][p.x]) {
            l.push_back({p.x, p.y + 1});
            obj.add({p.x, p.y + 1});
          }

        //check right pixel
        if (p.x < width - 1)
          if ((   scene[p.y * width + p.x + 1] == 345
               || scene[p.y * width + p.x + 1] == 418
               || scene[p.y * width + p.x + 1] == 346
               || scene[p.y * width + p.x + 1] == 354
               || scene[p.y * width + p.x + 1] == 273
               || scene[p.y * width + p.x + 1] == 281
               || scene[p.y * width + p.x + 1] == 209
               || scene[p.y * width + p.x + 1] == 282
               || scene[p.y * width + p.x + 1] == 351)
              && !ignorePixel[p.y][p.x + 1]) {
            l.push_back({p.x + 1, p.y});
            obj.add({p.x + 1, p.y});
          }

        //check upper pixel
        if (p.y > 0)
          if ((   scene[(p.y - 1) * width + p.x] == 345
               || scene[(p.y - 1) * width + p.x] == 418
               || scene[(p.y - 1) * width + p.x] == 346
               || scene[(p.y - 1) * width + p.x] == 354
               || scene[(p.y - 1) * width + p.x] == 273
               || scene[(p.y - 1) * width + p.x] == 281
               || scene[(p.y - 1) * width + p.x] == 209
               || scene[(p.y - 1) * width + p.x] == 282
               || scene[(p.y - 1) * width + p.x] == 351)
              && !ignorePixel[p.y - 1][p.x]) {
            l.push_back({p.x, p.y - 1});
            obj.add({p.x, p.y - 1});
          }

        //check left pixel
        if (p.x > 0)
          if ((   scene[p.y * width + p.x - 1] == 345
               || scene[p.y * width + p.x - 1] == 418
               || scene[p.y * width + p.x - 1] == 346
               || scene[p.y * width + p.x - 1] == 354
               || scene[p.y * width + p.x - 1] == 273
               || scene[p.y * width + p.x - 1] == 281
               || scene[p.y * width + p.x - 1] == 209
               || scene[p.y * width + p.x - 1] == 282
               || scene[p.y * width + p.x - 1] == 351)
              && !ignorePixel[p.y][p.x - 1]) {
            l.push_back({p.x - 1, p.y});
            obj.add({p.x - 1, p.y});
          }

        //ignore checked pixels
        if (p.y < height - 1)
          ignorePixel[p.y + 1][p.x] = true;
        if (p.x < width - 1)
          ignorePixel[p.y][p.x + 1] = true;
        if (p.y > 0)
          ignorePixel[p.y - 1][p.x] = true;
        if (p.x > 0)
          ignorePixel[p.y][p.x - 1] = true;

      }
      int* hist = histogram(obj, width, height);

      // abandon shelf underneath
      if (obj.h > 10) {
        Rectangle col(obj.x, obj.y, 1, obj.h);
        int* histCol = histogram(col, width, height);


        if (   scene[obj.y * width + obj.x] == 511
            || scene[obj.y * width + obj.x] == 447) {
          for (int m = obj.y; m < obj.y + obj.h; m++) {
            if (   scene[m * width + obj.x] == 345
                || scene[m * width + obj.x] == 418
                || scene[m * width + obj.x] == 346
                || scene[m * width + obj.x] == 354
                || scene[m * width + obj.x] == 273
                || scene[m * width + obj.x] == 281
                || scene[m * width + obj.x] == 209
                || scene[m * width + obj.x] == 282
                || scene[m * width + obj.x] == 351) {
              obj.setSize(obj.w, m - obj.y);
              break;
            }
          }
        }

        while (histCol[511] >= obj.h * 0.8) {
          obj.setBounds(obj.x + 1, obj.y, obj.w - 1,
                        obj.h);
          col.setBounds(obj.x + 1, obj.y, 1, obj.h);
          delete[] histCol;
          histCol = histogram(col, width, height);
        }

        col.setBounds(obj.x + obj.w, obj.y, 1, obj.h);
        delete[] histCol;
        histCol = histogram(col, width, height);
        while (histCol[511] >= obj.h * 0.8 && obj.h > 10) {
          obj.setSize(obj.w - 1, obj.h);
          col.setBounds(obj.x + obj.w, obj.y, 1, obj.h);
          delete[] histCol;
          histCol = histogram(col, width, height);
        }
      }

      if (obj.w > obj.h)
        continue;

      if ((hist[345] > std::max(32.0, 0.1 * obj.w * obj.h))
          && (hist[64] != 0)) {
        obj.add(Rectangle(obj.x - obj.w / 10,
                          obj.y - obj.h / 3,
                          obj.w / 10 * 12,
                          obj.h / 3 * 4));
        double X_OFFSET = 0.5;
        double Y_OFFSET = 0.65;
        Point p = {(int)(obj.x + X_OFFSET * obj.w),
                   (int)(obj.y + Y_OFFSET * obj.w)};

        for(int h = obj.y; h < obj.y+obj.h; h++)
        {
          scene[h*width + obj.x] = 100;
          scene[h*width + obj.x+obj.w] = 100;
        }

        for(int w = obj.x; w < obj.x+obj.w; w++)
        {
          scene[obj.y*width + w] = 100;
          scene[(obj.y+obj.h)*width + w] = 100;
        }

        delete[] hist;
        // int c =
        //     scene[p.y*width + p.x]
        //   + scene[p.y*width + p.x + 1]
        //   + scene[p.y*width + p.x - 1]
        //   + scene[(p.y+1)*width + p.x]
        //   + scene[(p.y+1)*width + p.x + 1]
        //   + scene[(p.y+1)*width + p.x - 1]
        //   + scene[(p.y-1)*width + p.x]
        //   + scene[(p.y-1)*width + p.x + 1]
        //   + scene[(p.y-1)*width + p.x - 1];
        //   printf("color slc: %d", c);
        return p;
      }
    }
  }
  return {0, 0};
}



void findConnectedComponents(int* image,
                             int width, int height)
{
  int n = 0;
  if (components != nullptr) {
    delete[] components;
  }
  components = new int[width * height];

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      components[y * width + x] = -1;
    }
  }

  // iterate over all pixels
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      // skip negative pixels
      if (image[y * width + x] == -1)
        continue;

      // check if component was already numbered
      if (components[y * width + x] != -1)
        continue;

      // number the new component
      std::queue<Point> q;
      q.push({x, y});

      components[y * width + x] = n;
      while (!q.empty()) {
        Point p = q.front();
        q.pop();
        if ((p.y > 0)
            && (image[(p.y - 1) * width + p.x] == image[p.y * width + p.x])
            && (components[(p.y - 1) * width + p.x] == -1)) {
          q.push({p.x, p.y - 1});
          components[(p.y - 1) * width + p.x] = n;
        }
        if ((p.x > 0)
            && (image[p.y * width + p.x - 1] == image[p.y * width + p.x])
            && (components[p.y * width + p.x - 1] == -1)) {
          q.push({p.x - 1, p.y});
          components[p.y * width + p.x - 1] = n;
        }
        if ((p.y < height - 1)
            && (image[(p.y + 1) * width + p.x] == image[p.y * width + p.x])
            && (components[(p.y + 1) * width + p.x] == -1)) {
          q.push({p.x, p.y + 1});
          components[(p.y + 1) * width + p.x] = n;
        }
        if ((p.x < width - 1)
            && (image[p.y * width + p.x + 1] == image[p.y * width + p.x])
            && (components[p.y * width + p.x + 1] == -1)) {
          q.push({p.x + 1, p.y});
          components[p.y * width + p.x + 1] = n;
        }
      }
      n = n + 1;
    }
  }
  numComponents = n;
  // printf("num comp %d\n",n);
  fflush(stdout);

  if (n > 10) {
    if (compColors != nullptr) {
      delete[] compColors;
    }
    compColors = new int[numComponents];
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        compColors[components[y * width + x]] = image[y * width + x];
      }
    }
  }
  findBoundingBoxes(components, width, height, numComponents);
}

void findBoundingBoxes(int* image, int width, int height, int numComp) {
  if (boxes != nullptr) {
    delete[] boxes;
  }
  boxes = new Rectangle[numComp];
  for (int c = 0; c < numComp; c++) {
    boxes[c] = Rectangle();
  }
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int n = image[y * width + x];
      if (n < 0)
        continue;
      if (boxes[n].getSize() == 0) {
        boxes[n] = Rectangle(x, y, 1, 1);
      } else {
        boxes[n].add({x, y});
      }
    }
  }
}


int getScoreInGame(int const* screenshot, int* output, int width, int height) {
  // crop score image
  int subImgH = 32;
  int subImgW = 200;
  int startX = 632;
  int startY = 21;

  // extract characters
  for (int y = 0; y < subImgH; y++) {
    for (int x = 0; x < subImgW; x++) {
      int color = screenshot[(y + startY) * width + (x + startX)];
      output[y * subImgW + x] = color == 511 ? 511 : -1;
    }
  }

  findConnectedComponents(output, subImgW, subImgH);

  std::vector<int> boxesXs;
  for (int i = 0; i < numComponents; i++) {
    boxesXs.push_back(boxes[i].x);
  }
  std::sort(boxesXs.begin(), boxesXs.end());

  int score = 0;
  for (int i = 0; i < numComponents; i++) {
    for (int j = 0; j < numComponents; j++) {
      if (boxes[j].x != boxesXs[i]) {
        continue;
      }
      if (boxes[j].w < 2)
        continue;
      int n = 0;

      for (int y = boxes[j].y; y < boxes[j].y+boxes[j].h; y++) {
        for (int x = boxes[j].x; x < boxes[j].x+boxes[j].w; x++) {
          if (output[y*subImgW+x] > 0)
            ++n;
        }
      }
      // printf("count: %d\n", n);

      score = score * 10 + countToDigitInGame(n);
    }
  }

  return score;
}

int countToDigitInGame(int count)
{
  int digit = 0;
  switch (count) {
  case 117: digit = 0; break;
  case  53: digit = 1; break;
  case  82: digit = 2; break;
  case  77: digit = 3; break;
  case  79: digit = 4; break;
  case  66: digit = 5; break;
  case  94: digit = 6; break;
  case  64: digit = 7; break;
  case 107: digit = 8; break;
  case 108: digit = 9; break;
  default:
    printf("INVALID DIGIT\n");
    exit(-1);
    break;
  }
  return digit;
}


int getScoreEndGame(unsigned char const* screenshot, int width, int height) {
  // crop score image
  int subImgH = 32;
  int subImgW = 100;
  int startX = 370;
  int startY = 265;

  unsigned char* output = new unsigned char[subImgH * subImgW];
  // extract characters
  for (int y = 0; y < subImgH; y++) {
    for (int x = 0; x < subImgW; x++) {
      int r = screenshot[3*((y + startY) * width + (x + startX)) + 0];
      int g = screenshot[3*((y + startY) * width + (x + startX)) + 1];
      int b = screenshot[3*((y + startY) * width + (x + startX)) + 2];
      output[y * subImgW + x] = (r+g+b)/3;
    }
  }

  tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();
  // Initialize tesseract-ocr with English, without specifying tessdata path
  if (api->Init(NULL, "eng")) {
    fprintf(stderr, "Could not initialize tesseract.\n");
    exit(1);
  }
  api->SetVariable("tessedit_char_whitelist", "0123456789");

  char* text = api->TesseractRect(output, 1, 100, 0, 0, 100, 32);
  std::cout << "recognized text: " << text << std::endl;

  delete[] output;

  int score = -1;
  try {
    score = std::stoi(text);
  } catch (...) {

  }

  return score;
}

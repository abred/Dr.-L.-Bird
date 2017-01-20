default:
	g++ -c birdVision.cc -fPIC -o birdVision.o -g -Wall -O3 -std=c++11 -I/sw/taurus/libraries/tesseract/3.04/include
	g++ -c birdwrap.cc -fPIC -o birdwrap.o -g -Wall  -O3 -std=c++11 -I/sw/taurus/libraries/tesseract/3.04/include
	g++ -shared -Wl,-soname,libbirdwrap.so -o libbirdwrap.so  birdwrap.o birdVision.o -g -Wall -O3 -std=c++11 -L/sw/taurus/libraries/tesseract/3.04/lib64  -ltesseract -Wl,-rpath,/sw/taurus/libraries/tesseract/3.04/lib64

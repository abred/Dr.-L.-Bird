default:
	g++ -c birdwrap.cc -fPIC -o birdwrap.o -g -Wall  -std=c++11 -I/sw/taurus/libraries/tesseract/3.04/include -O3
	g++ -shared -Wl,-soname,libbirdwrap.so -o libbirdwrap.so  birdwrap.o  -g -Wall -g -O3 -std=c++11 -L/sw/taurus/libraries/tesseract/3.04/lib64  -ltesseract -Wl,-rpath,/sw/taurus/libraries/tesseract/3.04/lib64

default:
	g++ -c birdVision.cc -fPIC -o birdVision.o -g -Wall -O3 -std=c++11
	g++ -c birdwrap.cc -fPIC -o birdwrap.o -g -Wall  -O3 -std=c++11
	g++ -shared -Wl,-soname,libbirdwrap.so -o libbirdwrap.so  birdwrap.o birdVision.o -g -Wall -O3 -std=c++11  -ltesseract

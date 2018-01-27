CC 	= g++
CFLAGS	= -I./include -O2
LDFLAGS = -lraspicam

raspicam_test: src/raspicam_test.cpp
	@mkdir -p ./bin
	$(CC) $(CFLAGS) $< -o ./bin/$@ $(LDFLAGS)

raspicam_cv_test: src/raspicam_test.cpp
	@mkdir -p ./bin
	$(CC) $(CFLAGS) $< -o ./bin/$@ $(LDFLAGS) `pkg-config --libs opencv`

.PHONY: clean

clean:
	rm -rf ./bin/*

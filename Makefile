.PHONY: build clean configure

PYTHON ?= python3

build: configure
	cmake --build build -j

configure:
	cmake -S . -B build -DPython_EXECUTABLE=$(shell which $(PYTHON))

clean:
	rm -rf build
	rm -f studious/*.so
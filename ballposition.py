#!/usr/bin/python3

import sharemem
import struct

format = "HH"
mem = None
size = 0;

def init(count=1):
	global mem
	global size
	size = count
	if size:
		mem = sharemem.open(3145914, count*struct.calcsize(format))

def unpack(format, offset=0):
	if not mem:
		raise RuntimeError("BallPosition module not initialised, you must call ballposition.init(size)")
	return struct.unpack(format, sharemem.read(mem+offset, struct.calcsize(format)))

def pack(format, *args, offset=0):
	if not mem:
		raise RuntimeError("BallPosition module not initialised, you must call ballposition.init(size)")
	sharemem.write(mem+offset, struct.pack(format, *args))

def read(offset=0):
	if offset >= size:
		raise RuntimeError("ballposition.read(offset={}) call error. Offset si bigger then initialized memory ({}). Maybe not properky opened?".format(offset, size))
	return unpack(format, offset=offset*struct.calcsize(format))

def write(pos, offset=0):
	if offset >= size:
		raise RuntimeError("ballposition.read(offset={}) call error. Offset si bigger then initialized memory ({}). Maybe not properky opened?".format(offset, size))
	pack(format, *pos, offset=offset*struct.calcsize(format))


if __name__ == '__main__':
	import click
	import time

	@click.command()
	@click.option('--count', '-c', default=1, help="Count of measured samles")
	@click.option('--pause', '-p', default=0.1, help="Pause between samples")
	@click.option('--number', '-n', default=1, help="Number of data to read")
	def main(count, pause, number):
		init(number)
		for _ in range(count):
			print([read(x) for x in range(number)])
			time.sleep(pause)
	main()
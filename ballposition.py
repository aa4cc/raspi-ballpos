#!/usr/bin/python3

import sharemem
import struct

format = "HH"
mem = None

def init(count=1):
	global mem
	mem = sharemem.open(3145914, count*struct.calcsize(format))

def unpack(format, offset=0):
	return struct.unpack(format, sharemem.read(mem+offset, struct.calcsize(format)))

def pack(format, *args, offset=0):
	sharemem.write(mem+offset, struct.pack(format, *args))

def read(offset=0):
	return unpack(format, offset=offset*struct.calcsize(format))

def write(pos, offset=0):
	pack(format, *pos, offset=offset*struct.calcsize(format))


if __name__ == '__main__':
	import click
	import time

	@click.command()
	@click.option('--count', '-c', default=1, help="Count of measured samles")
	@click.option('--pause', '-p', default=0.1, help="Pause between samples")
	def main(count, pause):
		for _ in range(count):
			print("{} {}".format(*read()))
			time.sleep(pause)
	main()
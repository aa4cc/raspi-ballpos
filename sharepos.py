#!/usr/bin/python3

from sharemem import SharedMemory
from struct import pack, unpack, calcsize

KEY = 3145914
DEFAULT_FORMAT = "ff"

class SharedPosition(SharedMemory):
	"""docstring for SharedPosition"""
	def __init__(self, count=1, create=True, format=None):
		self.count=count;
		if self.count:
			if not format:
				format=DEFAULT_FORMAT
			self._format=format;
			self._itemsize=calcsize(format)
			SharedMemory.__init__(self, key=KEY, size=count*self._itemsize, create=create)

	def read(self, offset=0, lock=True):
		if not self.count:
			raise RuntimeError("Reading from zero size SharedPosition")
		return unpack(self._format, SharedMemory.read(self, length=self._itemsize, offset=offset*self._itemsize, lock=lock))

	def write(self, data, offset=0, lock=True):
		if not self.count:
			raise RuntimeError("Writing to zero size SharedPosition")
		if data is not None:
			SharedMemory.write(self, data=pack(self._format, *data), offset=offset*self._itemsize, lock=lock)

	def read_many(self, count=0, offset=0):
		if not count:
			count=self.count;

		self.lock()
		try:
			pos = [self.read(x+offset, lock=False) for x in range(count)];
		finally:
			self.unlock()
		return pos

	def write_many(self, data, offset=0):
		self.lock()
		try:
			[self.write(x,offset=i+offset, lock=False) for i, x in enumerate(data)];
		finally:
			self.unlock()

if __name__ == '__main__':
	import click
	import time

	@click.command()
	@click.option('--count', '-c', default=1, help="Count of measured values")
	@click.option('--pause', '-p', default=0.0, help="Pause between samples")
	@click.option('--create', '-m', default=False, is_flag=True, help="Create shared memory")
	@click.option('--format', '-f', default=None, type=str, help="Format of items in shared memory. See struct")
	def main(count, pause, create, format):
		sp = SharedPosition(count, create=create, format=format)
		try:
			while pause:
				print(sp.read_many())
				time.sleep(pause)
			print(sp.read_many())
		except KeyboardInterrupt:
			pass
	main()
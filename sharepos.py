#!/usr/bin/python3

from sharemem import SharedMemory
from struct import pack, unpack, calcsize

KEY = 3145914
DEFAULT_FORMAT = "fff"

class SharedPosition(SharedMemory):
	"""docstring for SharedPosition"""
	def __init__(self, count=1, create=True, format=None, key=None):
		if format is None:
			format = DEFAULT_FORMAT
		if key is None:
			key = KEY

		self.count=count;
		
		self._format=format;
		self._itemsize=calcsize(format)

		SharedMemory.__init__(self, key=key, size=count*self._itemsize, create=create)

	def __getitem__(self, n):
		return self.read(n)

	def __setitem__(self, n, item):
		self.write(item, n)

	def __repr__(self):
		return "sharepos.SharedPosition(count={}, create={}, format={}, key={})".format(
			self.count,
			self.created,
			self._format,
			self.key)

	def __len__(self):
		return self.count

	def read(self, offset=0, lock=True):
		return unpack(self._format, SharedMemory.read(self, length=self._itemsize, offset=offset*self._itemsize, lock=lock))

	def write(self, data, offset=0, lock=True):
		if data is not None:
			SharedMemory.write(self, data=pack(self._format, *data), offset=offset*self._itemsize, lock=lock)

	def read_many(self, count=None, offset=0):
		if count is None:
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
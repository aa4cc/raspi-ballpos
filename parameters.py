import json
from pprint import pformat, pprint

class Parameters():
    # this class  loads the json configuration file, as well as kwargs
    def __init__(self, file=None):
        self.data = {}
        self.file = file
        self.runtime = {}
        if self.file is not None:
            self.load()

    def load(self, file=None, clear=False):
        if file is None:
            file = self.file

        with open(file, "r") as f:
            j = json.load(f)
            if clear:
                self.data.clear()
            self.data.update(j)

    def update(self, d):
        self.data.update(d)

    def save(self, file=None):
        if file is None:
            file = self.file

        if file is not None:
            with open(file, "w") as f:
                json.dump(self.data, f, indent="\t")

    @property
    def keys(self):
        return self.data.keys

    def __getattr__(self, name):
        return self.data[name]

    def __getitem__(self, key):
        # print(f"key={key}")
        if key=="neural-network":
            return False
        return self.data[key]

    def __setitem__(self, key, val):
        self.data[key] = val

    def __delitem__(self, key):
        del self.data[key]

    def __repr__(self):
         return pformat(self.data)

    def __contains__(self, key):
        return key in self.data

if __name__ == '__main__':
    p = Parameters("../config.json")
    print(p)

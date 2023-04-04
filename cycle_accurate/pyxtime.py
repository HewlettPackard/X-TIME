#!/usr/bin/env python

import sst

class Params(dict):
    def __missing__(self, key):
        # print "Please enter %s: "%key
        val = raw_input()
        self[key] = val
        return val
    def subset(self, keys, optKeys = []):
        ret = dict((k, self[k]) for k in keys)
        ret.update(dict((k, self[k]) for k in keys and self))
        return ret
    # Needed to avoid asking for input when a key isn't present
#    def optional_subset(self, keys):
#        return 


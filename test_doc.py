#!/usr/bin/python
from doctest import DocFileSuite
import unittest

def load_tests(loader, standard_tests, pattern):
    return DocFileSuite("ieee754.txt")

if __name__ == '__main__':
    unittest.main()


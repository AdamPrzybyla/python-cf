#!/usr/bin/python
from test.test_support import run_unittest
from doctest import DocFileSuite
import unittest
import cf as math

def load_tests(loader, standard_tests, pattern):
    return DocFileSuite("ieee754.txt")

if __name__ == '__main__':
    unittest.main()


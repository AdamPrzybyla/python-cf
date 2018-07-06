#! /usr/bin/python
# -*- coding: utf-8 -*-
# Metaprogramming Driven Development

import unittest,os,struct
import cf as math
eps = 1E-05
math_testcases = 'math_testcases.txt'
test_file = 'cmath_testcases.txt'

def ulps_check(expected, got, ulps=20):
	"""Given non-NaN floats `expected` and `got`,
	check that they're equal to within the given number of ulps.

	Returns None on success and an error message on failure."""

	ulps_error = to_ulps(got) - to_ulps(expected)
	if abs(ulps_error) <= ulps:
		return None
	return "error = {} ulps; permitted error = {} ulps".format(ulps_error,ulps)

def to_ulps(x):
	"""Convert a non-NaN float x to an integer, in such a way that
	adjacent floats are converted to adjacent integers.  Then
	abs(ulps(x) - ulps(y)) gives the difference in ulps between two
	floats.

	The results from this function will only make sense on platforms
	where C doubles are represented in IEEE 754 binary64 format.

	"""
	n = struct.unpack('<q', struct.pack('<d', x))[0]
	if n < 0:
		n = ~(n+2**63)
	return n

class gen_math_test(type):
	def __init__(cls, name, bases, nmspc):
		super(gen_math_test, cls).__init__(name, bases, nmspc)
		tem=list(cls.parse_testfile(test_file))
		cls.uses_metaclass = lambda self : True
		cls.test_sem2 = lambda self : self.assertTrue(True)
		t="setattr_test"
		setattr(cls,"test_%s" % t.replace('.',"_"),lambda self: self.checker(t))
		cls.te=dict([[e[0],e[1:]] for e in tem])
		for f in [e for e in cls.te.keys()]: setattr(cls,"test_%s" % f,(lambda g: lambda self: self.checker(g)) (f))

	def parse_testfile(self,fname):
		"""Parse a file with test values

		Empty lines or lines starting with -- are ignored
		yields id, fn, arg_real, arg_imag, exp_real, exp_imag
		"""
		with open(fname) as fp:
			for line in fp:
				# skip comment lines and blank lines
				if line.startswith('--') or not line.strip(): continue
				lhs, rhs = line.split('->')
				id, fn, arg_real, arg_imag = lhs.split()
				rhs_pieces = rhs.split()
				exp_real, exp_imag = rhs_pieces[0], rhs_pieces[1]
				flags = rhs_pieces[2:]
				yield (id, fn,
					   float(arg_real), float(arg_imag),
					   float(exp_real), float(exp_imag),
					   flags
					  )

class test_math_sem(unittest.TestCase):
	__metaclass__ = gen_math_test

	def checker(self,n):
		idn=n
		if n=='setattr_test':
			self.assertTrue(True)
		else:
			fn, ar, ai, er, ei, flags=self.te[n]
			# Skip if either the input or result is complex, or if
			# flags is nonempty
			if ai != 0. or ei != 0. or flags:
				return
			if fn in ['rect', 'polar']:
				# no real versions of rect, polar
				return
			func = getattr(math, fn)
			try:
				result = func(ar)
			except ValueError:
				message = ("Unexpected ValueError in test %s:%s(%r)\n" % (idn, fn, ar))
				self.fail(message)
			except OverflowError:
				message = ("Unexpected OverflowError in test %s:%s(%r)\n" % (idn, fn, ar))
				self.fail(message)
			self.ftest("%s:%s(%r)" % (idn, fn, ar), result, er)

	def ftest(self, name, value, expected):
		if abs(value-expected) > eps:
			# Use %r instead of %f so the error message
			# displays full precision. Otherwise discrepancies
			# in the last few bits will lead to very confusing
			# error messages
			self.fail('%s returned %r, expected %r' % (name, value, expected))

	def test_sem(self):
		self.assertTrue(True)

	def test_meta(self):
		self.assertTrue(self.uses_metaclass())

class gen_math_mtest(type):
	def __init__(cls, name, bases, nmspc):
		super(gen_math_mtest, cls).__init__(name, bases, nmspc)
		tem=list(cls.parse_mtestfile(math_testcases))
		cls.uses_metaclass = lambda self : True
		cls.test_sem2 = lambda self : self.assertTrue(True)
		t="setattr_test"
		setattr(cls,"test_%s" % t.replace('.',"_"),lambda self: self.checker(t))
		cls.te=dict([[e[0],e[1:]] for e in tem])
		for f in [e for e in cls.te.keys()]: setattr(cls,"test_%s" % f,(lambda g: lambda self: self.checker(g)) (f))

	def parse_mtestfile(self,fname):
		"""Parse a file with test values

		-- starts a comment
		blank lines, or lines containing only a comment, are ignored
		other lines are expected to have the form
		  id fn arg -> expected [flag]*

		"""
		with open(fname) as fp:
			for line in fp:
				# strip comments, and skip blank lines
				if '--' in line: line = line[:line.index('--')]
				if not line.strip(): continue
				lhs, rhs = line.split('->')
				idn, fn, arg = lhs.split()
				rhs_pieces = rhs.split()
				exp = rhs_pieces[0]
				flags = rhs_pieces[1:]
				yield (idn, fn, float(arg), float(exp), flags)

class test_mmath_sem(unittest.TestCase):
	__metaclass__ = gen_math_mtest

	def checker(self,n):
		fail_fmt = "{}:{}({!r}): expected {!r}, got {!r}"
		failures = []
		idn=n
		if n=='setattr_test':
			self.assertTrue(True)
		else:
			fn, arg, expected, flags=self.te[n]
			if fn in ['lgamma','gamma','expm1']: return
			func = getattr(math, fn)

			if 'invalid' in flags or 'divide-by-zero' in flags:
				expected = 'ValueError'
			elif 'overflow' in flags:
				expected = 'OverflowError'

			try:
				got = func(arg)
			except ValueError:
				got = 'ValueError'
			except OverflowError:
				got = 'OverflowError'

			accuracy_failure = None
			if isinstance(got, float) and isinstance(expected, float):
				if math.isnan(expected) and math.isnan(got):
					return
				if not math.isnan(expected) and not math.isnan(got):
					if fn == 'lgamma':
						# we use a weaker accuracy test for lgamma;
						# lgamma only achieves an absolute error of
						# a few multiples of the machine accuracy, in
						# general.
						accuracy_failure = acc_check(expected, got,
												  rel_err = 5e-15,
												  abs_err = 5e-15)
					elif fn == 'erfc':
						# erfc has less-than-ideal accuracy for large
						# arguments (x ~ 25 or so), mainly due to the
						# error involved in computing exp(-x*x).
						#
						# XXX Would be better to weaken this test only
						# for large x, instead of for all x.
						accuracy_failure = ulps_check(expected, got, 2000)

					else:
						accuracy_failure = ulps_check(expected, got, 20)
					if accuracy_failure is None:
						return

			if isinstance(got, str) and isinstance(expected, str):
				if got == expected:
					return

			fail_msg = fail_fmt.format(id, fn, arg, expected, got)
			if accuracy_failure is not None:
				fail_msg += ' ({})'.format(accuracy_failure)
			failures.append(fail_msg)
		if failures:
			self.fail('Failures in test_mtestfile:\n ' + '\n  '.join(failures))

	def test_sem(self):
		self.assertTrue(True)

	def test_meta(self):
		self.assertTrue(self.uses_metaclass())

if __name__=="__main__":
	unittest.main()

#! /usr/bin/python
# -*- coding: utf-8 -*-
"""Gosper's continued fractions for Python, using lazy evaluation.

Author:
Marcin Ciura <Marcin.Ciura at polsl.pl>

Web references:
* Ralph William Gosper, Continued Fractions. In: HAKMEM,
  http://home.pipeline.com/~hbaker1/hakmem/cf.html
* Ralph William Gosper, Continued Fraction Arithmetic
  http://www.tweedledum.com/rwg/cfup.htm
* Mark Jason Dominus, Arithmetic with Continued Fractions,
  http://perl.plover.com/yak/cftalk/
* Jean Vuillemin, Exact Real Computer Arithmetic with Continued Fractions
  ftp://ftp.inria.fr/INRIA/publication/publi-pdf/RR/RR-0760.pdf
* Philippe Flajolet, Brigitte Vall\'ee, Ilan Vardi,
  Continued fractions from Euclid to the present day,
  http://citeseer.ist.psu.edu/flajolet00continued.html

Changed by Adam Przybyla <adam@ertel.com.pl> 07.09.2011
- added Python 2.6 math compatibility functions: 
- ldexp(x,y) frexp(x) fsum(x) isnan(x) isinf(x) atanh(x) copysign(x,y) log3p(x) trunc(x) acosh(x) asinh(x) factorial(x)
- fixed NaN error

Changed by Adam Przybyla <adam.przybyla@gmail.com> 05.07.2018
- added Python 2.7 math compatibility functions: 
- erf erfc
"""

# TODO:
# Check IEEE-754 about NaNs - when to signal them?
# Maybe use __slots__.
# Maybe make the transcendental functions (and __cmp__())
# work when we feed them with non-positive partial quotients
# or the last partial quotient equal one. If done, fix also
# digits(cf((...,),(0,)) hanging.
# Rather don't split NaN into infinity and NaN.
# Rather don't make cf(number,number) not lazy.

# If we're using Python 2.2, then enable generators and override
# int() with long(). No effect in Python version 2.3 and later.
from __future__ import generators
__version__ = "0.1"
max_iters = 100
# The global variable max_iters is used in two places
# to limit the number of iterations when the result of
# an operation probably has a finite continued fraction
# while the arguments probably have infinite continued
# fractions. Resolving such situations is undecidable
# in a finite number of iterations.
#   If the _cf_homographic() function consumes in total
# max_iters partial quotients from both its arguments,
# with unchanging, differing by one lower and upper bounds
# for the next partial quotient to be output, or with an
# unchanging infinite upper bound, then it outputs the upper
# bound, followed by None (which stands for an infinite partial
# quotient, i.e. the end of a continued fracion). This
# behaviour also implicitly influences log() [with log10()]
# and atan() [with asin(), acos(), and atan2()].
#   If the __cmp__() method consumes max_iters/2 equal partial
# quotients of each argument, then it decides that the two
# numbers are equal.
#   The _cf_homographic() function needs 81 iterations to decide
# that (math.sqrt(5) - 1)/2 - (cf.sqrt(5) - 1)/2 > 0; the
# __cmp__() method needs 39 == 78/2 iterations to find out that
# these two numbers are unequal.
#   This is the worst case: each partial quotient of phi
# contributes log10((sqrt(5) + 1)/2) =~= 0.21 decimal digits
# of accuracy. However, by the Lochs' theorem [1], in almost
# every (with respect to Lebesgue measure) continued fraction
# each partial quotient contributes on the average
# pi**2/(6*ln(2)*ln(10)) =~= 1.03 decimal digits of accuracy.
# Empirical studies [2] show that the variance of decimal
# digits per partial quotient is about 0.62 (if anyone can
# point me to the relation between this number and the Hensley
# constant =~= 0.5160624 [3, 4], please do so). After 100
# iterations with unchanging lower and upper bounds, the number
# of correct digits after the decimal point in the complete
# quotient (i.e. the value of the not-yet-emitted tail) of
# the result has an approximately normal distribution with
# mean 1.03*100 and variance 0.62*100, so the accuracy of
# the complete quotient has an approximately log-normal
# distribution with mean 10**(-1.03*100 + 0.62*100*ln(10)/2)
# =~= 2.07**-100 == roughly somewhere between 1e-32 and 1e-31.
# The algorithm outputs the partial quotients (k, infinity),
# while the exact result might consist of (k, l1,...) or
# (k-1, 1, l2,...), where l1 >= 1e+31, l2 >= 1e+31-1. By the
# Gauss-Kuzmin theorem [5, 6], a partial quotient in the
# continued fraction expansion of almost every number exceeds
# 1e+31 with probability log2(1+1e-31) =~= 1e-31/ln(2) =~=
# 1.5e-31. Therefore the probability that the algorithm
# erroneously cuts the continued fraction expansion of a random
# number doesn't exceed 1.5e-31 times the number of partial
# quotients computed.
#   Note that there is no limit for the value of the initial
# partial quotient (the floor of the computed value) and
# partial quotients computed in less than max_iters steps
# (resulting from largish partial quotients in the arguments).
#   Idealistically inclined folks, who would prefer their
# computations to hang rather than to yield a heuristic result,
# can achieve this effect by setting accuracy to a negative
# value.
# References:
# [1] G. Lochs, Vergleich der Genauigkeit von Dezimalbruch und
#     Kettenbruch, Abhandlungen aus dem Mathematischen Seminar
#     der Universit\"at Hamburg 27 (1964), 142-144
# [2] Wieb Bosma, Karma Dajani, Cor Kraaikamp, Entropy and
#     counting correct digits, Report no. 9925, Department
#     of Mathematics, University of Nijmegen, 1999,
#     webdoc.ubn.kun.nl/mono/b/bosma_w/entrancoc.ps
# [3] Doug Hensley, The number of steps in the Euclidean algorithm,
#     Journal of Number Theory, 49 (1994), 142-182,
#     http://www.math.tamu.edu/~dhensley/euclidstep.tex
# [4] Lo{\"\i}ck Lhote, Computation of a Class of Continued
#     Fraction Constants, in: Proceedings of the Sixth Workshop on
#     Algorithm Engineering and Experiments and the First Workshop
#     on Analytic Algorithmics and Combinatorics, New Orleans,
#     2004, 199-210
# [5] Carl Friedrich Gauss, Werke, vol. 10^1, 552-556
# [6] Rodion O. Kuzmin, Sur un probl\`eme de Gauss, Atti del
#     Congresso Internazionale dei Matematici 6, Bologna, 1928,
#     83-89

exp_tan_max_pq = 10**31
# The exp() and tan() functions don't try to find a better
# approximation of the result, if the last best approximation
# and the current best approximation give the values k-1 and k
# (in any order) for the next partial quotient, and the complete
# quotient of the current best approximation (the value of its
# not-yet-emitted tail) differs from k less than 1/exp_tan_max_pq.
# In this situation, they decide that the result is a rational
# number, and output k followed by infinity. If the true result
# is not a rational number, then it would contain partial
# quotients (k, l1,...) or (k-1, 1, l2,...), where l1 >= 1e+32,
# l2 >= 1e+32-1. By the Gauss-Kuzmin theorem, the probability
# of a partial quotient greater or equal 1e+31 at a particular
# place in the continued fraction expansion of almost every number
# is log2(1+1e-31) =~= 1e-31/ln(2) =~= 1.5e-31. Therefore the
# probability of an erroneous cutting the continued fraction
# expansion of a random number is less than 1.5e-31 times the
# number of partial quotients generated.
#   This behaviour also implicitly influences raising numbers
# to a fractional power and computing sin() and cos().
#   Note that the accuracy of exp() and tan() in the case when
# the result is an irrational number is not affected, and that
# not all big partial quotients are cut. In particular, the
# initial partial quotient (the floor of the result) can be
# arbitrarily large.
#   As with max_iters, setting exp_tan_max_pq to a negative value
# disables the heuristics, making the computation hang when
# the result is indeed a rational number.

# The number of digits output by str() after the decimal point.
decimal_digits = 28

# If the number is less than 10**-scientific_notation_threshold,
# then scientific notation will be used in converting it to a string.
# Setting the threshold to a negative value disables the scientific
# notation altogether.
scientific_notation_threshold = 4

# The number of partial quotients output by repr().
repr_pqs = 17

def set_cf_parameter(name, value):
    """Sets the global variable with a given name to the
    given value. Useful if you do 'from cf import *'."""

    globals()[name] = value

class cf_base(object):
    """The abstract base class for continued fractions.
    Defines the methods required to simulate numbers.

    The only protocol that instances of derived classes
    must obey is that self.pq(n) should return the nth
    partial quotient of the continued fraction expansion
    of self when called with subsequent n's from zero
    upwards and return None to signal an infinite partial
    quotient, i.e.  the end of the continued fraction.
    After self.pq(n) returns None, self.pq(n+1) will
    never be called. All the partial quotients except the
    initial one must be positive; the last finite partial
    quotient must be greater than one.

    You must not use __init__() in subclasses of cf_base if you
    want your code to run in Python 2.2. Move the functionality
    to __new__() instead. This is due to the ubiquity of
    x = cf(x) in the code of this module and a peculiar behaviour
    of Python 2.2's __new__(). See
    http://sourceforge.net/tracker/index.php?func=detail&aid=537450&group_id=5470&atid=105470
    and the comment in cf.__new__() below."""

    def pq(self, n):
        """Returns the nth partial quotient of self.

        Caches the partial quotients in those derived classes
        that generate them with a stateful generator.
        The self.next_pq field must be set to the generator's
        next() method; the self.cache field must be set to an
        initially empty list."""

        self_cache = self.cache
        if n < len(self_cache):
            return self_cache[n]
        t = self.next_pq()
        self_cache.append(t)
        if t is None:
            # Allow the gc'ing of whatever contributed to self.
            del self.next_pq
        return t

    def __str__(self):
        """Return a string representation of self: 'NaN',
        '-?[0-9]+\.[0-9]*' or '-?[1-9]\.[0-9]*e-[1-9][0-9]*'.
        Does not append excess zeroes to the fractional part."""

        get_digit = digits(self).next
        try:
            integer_part = get_digit()
            if integer_part < 0:
                return '-' + str(-self)
            else:
                # Don't use self.pq(0) as the integer part,
                # because e.g. cf((1, 0, -2)) == -1.
                # The digits() function handles non-initial
                # non-positive partial-quotients properly.
                digit_list = [str(integer_part), '.']
        except StopIteration:
            return 'NaN'
        only_zeroes = (integer_part == 0)
        initial_zeroes = 0
        try:
            for i in xrange(decimal_digits):
                digit = get_digit()
                if only_zeroes:
                    if digit == 0:
                        initial_zeroes += 1
                    else:
                        only_zeroes = 0
                digit_list.append(str(digit))
        except StopIteration:
            pass
        if initial_zeroes//scientific_notation_threshold < 1:
            return ''.join(digit_list)

        # Get more digits, until accumulate decimal_digits
        # of them after the initial zeroes.
        try:
            while i < initial_zeroes + decimal_digits:
                digit = get_digit()
                if only_zeroes:
                    if digit == 0:
                        initial_zeroes += 1
                    else:
                        only_zeroes = 0
                digit_list.append(str(digit))
                i += 1
        except StopIteration:
            pass
        return ''.join([digit_list[initial_zeroes + 2], '.'] +
            digit_list[initial_zeroes + 3:] +
            ['e-', str(initial_zeroes + 1)])

    def __repr__(self):
        """Return a printable representation of self's
        partial quotients:
        'cf(NaN)' for a NaN,
        'cf([0-9]+)' for integer continued fractions,
        'cf(-?[0-9];([0-9]+,){0,repr_pqs-1}[0-9]+(,..)?'
        for continued fractions with a fractional part.
        The result ends with ',..' if self has more partial
        quotients than the repr_pqs shown."""

        pq_list = []
        for i in xrange(repr_pqs):
            pq = self.pq(i)
            if pq is None:
                break
            pq_list.append(str(pq))
        if not pq_list:
            return 'cf(NaN)'
        elif len(pq_list) == 1:
            return 'cf(%s)' % (pq_list[0])
        elif pq is not None:
            pq_list.append('..')
        return 'cf(%s;%s)' % (pq_list[0], ','.join(pq_list[1:]))

    def __cmp__(self, other):
        """Compare self to other. Returns a negative number
        when self < other, a positive number when self > other,
        0 when self == other and they are rational numbers with
        less than accuracy/2 partial quotients, and 0L when the
        first accuracy/2 partial quotients of self and other are
        equal."""

        # This method is more lazy than would be checking the
        # sign of (self - other), but doesn't return zero when
        # comparing a continued fraction with non-initial
        # non-positive partial quotients or a final partial
        # quotient equal to 1 against another fraction that
        # has the same value but differs in particular partial
        # quotients.
        self_pq = self.pq
        if not self_pq(0) is None and isinstance(other, (int, long)):
            # The casts to long are necessary, because
            # (int).__cmp__(long) and (long).__cmp__(int)
            # don't work.
            cmp = long(self_pq(0)).__cmp__(long(other))
            if cmp:
                return cmp
            elif self_pq(1) is None:
                return 0
            else:
                return long(self_pq(1)).__cmp__(0L)
        other = cf(other)
        other_pq = other.pq
        if other_pq(0) is None and self_pq(0) is None:
            return True
        for i in xrange(max_iters/2):
            self_pq_i = self_pq(i)
            other_pq_i = other_pq(i)
            if self_pq_i is None:
                if other_pq_i is None:
                    return 0
                else:
                    # 1-2*(i&1) == (-1)**i
                    return (1-2*(i&1))*long(other_pq_i).__cmp__(0L)
            else:
                if other_pq_i is None:
                    return (1-2*(i&1))*(0L).__cmp__(long(self_pq_i))
                else:
                    cmp = long(self_pq_i).__cmp__(long(other_pq_i))
                    if cmp:
                        return (1-2*(i&1))*cmp
        # Whoever cares about this, can check whether self and other
        # are strictly or approximately equal by checking whether the
        # type of the returned zero is int or long.
        return 0L

    def __nonzero__(self):
        """Return True if self == 0."""

        self_pq = self.pq
        self_floor = self_pq(0)
        if self_floor is None:
            raise ValueError, 'NaN detected'
        return (self_floor != 0) or (self_pq(1) is not None)

    def __add__(self, other):
        """Add other to self."""

        if isinstance(other, (int,long)):
            if other:
                return unop(self, 1, other, 0, 1)
            else:
                return self
        return binop(self, cf(other), 0, 1, +1, 0, 0, 0, 0, 1)

    def __sub__(self, other):
        """Subtract other from self."""

        if isinstance(other, (int,long)):
            if other:
                return unop(self, -1, other, 0, -1)
            else:
                return self
        return binop(self, cf(other), 0, 1, -1, 0, 0, 0, 0, 1)

    def __mul__(self, other):
        """Multiply self by other."""

        if isinstance(other, (int,long)):
            # Without this condition NaN*0 would be 0.
            if self.pq(0) is None:
                return NaN
            elif other == 1:
                return self
            else:
                return unop(self, other, 0, 0, 1)
        return binop(self, cf(other), 1, 0, 0, 0, 0, 0, 0, 1)

    def __div__(self, other):
        """Divide self by other."""

        if isinstance(other, (int,long)):
            if other == 1:
                return self
            else:
                return unop(self, 1, 0, 0, other)
        other = cf(other)
        other_pq = other.pq
        divisor = other_pq(0)
        # Without the condition (divisor is None) the result
        # would be 0 for any self != NaN. Without the condition
        # (divisor == 0) and (other_pq(1) is None) the result
        # would be 0 for self == 0, and this is unacceptable,
        # since NaN can stand for 0, too. TODO: can it really?
        if (divisor is None) or ((divisor == 0) and (other_pq(1) is None)):
            return NaN
        return binop(self, other, 0, 1, 0, 0, 0, 0, 1, 0)

    __truediv__ = __div__

    def __floordiv__(self, other):
        """Return self divided by other, rounded down to an integer."""

        # TODO: do we need to return a cf instead of int?
        return (self/other).pq(0)

    def __mod__(self, other):
        """Return the remainer of self divided by other,
        with the same sign as other."""

        return self - (self//other)*other

    def __divmod__(self, other):
        """Return (self//other, self%other)."""

        quotient = self//other
        return (quotient, self - quotient*other)

    def __pow__(self, other, z=None):
        """Return self to the power other, modulo z."""

        if z is not None:
            return (self**other)%z
        elif self.pq(0) is None:
            # TODO: should NaN**0 return NaN or 1?
            return NaN
        elif isinstance(other, (int,long)):
            # Fast track for integer other.
            return _cf_ipow(self, other)
        other = cf(other)
        exponent = other.pq(0)
        if exponent is None:
            # TODO: should 0**NaN return NaN or 0?
            return NaN
        elif other.pq(1) is None:
            # Fast track for integer other.
            return _cf_ipow(self, exponent)
        elif self.pq(0) < 0:
            raise (ValueError,
                'negative number cannot be raised to a fractional power')
        else:
            return exp(other*log(self))

    def __radd__(self, other):
        """Add self to other."""

        return self + other

    def __rsub__(self, other):
        """Subtract self from other."""

        if isinstance(other, (int,long)):
            return unop(self, -1, other, 0, 1)
        else:
            return cf(other) - self

    def __rmul__(self, other):
        """Multiply other by self."""

        return self*other

    def __rdiv__(self, other):
        """Divide other by self."""

        if isinstance(other, (int,long)):
            # Without this condition other/NaN would be 0.
            # TODO: do we really need it?
            if self.pq(0) is None:
                return NaN
            else:
                return unop(self, 0, other, 1, 0)
        else:
            return cf(other)/self

    __rtruediv__ = __rdiv__

    def __rfloordiv__(self, other):
        """Return other/self, rounded down to an integer."""

        # TODO: do we need to return a cf instead of int?
        return (other/self).pq(0)

    def __rmod__(self, other):
        """Return the remainer of other divided by self,
        with the same sign as self."""

        return other - (other//self)*self

    def __rdivmod__(self, other):
        """Return (other//self, other%self)."""

        quotient = other//self
        return (quotient, other - quotient*self)

    def __rpow__(self, other):
        """Raise other to the power self."""

        return cf(other)**self

    def __neg__(self):
        """Return self negated."""

        return unop(self, -1, 0, 0, 1)

    def __pos__(self):
        """Return self positive."""

        return self

    def __abs__(self):
        """Return the absolute value of self."""

        if (self.pq(0) is None) or (self.pq(0) >= 0):
            return self
        else:
            return -self

    def __int__(self):
        """Return an integer equal to self rounded towards zero.

        Silently emits long integers for large numbers, which is
        consistent with the behaviour of int() in Python 2.3 and
        later, but in Python 2.2 may cause an OverflowError in
        subsequent calculations."""

        self_floor = self.pq(0)
        if self_floor is None:
            raise ValueError, 'NaN detected'
        if (self_floor >= 0) or (self.pq(1) is None):
            return self_floor
        else:
            return self_floor + 1

    def __long__(x):
        """Return a long integer equal to self rounded towards zero."""

        # We mustn't use long(int(x)), because it would cause
        # infinite recusion in Python 2.2, where we override int()
        # with long().
        return long(x.__int__())

    def __float__(self):
        """Convert self to a floating point number."""

        self_pq = self.pq
        if self_pq(0) is None:
            # TODO: find the NaN strings used by various
            # C libraries; do a cascaded try...except on them.
            return float('NaN')
        n = 1
        last_num = 1
        last_den = 0
        curr_num = self_pq(0)
        curr_den = 1
        curr_convergent = float(curr_num)
        pq = self_pq(1)
        while pq is not None:
            last_num, curr_num, last_den, curr_den = (
                curr_num, pq*curr_num + last_num,
                curr_den, pq*curr_den + last_den)
            last_convergent = curr_convergent
            curr_convergent = float(curr_num)/float(curr_den)
            if curr_convergent == last_convergent:
                break
            n += 1
            pq = self_pq(n)
        return curr_convergent

    def __complex__(self):
        """Convert self to a complex number."""

        return float(self) + 0j

class cf(cf_base):
    """Class for continued fractions constructed from numbers,
    quotients of numbers, or canned partial quotients."""

    def __new__(cls, x, y=None):
        """Construct a continued fraction object.

        Allowed use cases:
        cf(any continued fraction object) returns the object;
        cf(number) constructs a lazy continued fraction representing
            the number - the laziness is apparent when the number
            is not integer;
        cf(number, number) constructs a lazy continued fraction
            representing the ratio of given numbers;
        cf(sequence) constructs a continued fraction with initial
            partial quotients taken from the sequence, followed
            by None;
        cf(sequence, sequence) constructs a continued fraction
            with initial partial quotients taken from the first
            sequence, followed by a cyclic repetition of the second
            sequence."""
        #for  isinstance(x, float) and str(x)=='inf':
	#		returns x

        if isinstance(x, float) and isnan(x):
			return cf(())

        if isinstance(x, cf_base):
            # Be idempotent.
            if y is not None:
                raise TypeError, 'cf(cf, anything) is invalid'
            # Python 2.2 will try to call x.__init__(x) after
            # __new__() returns, which will cause all kinds of
            # funny errors if x.__init__() is defined.
            return x

        self = object.__new__(cls)
        if hasattr(x, '__getitem__'):
            # x is sequence-like; if y is not None,
            # it'd better be a sequence, too.
            def fixed_pqs_closure(n):
                """First return the partial quotients from x,
                then from y, then from y, then from y,..."""
                n -= len(x)
                if n < 0:
                    return x[n]
                else:
                    return y[n % len(y)]
            if y is None:
                y = (None,)
            self.pq = fixed_pqs_closure
        else:
            # x is presumably a number; if y is not None,
            # it'd better be a number, too.
            def ratio(x, y):
                """Lazily generate subsequent partial quotients
                of a rational number. Works also fine for ratios,
                where at least one part is a floating point number."""
                while y:
                    x_div_y, x_mod_y = divmod(x, y)
                    # int() is crucial, because the result of
                    # division involving at least one float is
                    # a float, albeit without fractional part.
                    yield int(x_div_y)
                    x, y = y, x_mod_y
                yield None
            if y is None:
                y = 1
            self.cache = []
            if not isinstance(x,(float,int)):
                x=float(x)
            self.next_pq = ratio(x, y).next
        return self

# Not a Number, including also infinities.
NaN = cf(())

# Static sources for the continued fractions of basic constants,
# lest they should be re-created every time they're needed.
zero = cf(0)
one = cf(1)

class binop(cf_base):
    """Class for bihomographic binary operations."""

    def __new__(cls, x, y, a, b, c, d, e, f, g, h):
        """Return (a*x*y + b*x + c*y + d)/(e*x*y + f*x + g*y + h)."""

        self = object.__new__(cls)
        self.cache = []
        self.next_pq = _cf_bihomographic(
            x.pq, y.pq, a, b, c, d, e, f, g, h).next
        return self

class unop(cf_base):
    """Class for homographic unary operations."""

    def __new__(cls, x, a, b, c, d):
        """Return (a*x + b)/(c*x + d)."""

        self = object.__new__(cls)
        self.cache = []
        self.next_pq = _cf_homographic(0, x.pq, a, b, c, d).next
        return self

def _cf_bihomographic(x_pq, y_pq, a, b, c, d, e, f, g, h):
    """Generate subsequent partial quotients of the
    continued fraction
    z(x,y) = (a*x*y + b*x + c*y + d)/(e*x*y + f*x + g*y + h),
    given x.pq, y.pq and the parameters a--h."""

    # This function is the workhorse of the module,
    # so it is extensively optimized at the cost of
    # readability, but the algorithm is simple:
    # while True:
    #   Calculate the values of z(x,y) at the points
    #   (inf,inf), (inf,0), (0,inf), (0,0).
    #   If all four values have the same integral part,
    #   then emit it as the next partial quotient of z(x,y)
    #   and set a--h to new values; else ingest another
    #   partial quotient from x_pq or y_pq, depending on
    #   the sign of (abs(b/f-d/h) - abs(c/g-d/h)), and
    #   set a--h to new values.

    # Cache accuracy in a local variable for faster lookup.
    iters_left = allowed_iters = max_iters

    # nx and ny count partial quotients requested from x_pq and y_pq.
    nx = ny = 0
    while 1:
        ingest_x = None

        # a/e, b/f, c/g, d/h are the values of z at the points
        # (inf,inf), (inf,0), (0,inf), (0,0). lower and upper are the
        # bounds for the next output partial quotient:
        # floor(min(a/e,b/f,c/g,d/h)), floor(max(a/e,b/f,c/g,d/h)),
        # not counting the 0/0's. lower or upper == None means that
        # the corresponding bound is infinite.
        # For speed, don't use the builtin functions min() and max()
        # and explicitly test the denominators against zero instead
        # of using try...except.
        if e:
            lower = upper = a//e
            any_results = 1
        else:
            lower = upper = None
            # any_results == 0 iff a/c == 0/0.
            any_results = a

        if f:
            bf = b//f
            if not any_results:
                lower = upper = bf
            elif (lower is None) or (bf < lower):
                lower = bf
            else:
                # Here we know that lower is not None,
                # bf >= lower, and lower == upper,
                # so bf >= upper.
                upper = bf
        elif b:
            upper = bf = None
            any_results = 1
        else:
            ingest_x = 0

        if g:
            cg = c//g
            if not any_results:
                lower = upper = cg
            elif (lower is None) or (cg < lower):
                lower = cg
            elif (upper is not None) and (cg > upper):
                upper = cg
        elif c:
            upper = cg = None
            any_results = 1
        else:
            ingest_x = 1

        if h:
            dh = d//h
            if not any_results:
                lower = upper = dh
            elif (lower is None) or (dh < lower):
                lower = dh
            elif (upper is not None) and (dh > upper):
                upper = dh
        elif d:
            upper = dh = None
        else:
            # A dummy value of dh for calculating bf-dh and cg-dh.
            dh = 0

        if lower == upper:
            # If lower == upper, then we can output it as the next
            # partial quotient. This includes also the case when
            # lower == upper == None.
            yield upper
            a,b,c,d,e,f,g,h = (e,f,g,h,
                a-e*upper,b-f*upper,c-g*upper,d-h*upper)
            iters_left = allowed_iters
            continue
        elif (upper is None) or (lower == upper - 1):
            # We also output a partial quotient when we were unable
            # to decide whether lower or upper should be output,
            # despite us having consumed a heck of a lot of input
            # partial quotients. It happens when both operands are
            # irrational (have infinite continued fractions) and the
            # result is probably rational (has a finite continued
            # fraction). Either the non-rounded values of lower and
            # upper are closer to an integer number with each
            # iteration or upper stays at infinity and lower grows
            # with each iteration.
            if not iters_left:
                yield upper
                # We might give the generator a chance to emit
                # a finite next partial quotient instead of None,
                # continuing the loop and subsequently emitting
                # None only when (lower < 0) and (upper > 0) for
                # more than allowed_iters, but such cases would be
                # exceptionally rare, so we prefer to crudely end
                # the continued fraction.
                yield None
            else:
                iters_left -= 1

        # If both inputs haven't ended yet, then determine which one
        # to poll. The convoluted conditions below are equivalent to
        # ingest_x = (abs(bf-dh) > abs(cg-dh)). We don't use abs()
        # for maximal speed.
        if ingest_x is None:
            if bf is None:
                ingest_x = (dh is not None)
            elif cg is None:
                ingest_x = (dh is None)
            elif dh is None:
                if bf > 0:
                    if cg > 0:
                        ingest_x = (bf < cg)
                    else:
                        ingest_x = (bf < -cg)
                else:
                    if cg > 0:
                        ingest_x = (-bf < cg)
                    else:
                        ingest_x = (bf > cg)
            elif bf > dh:
                if cg > dh:
                    ingest_x = (bf > cg)
                else:
                    ingest_x = (bf-dh > dh-cg)
            else:
                if cg > dh:
                    ingest_x = (dh-bf > cg-dh)
                else:
                    ingest_x = (cg > bf)

        # Reuse bf instead of introducing another variable.
        if ingest_x:
            bf = x_pq(nx)
            nx += 1
            if bf is not None:
                a,b,c,d,e,f,g,h = c+a*bf,d+b*bf,a,b,g+e*bf,h+f*bf,e,f
            else:
                for bf in _cf_homographic(ny, y_pq, a, b, e, f):
                    yield bf
        else:
            bf = y_pq(ny)
            ny += 1
            if bf is not None:
                a,b,c,d,e,f,g,h = b+a*bf,a,d+c*bf,c,f+e*bf,e,h+g*bf,g
            else:
                for bf in _cf_homographic(nx, x_pq, a, c, e, g):
                    yield bf

def _cf_homographic(nx, x_pq, a, b, c, d):
    """Generate subsequent partial quotients of the continued
    fraction z(x) = (a*x + b)/(c*x + d), given the number of
    x's partial quotients consumed so far, x.pq, and the
    parameters a--d."""

    while 1:
        # ac, bd == floor(z(infinity)), floor(z(0)).
        # ac or bd == None means that the corresponding value
        # is infinite. Using divmod to precompute a - c*ac == a%c
        # and b - d*bd == b%d doesn't pay off.
        if c:
            ac = a//c
            if d:
                bd = b//d
            elif b:
                bd = None
            else:
                bd = ac
        elif d:
            if a:
                # Any ac != bd will do.
                ac = None
                bd = 0
            else:
                ac = bd = b//d
        else:
            ac = bd = None

        if ac == bd:
            yield ac
            a, b, c, d = c, d, a-c*ac, b-d*bd
        else:
            # Reuse ac instead of introducing another variable.
            ac = x_pq(nx)
            nx += 1
            if ac is not None:
                a, b, c, d = b+a*ac, a, d+c*ac, c
            else:
                while c:
                    ac, bd = divmod(a, c)
                    yield ac
                    a, c = c, bd
                yield None

def digits(x, base=10):
    """Generate subsequent digits of x in a given base.
    Raises StopIteration when all the subsequent digits
    would be 0. The first result is actually the floor of x;
    subsequent results are digits of the fractional part of x.
    For negative x's you should call digits(-x) and prepend
    '-' to the accumulated result."""

    a, b, c, d, output_digits, nx, x_pq = 1, 0, 0, 1, 0, 0, x.pq
    while a or b:
        if c:
            ac = a//c
            if d:
                bd = b//d
            elif b:
                bd = None
            else:
                bd = ac
        elif d:
            if a:
                # Any ac != bd will do.
                ac = None
                bd = 0
            else:
                ac = bd = b//d
        else:
            return

        if ac == bd:
            yield ac
            a, b = base*(a - c*ac), base*(b - d*bd)
            output_digits = 1
        else:
            # Reuse ac instead of introducing another variable.
            ac = x_pq(nx)
            nx += 1
            if ac is not None:
                a,b,c,d = b+a*ac,a,d+c*ac,c
            else:
                b,d = a,c
    if not output_digits:
        yield 0

def floor(x):
    """Round x down to an integer."""

    if str(x)=='nan':
	return float('nan')
    # TODO: do we need to return a cf instead of int?
    return float(cf(x).pq(0))

def ceil(x):
    """Round x up to an integer."""

    # TODO: maybe return an int instead of cf?
    if isinstance(x, float) and str(x)=='inf':
	return x

    if isinstance(x, float) and str(x)=='-inf':
	return x

    x = cf(x)
    x_floor = x.pq(0)
    if x_floor is None:
        return x
    elif x.pq(1) is None:
        return cf(x_floor)
    else:
        return cf(x_floor + 1)

def fabs(x):
    """Return the absolute value of x."""

    return abs(cf(x))

def fmod(x, y):
    """Return the remainder of x divided by y,
    with the same sign as x."""
    if y==0 or str(x)=='inf' or str(x)=='-inf':
        raise ValueError

    if x*y > 0:
        return x%y
    else:
        return -((-x)%y)

def modf(x):
    """Return the fractional and integer parts of x.
    Both results carry the sign of x."""

    # TODO: maybe return an int instead of cf
    # as the integer part?
    if isinstance(x, float) and str(x)=='inf':
	return (0.0,x)
    if isinstance(x, float) and str(x)=='-inf':
	return (-0.0,x)
    if isinstance(x, float) and str(x)=='nan':
	return (x,x)
    integer = int(x)
    return (cf(x)-integer, cf(integer))

def _cf_ipow(x, n):
    """Raise x to an integer power n."""

    # Right-to-left binary exponentiation algorithm is faster
    # than left-to-right algorithm for an argument which is
    # an irrational 2**k'th root of a rational number.

    if n < 0:
        negative_exponent = 1
        n = -n
    else:
        negative_exponent = 0
    result = one
    while n:
        if n&1:
            if negative_exponent and (n == 1):
                if result is one:
                    return 1/x
                else:
                    # Save one level of unop() by inverting the
                    # result in the last iteration instead of
                    # returning 1/_cf_ipow(x, -n).
                    # returns 1/(result*x)
                    return binop(result, x, 0, 0, 0, 1, 1, 0, 0, 0)
            else:
                if result is one:
                    result = x
                else:
                    result *= x
            n -= 1
        else:
            if negative_exponent and (n == 2) and (result is one):
                # Save one level of unop().
                # returns 1/(x**2)
                return binop(x, x, 0, 0, 0, 1, 1, 0, 0, 0)
            else:
                x *= x
                n >>= 1
    return result

def pow(x, y):
    """Raise x to the yth power."""

    if y==0:
        return 1.0
    if isinstance(x, float) and str(x)=='inf':
	if y<0:
		return 0.0
	else:
		if str(y)=='nan':
			return y
		else:
	        	return x
    if isinstance(x, float) and str(x)=='-inf':
	if y<0:
		return 0.0
	else:
		if str(y)=='nan':
			return y
		else:
                        if y%2==1:
	        	    return x
			else:
			    return -x

    if x==-1 and isinstance(y, float):
        if str(y)=='inf':
             return 1.0
        if str(y)=='-inf':
             return 1.0
        if y%2==0:
             return 1.0
        if y%2==1.0:
             return -1.0

    if isinstance(y, float) and str(y)=='nan':
	if x==1:
            return 1.0
        else:
            return float('nan')
    if isinstance(y, float) and str(y)=='-inf' and x==1:
        return 1.0
    if isinstance(y, float) and str(y)=='inf' and x==1:
        return 1.0
    if isinstance(y, float) and str(y)=='-inf':# and x!=0:
	if x==0:
            raise ValueError,"math domain error"
        if abs(x)<1:
            return -y
        else:
            return 0.0

    if isinstance(y, float) and str(y)=='inf':# and x!=0:
        if abs(x)<=1:
            return 0.0
        else:
            return y
    if x==0 and y>0 and not str(y)=='nan':
	return 0
    if x==0 and y<0 and not str(y)=='nan':
	raise ValueError
    return cf(x)**y

def ldexp(x,y):
    """'ldexp(x, i) -> x * (2**i)'"""

    if isinstance(x, float) and str(x)=='inf':
        return x
    if isinstance(x, float) and str(x)=='-inf':
        return x
    if x==0:
	return x
    return cf(x)*(cf(2)**y)

def frexp(x):
    """Return the mantissa and exponent of x, as pair (m, e).
    m is a float and e is an int, such that x = m * 2.**e.
    If x is 0, m and e are both 0.  Else 0.5 <= abs(m) < 1.0."""
    if isinstance(x, float) and str(x)=='inf':
        return [x,x]
    if isinstance(x, float) and str(x)=='-inf':
        return [x,x]
    if x==0 or x==0.0:
        return [0.0,0]
    else:
	z=-1.0
	if x>0:
            z=1.0
        x=abs(x)
	p=ceil(log(cf(x))/log_of_2)
	m=x/(2**p)
	if m==1:
	    return [z*0.5, p+1]
	else:
            return  [ z*m,p] 
		
def fsum(x):
    """Return an accurate sum of values in the iterable."""   
    if len(x)==1 and x[0]==[]:
	return 0.0
    if not x:
	return 0.0
    return reduce(lambda k1,k2: cf(k1)+k2,x)

def isnan(x):
    y=x
    return not y==x

def isinf(x):
    if str(x)=="inf" or str(x)=="-inf":
        return True

    y=x
    if str(cf(x))=="nan" and x==y:
        return True
    else:
        return False

def atanh(x):
    #1/2*log(1+z/1-z)
    if x==1 or x==-1:
	raise ValueError
    if isnan(x):
        return x
    return log(unop(cf(x),1,1,-1,1))/2

def copysign(x,y):
    """Return x with the sign of y."""
    if y==0 and str(y)[0]=='-':
	y=-1

    if (y<0)^(x<0):
        return -x
    else:
        return x

def log1p(x):
    """Return the natural logarithm of 1+x (base e).
          The result is computed in a way which is accurate for x near zero."""
    return log(x+1)

def trunc(x):
    """Truncates x to the nearest Integral toward 0. Uses the __trunc__ magic method."""
    try:
        return x.__trunc__()
    except:
        raise AttributeError

def acosh(x):
    """Return the hyperbolic arc cosine (measured in radians) of x."""
    if isinstance(x, float) and str(x)=='inf':
        return x
    if isinstance(x, float) and str(x)=='-inf':
	raise ValueError
    if isinstance(x, float) and str(x)=='nan':
        return float('nan')
    return log(x+sqrt((cf(x)**2)-1))

def asinh(x):
    """Return the hyperbolic arc sine (measured in radians) of x."""
    if isinstance(x, float) and str(x)=='nan':
        return float('nan')
    if isinstance(x, float) and str(x)=='-inf':
        return x
    if isinstance(x, float) and str(x)=='inf':
        return x
    return log(x+sqrt((cf(x)**2)+1))

def factorial(x):
    if x<0 or int(x)!=x:
	raise ValueError
    if x==1 or x==0:
        return 1
    else:
        import operator
        return reduce(operator.mul,(k for k in range(1,1+int(x))))

def _cf_isqrt(x):
    """Calculate an integer approximation of square root of x."""

    if x < 0:
        raise (ValueError,
            'the square root of a negative number cannot be computed')
    r1 = 1
    r2 = x
    while not (-1 <= (r1-r2) <= 1):
        r1 = (r1 + r2)//2
        r2 = x//r1
    return r1

class sqrt(cf_base):
    """Lazily calculate the square root using Newton's method,
    which doubles its accuracy with each iteration."""

    def __new__(cls, x):
        """Initialize the lazy calculation: set self.plain to the
        best approximation of sqrt(x) that can be achieved with
        integer calculations; set self.converse to x/self.plain."""

        if x == 0:
            # Newton's method doesn't work for x == 0.
            return zero
        if isinstance(x, float) and str(x)=='inf':
	    return x
        if isinstance(x, float) and str(x)=='nan':
	    return x
        self = object.__new__(cls)
        if isinstance(x, (int, long)):
            # Precompute self.plain as the integer approximation
            # to sqrt(x).
            integer_root = _cf_isqrt(x)
            self.plain = cf(integer_root)
            self.converse = cf(x, integer_root)
        else:
            x = cf(x)
            if (x.pq(0) == 0) and (x.pq(1) > 1):
                # For 0 < x <= 1/2 precompute self.plain as the
                # inverse of the integer approximation of sqrt(1/x).
                integer_root = _cf_isqrt(x.pq(1))
                self.plain = cf(1,integer_root)
                self.converse = x*integer_root
            else:
                # For x > 1/2 precompute self.plain as the integer
                # approximation to sqrt(x).
                integer_root = _cf_isqrt(x.pq(0))
                self.plain = cf(integer_root)
                self.converse = x/integer_root
        self.x = x
        self.cache = []
        return self

    def pq(self, n):
        """Return the nth partial quotient of sqrt(self.x)."""

        # I'd love to plug the result back to itself, but alas
        # it wouldn't work if the continued fraction of the
        # square root ends in 1, 1, 1,...
        if n < len(self.cache):
            return self.cache[n]
        while 1:
            # As Newton's method doubles the accuracy with each
            # iteration, it should also double the number of correct
            # partial quotients, so the loop should actually iterate
            # only twice.
            pq = self.plain.pq(n)
            if pq == self.converse.pq(n):
                # Memoize the partial quotient and return it.
                self.cache.append(pq)
                return pq

            # self.plain equal (self.plain + self.converse)/2
            self.plain = binop(self.plain, self.converse,
                0, 1, 1, 0, 0, 0, 0, 2)
            self.converse = self.x/self.plain

            # Compute the partial quotients number 0..n-1
            # of self.plain and self.converse, so that we
            # can examine their pq(n) in the next iteration
            # of the main loop.
            plain_pq = self.plain.pq
            converse_pq = self.converse.pq
            for i in xrange(n):
                plain_pq(i)
                converse_pq(i)

def hypot(x, y):
    """Return the Euclidean norm of (x, y)."""
    if str(x)=='inf' or str(x)=='-inf':
	return float('inf')
    if str(y)=='inf' or str(y)=='-inf':
	return float('inf')
    if str(y)=='nan' or str(x)=='nan':
	return float('nan')

    return sqrt(x*x + y*y)

class _cf_exp_1n(cf_base):
    """Return e**(1/n) == cf(1;n-1,1,1,3*n-1,1,1,5*n-1,1,1,...).
    Used in exp() and log()."""

    def __new__(cls, inverse_exponent):
        """Set self.pq to a closure that returns the nth partial
        quotient of e**(1/inverse_exponent) quickly, regardless of n."""

        self = object.__new__(cls)
        if inverse_exponent > 1:
            # This is the most frequent case; called repeatedly
            # from exp.pq().
            def exp_1n_pq_closure(index):
                if index%3 == 1:
                    return (2*(index//3)+1)*inverse_exponent - 1
                else:
                    return 1
            self.pq = exp_1n_pq_closure
            return self
        elif inverse_exponent == 1:
            # The above closure would also work for inverse_exponent == 1,
            # but would emit 0 as the partial quotient at index 1.
            # By special-casing e**(1/1) we not only avoid emitting the
            # zero partial quotient, but also get a faster e.pq(), since
            # there are no accesses to the external variable inverse_exponent.
            def e_pq(index):
                if index%3 == 2:
                    return 2*(index//3) + 2
                elif index:
                    return 1
                else:
                    return 2
            self.pq = e_pq
            return self
        elif inverse_exponent:
            # Don't choke on negative arguments.
            return 1/_cf_exp_1n(-inverse_exponent)
        else:
            # e**(1/0) == NaN.
            return NaN

# The base of the natural logarithm.
e = _cf_exp_1n(1)

def _cf_exp_2_to_nth(n, cache=[e]):
    """Return e to the power (2**n), caching the results.
    Used to speed up exp() and log()."""

    while len(cache) <= n:
        cache.append(cache[-1]*cache[-1])
    return cache[n]

def _cf_iexp(n):
    """Return e to the power n for an integer n,
    using a cached list of e**(2**n)."""

    # Essentially the same algorithm as in _cf_ipow(),
    # except that e**(-2**k) is returned as 1/e**(2**k)
    # instead of using binop and e**(2**(k-1)), so that
    # e**(2**k) can be cached.

    if n < 0:
        negative_exponent = 1
        n = -n
    else:
        negative_exponent = 0
    result = one
    k = 0
    while n:
        if n&1:
            if negative_exponent and (n == 1):
                if result is one:
                    return 1/_cf_exp_2_to_nth(k)
                else:
                    # Save one level of unop, returning
                    # 1/(result*_cf_exp_2_to_nth(k))
                    return binop(result, _cf_exp_2_to_nth(k), 0, 0, 0, 1, 1, 0, 0, 0)
            else:
                if result is one:
                    result = _cf_exp_2_to_nth(k)
                else:
                    result *= _cf_exp_2_to_nth(k)
        n >>= 1
        k += 1
    return result

class cf_base2(cf_base):
    def pq(self, n):
        """Return the common partial quotients of tan(the last
        partial sum) and tan(the penultimate partial sum),
        extending the sum if necessary. Decides that the result
        is a rational number when the two partial quotients
        differ by one and future partial quotients would exceed
        exp_tan_max_pq.

        After a term 1/q, the next term in the sum is at most
        1/(q*(q+1)), so the accuracy of the approximation at
        least doubles with each term."""

        if n < len(self.cache):
            return self.cache[n]
        # We need to compute another term.
        while 1:
            # Here lesser needn't be less than greater at all.
            lesser = self.worse.pq(n)
            greater = self.better.pq(n)
            if lesser == greater:
                # x always lies between the last two partial sums,
                # and tan(x) is monotonic for x in [0, pi/4], so
                # when the partial quotients of self.worse and
                # self.better coincide, the actual partial quotient
                # of e**x will be equalto them.
                self.cache.append(greater)
                return greater
            elif lesser is not None:
                if lesser > greater:
                    # Only here we assure that lesser < greater.
                    lesser, greater = greater, lesser
                if lesser == greater - 1:
                    # Compute the accuracy of the complete quotient:
                    # it differs from greater by at most 1/next_pq.
                    if self.better.pq(n) == greater:
                        # self.better is an overestimation.
                        next_pq = self.better.pq(n + 1)
                    elif self.better.pq(n + 1) == 1:
                        # self.better is an underestimation.
                        next_pq = self.better.pq(n + 2) + 1
                    else:
                        # self.better hasn't got enough accuracy yet.
                        next_pq = None
                    # next_pq may be None when we set it explicitly
                    # or when self.better(n + 1) is None, i.e.
                    # self.better itself is a rational number.
                    if ((next_pq is not None)
                    and (next_pq//exp_tan_max_pq >= 1)):
                        # If greater differs from the complete
                        # quotient by at most 1/exp_log_accuracy,
                        # then heuristically decide that the result
                        # is a rational number: emit greater and
                        # end the continued fraction.
                        self.cache.append(greater)
                        self.cache.append(None)
                        return greater
            # Compute self.x.pq(0), so that
            # we can compute self.x.pq(1).
            self.x.pq(0)
            assert self.x.pq(0) == 0
            # 1/denominator is the next term
            # of the alternating sum for x.
            denominator = self.x.pq(1)
            if denominator is None:
                # self.x == 0; x has a finite alternating
                # sum representation (it's a rational number).
                self.worse = self.better
                continue
            self.worse = self.better
            self.pq_own(n,denominator)

class exp(cf_base2):
    """Calculate e to the power x, lazily decomposing x into
    an alternating sum of fractions with alternatingly a bit
    too large (or exact) and a bit too small (or exact)
    denominators, known as the Ostrogradsky series of second
    kind."""

    # References for the Ostrogradsky series of second kind:
    # * Wac{\l}aw Sierpi\'nski, O kilku algorytmach dla rozwijania
    #   liczb rzeczywistych na szeregi, Sprawozdania z posiedze\'n
    #   Towarzystwa Naukowego Warszawskiego, Wydzia{\l} III 4 (1911),
    #   56-77; also: Sur quelques algorithmes pour d\'evelopper les
    #   nombres r\'eels en s\'eries, in: Oeuvres choisies, tome I,
    #   PWN, Warszawa, 1974, 236-254
    # * Evgeny Yakovlevych Remez, O zakonomernykh ryadakh,
    #   kotorye mogut byt' svyazany s dvumya algoritmami
    #   M. V. Ostrogradskogo dlya priblizheniya irracionalnykh
    #   chisel, Uspekhi matematicheskikh nauk 6 (1951), no. 5(45),
    #   33-42

    def __new__(cls, x):
        """Initialize the lazy calculation: set self.better
        to e**floor(x) and self.x to x - floor(x)."""

        if isinstance(x, (int,long)):
            # Fast track for integer exponents.
            return _cf_iexp(x)
        if isinstance(x, float) and str(x)=='inf':
            return x
        if isinstance(x, float) and str(x)=='-inf':
            return 0.0
        x = cf(x)
        exponent = x.pq(0)
        if exponent is None:
            return NaN
        self = object.__new__(cls)
        self.better = _cf_iexp(exponent)
        self.worse = NaN
        self.x = x - exponent
        self.muldiv = 1
        self.cache = []
        return self

    def pq_own(self,n,denominator):
        # When self.muldiv == 1, self.better *= exp(1/denominator).
        # When self.muldiv == 0, self.better /= exp(1/denominator).
        self.better = binop(self.better, _cf_exp_1n(denominator),
            self.muldiv, 1 - self.muldiv, 0, 0,
            0, 0, 1 - self.muldiv, self.muldiv)
        # If we've multiplied self.better, let's divide it next time;
        # if we've divided it, let's multiply it next time.
        self.muldiv = 1 - self.muldiv
        # Set self.x to x - partial sum computed so far
        # or partial sum - x.
        self.x = cf((0, denominator)) - self.x
        # Compute the partial quotients 0..n-1 of self.better,
        # so that we can examine self.better.pq(n) in the next
        # iteration of the main loop.
        better_pq = self.better.pq
        for i in xrange(n):
            better_pq(i)

def _cf_ilog(x):
    """Return a tuple (floor(log(x)), x/e**floor(log(x))); the
    second element belongs to the range [1, e). Uses a cached
    list of e**(2**n) via _cf_exp_2_to_nth()."""

    k = characteristic = 0
    if x >= 1:
        while _cf_exp_2_to_nth(k) <= x:
            k += 1
        while k:
            k -= 1
            new_x = x/_cf_exp_2_to_nth(k)
            if new_x >= 1:
                characteristic += 2**k
                x = new_x
    else:
        while _cf_exp_2_to_nth(k) <= e/x:
            k += 1
        while k:
            k -= 1
            new_x = x*_cf_exp_2_to_nth(k)
            if new_x < e:
                characteristic -= 2**k
                x = new_x
    return cf(characteristic), cf(x)

class log(cf_base):
    """Calculate the logarithm of x in the given base, lazily
    decomposing x into an alternating product of e**(a bit too
    large [or exact] exponent) and 1/e**(a bit too small
    [or exact] exponent) and adding the exponents."""

    def __new__(cls, x, base=e):
        """Initialize the lazy calculation: set self.better
        to floor(log(x)) and self.x to x/floor(log(x))."""

        if base is not e:
            return log(x)/log(base)
        if isinstance(x, cf_base) and (x.pq(0) is None):
            return NaN
        if isinstance(x, float) and str(x)=='inf':
            return x
        if isinstance(x, float) and str(x)=='nan':
            return x
        if x <= 0:
            raise (ValueError,
                'the logarithm of a non-positive number cannot be computed')
        self = object.__new__(cls)
        self.better, self.x = _cf_ilog(x)
        self.worse = NaN
        self.addsub = 1
        self.cache = []
        return self

    def pq(self, n):
        """Return the common partial quotients of the last partial
        sum and the penultimate partial sum of the exponents,
        extending the partial sum if necessary.

        After an exponent 1/q, the next exponent in the sum is at
        most 1/(q*(q+1)), so the accuracy of the approximation at
        least doubles with each term."""

        if n < len(self.cache):
            return self.cache[n]
        # We need to compute another term.
        while 1:
            q = self.worse.pq(n)
            if q == self.better.pq(n):
                # x always lies between self.better == e**(the
                # last partial sum) and self.worse == e**(the
                # penultimate partial sum), and e**x is
                # monotonic, so when the partial quotients of
                # the two approximations coincide, the actual
                # partial quotient of log(x) will be equal to
                # them.
                self.cache.append(q)
                return q
            # Compute self.x.pq(0), so that
            # we can compute self.x.pq(1).
            self.x.pq(0)
            assert 1 <= self.x.pq(0) <= 2
            if self.x.pq(0) == 2:
                # self.x >= 2.
                inverse_exponent = 1
                next_x = e/self.x
            elif self.x.pq(1) is None:
                # self.x == 1; log(x) has a finite alternating
                # sum representation (it's a rational number).
                self.worse = self.better
                continue
            else:
                # Here 1 < self.x < 2; we look at x.term(1)
                # to determine the least integer k such that
                # exp(1/k) > self.x.
                inverse_exponent = self.x.pq(1) + 1
                # We check whether exp(1/k)/self.x >= 1 instead of
                # (exp(1/k) > self.x), beacuse the latter expression
                # might give a different result for self.x close to
                # exp(1/k), and we will need next_x later, anyway.
                next_x = _cf_exp_1n(inverse_exponent)/self.x
                if next_x.pq(0) < 1:
                    # self.x is less than cf(1; k, 1, 1, 3*k+2, ...),
                    # but it must be greater than cf(1; k-1, ...).
                    inverse_exponent -= 1
                    next_x = _cf_exp_1n(inverse_exponent)/self.x
            assert next_x.pq(0) == 1
            self.worse = self.better
            # When self.addsub == +1, self.better += 1/inverse_exponent.
            # When self.addsub == -1, self.better -= 1/inverse_exponent.
            self.better = unop(self.better,
                inverse_exponent, self.addsub, 0, inverse_exponent)
            # If we've subtracted from self.better, let's add to it
            # next time; if we've added to it, let's subtract from
            # it next time.
            self.addsub = -self.addsub
            # Set self.x to x/e**(partial sum computed so far)
            # or its inverse.
            self.x = next_x
            # Compute the partial quotients 0..n-1 of self.better,
            # so that we can examine self.better.pq(n) in the next
            # iteration of the main loop.
            better_pq = self.better.pq
            for i in xrange(n):
                better_pq(i)

log_of_10 = log(10)
log_of_2 = log(2)

def log10(x):
    """Return the decimal logarithm of x."""

    return log(x)/log_of_10

def sinh(x):
    """Return the hyperbolic sine of x."""

    if isinstance(x, float) and ( str(x)=='inf' or str(x)=='-inf'):
        return x

    # returns (exp(x) - exp(-x))/2
    return binop(exp(x), exp(-x), 0, 1, -1, 0, 0, 0, 0, 2)

def cosh(x):
    """Return the hyperbolic cosine of x."""
    if isinstance(x, float) and ( str(x)=='inf' or str(x)=='-inf'):
        return float('inf')

    # returns (exp(x) + exp(-x))/2
    return binop(exp(x), exp(-x), 0, 1, 1, 0, 0, 0, 0, 2)

def tanh(x):
    """Return the hyperbolic tangent of x."""
    if isinstance(x, float) and str(x)=='inf':
        return 1

    if isinstance(x, float) and str(x)=='-inf':
        return -1

    if isinstance(x, float) and str(x)=='nan':
        return x

    if x==0:
	return x
    # returns (exp(x) - exp(-x))/(exp(x) + exp(-x))
    return binop(exp(x), exp(-x), 0, 1, -1, 0, 0, 1, 1, 0)

class _cf_tan_1n(cf_base):
    """Return tan(1/n) == cf(0;n-1,1,3*n-2,1,5*n-2,1,...)."""

    def __new__(cls, inverse_argument):
        """Set self.term to a closure that returns the nth
        partial quotient of tan(1/inverse_argument) quickly,
        regardless of n."""

        self = object.__new__(cls)
        if inverse_argument > 1:
            # This is the most frequent case; called repeatedly
            # from _cf_tan.term().
            def tan_1n_pq_closure(index):
                if index > 1:
                    if index%2:
                        return index*inverse_argument - 2
                    else:
                        return 1
                elif index:
                    return inverse_argument-1
                else:
                    return 0
            self.pq = tan_1n_pq_closure
            return self
        elif inverse_argument == 1:
            # The above closure would also work for inverse_argument
            # equal to 1, but would emit 0 as the partial quotient at
            # index 1. By special-casing tan(1/1) we avoid emitting
            # the zero partial quotient.
            def tan_1_pq(index):
                if index%2:
                    return index
                else:
                    return 1
            self.pq = tan_1_pq
            return self
        elif inverse_argument:
            # Don't choke on negative arguments.
            return -_cf_tan_1n(-inverse_argument)
        else:
            # tan(1/0) == NaN.
            return NaN

class _cf_tan(cf_base2):
    """Calculate the tangent of x for 0 <= x <= pi/4
    (actually even for 0 <= x < 1), lazily decomposing x
    into the Ostrogradsky series of second kind."""

    def __new__(cls, x):
        """Initialize the lazy calculation: set self.better
        to tan(0) == 0 and self.x to x."""

        if x.pq(0) is None: return NaN
        assert x.pq(0) == 0
        self = object.__new__(cls)
        self.x = x
        self.better = zero
        self.addsub = 1
        self.worse = NaN
        self.cache = []
        return self

    def pq_own(self,n,denominator):
        # When self.addsub == +1, self.better =
        #     (self.better + tan(1/n))/(1 - self.better*tan(1/n))
        # When self.addsub == -1, self.better =
        #     (self.better - tan(1/n))/(1 + self.better*tan(1/n))
        if self.better is zero:
            self.better = _cf_tan_1n(denominator)
        else:
            self.better = binop(
                self.better, _cf_tan_1n(denominator),
                0, 1, self.addsub, 0, -self.addsub, 0, 0, 1)
        # Change the operation to the opposite one.
        self.addsub = -self.addsub
        # Set self.x to x - partial sum computed so far
        # or partial sum - x.
        self.x = cf((0, denominator)) - self.x
        # Compute the partial quotients 0..n-1 of self.better,
        # so that we can examine self.better.pq(n) in the next
        # iteration of the main loop.
        better_pq = self.better.pq
        for i in xrange(n):
            better_pq(i)

def tan(x):
    """Return the tangent of x."""

    if isinstance(x, float) and str(x)=='nan':
        #returns NaN
        return float('nan')
    if isinstance(x, float) and str(x)=='inf':
        raise ValueError,"math domain error"
    if isinstance(x, float) and str(x)=='-inf':
        raise ValueError,"math domain error"
    octant = x//quarter_pi
    reduced_octant = octant%4
    if reduced_octant < 2:
        if reduced_octant == 0:
            return _cf_tan(x - octant*quarter_pi)
        else:
            return 1/_cf_tan((octant + 1)*quarter_pi - x)
    else:
        if reduced_octant == 2:
            return -1/_cf_tan(x - octant*quarter_pi)
        else:
            return -_cf_tan((octant + 1)*quarter_pi - x)

def sin(x):
    """Return the sine of x."""

    if isinstance(x, float) and str(x)=='nan':
        return x
    if isinstance(x, float) and str(x)=='inf':
        raise ValueError,"math domain error"
    if isinstance(x, float) and str(x)=='-inf':
        raise ValueError,"math domain error"
    # returns(2*tan(x/2))/(1+tan(x/2)**2)
    tangent = tan(x/2)
    return binop(tangent, tangent, 0, 2, 0, 0, 1, 0, 0, 1)

def cos(x):
    """Return the cosine of x."""
    if isinstance(x, float) and str(x)=='nan':
        return x

    if isinstance(x, float) and str(x)=='inf':
        raise ValueError,"math domain error"
    if isinstance(x, float) and str(x)=='-inf':
        raise ValueError,"math domain error"
    # returns (1-tan(x/2)**2)/(1+tan(x/2)**2)
    tangent = tan(x/2)
    return binop(tangent, tangent, -1, 0, 0, 1, 1, 0, 0, 1)

def degrees(x):
    """Convert radians to degrees."""

    # returns 180*x/pi
    if isinstance(x, (int, long)):
        return unop(pi, 0, 180*x, 1, 0)
    else:
        return binop(cf(x), pi, 0, 180, 0, 0, 0, 0, 1, 0)

def radians(x):
    """Convert degrees to radians."""

    # returns pi*x/180
    if isinstance(x, (int, long)):
        return unop(pi, x, 0, 0, 180)
    else:
        return binop(cf(x), pi, 1, 0, 0, 0, 0, 0, 0, 180)

''' gamma function and its natural logarithm'''

##/* @(#)er_lgamma.c 5.1 93/09/24 */
##/*
## * ====================================================
## * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
## *
## * Developed at SunPro, a Sun Microsystems, Inc. business.
## * Permission to use, copy, modify, and distribute this
## * software is freely granted, provided that this notice 
## * is preserved.
## * ====================================================
## *
## */
##original code from: http://www.sourceware.org/cgi-bin/cvsweb.cgi/~checkout~/src/newlib/libm/mathfp/er_lgamma.c?rev=1.6&content-type=text/plain&cvsroot=src
##/* Method:
## *   1. Argument Reduction for 0 < x <= 8
## * 	Since gamma(1+s)=s*gamma(s), for x in [0,8], we may 
## * 	reduce x to a number in [1.5,2.5] by
## * 		lgamma(1+s) = log(s) + lgamma(s)
## *	for example,
## *		lgamma(7.3) = log(6.3) + lgamma(6.3)
## *			    = log(6.3*5.3) + lgamma(5.3)
## *			    = log(6.3*5.3*4.3*3.3*2.3) + lgamma(2.3)
## *   2. Polynomial approximation of lgamma around its
## *	minimun ymin=1.461632144968362245 to maintain monotonicity.
## *	On [ymin-0.23, ymin+0.27] (i.e., [1.23164,1.73163]), use
## *		Let z = x-ymin;
## *		lgamma(x) = -1.214862905358496078218 + z^2*poly(z)
## *	where
## *		poly(z) is a 14 degree polynomial.
## *   2. Rational approximation in the primary interval [2,3]
## *	We use the following approximation:
## *		s = x-2.0;
## *		lgamma(x) = 0.5*s + s*P(s)/Q(s)
## *	with accuracy
## *		|P/Q - (lgamma(x)-0.5s)| < 2**-61.71
## *	Our algorithms are based on the following observation
## *
## *                             zeta(2)-1    2    zeta(3)-1    3
## * lgamma(2+s) = s*(1-Euler) + --------- * s  -  --------- * s  + ...
## *                                 2                 3
## *
## *	where Euler = 0.5771... is the Euler constant, which is very
## *	close to 0.5.
## *
## *   3. For x>=8, we have
## *	lgamma(x)~(x-0.5)log(x)-x+0.5*log(2pi)+1/(12x)-1/(360x**3)+....
## *	(better formula:
## *	   lgamma(x)~(x-0.5)*(log(x)-1)-.5*(log(2pi)-1) + ...)
## *	Let z = 1/x, then we approximation
## *		f(z) = lgamma(x) - (x-0.5)(log(x)-1)
## *	by
## *	  			    3       5             11
## *		w = w0 + w1*z + w2*z  + w3*z  + ... + w6*z
## *	where 
## *		|w - f(z)| < 2**-58.74
## *		
## *   4. For negative x, since (G is gamma function)
## *		-x*G(-x)*G(x) = pi/sin(pi*x),
## * 	we have
## * 		G(x) = pi/(sin(pi*x)*(-x)*G(-x))
## *	since G(-x) is positive, sign(G(x)) = sign(sin(pi*x)) for x<0
## *	Hence, for x<0, signgam = sign(sin(pi*x)) and 
## *		lgamma(x) = log(|Gamma(x)|)
## *			  = log(pi/(|x*sin(pi*x)|)) - lgamma(-x);
## *	Note: one should avoid compute pi*(-x) directly in the 
## *	      computation of sin(pi*(-x)).
## *		
## *   5. Special Cases
## *		lgamma(2+s) ~ s*(1-Euler) for tiny s
## *		lgamma(1)=lgamma(2)=0
## *		lgamma(x) ~ -log(x) for tiny x
## *		lgamma(0) = lgamma(inf) = inf
## *	 	lgamma(-integer) = +-inf
## *	
## */
##


two52=  4.50359962737049600000e+15
half=  5.00000000000000000000e-01
a0  =  7.72156649015328655494e-02
a1  =  3.22467033424113591611e-01
a2  =  6.73523010531292681824e-02
a3  =  2.05808084325167332806e-02
a4  =  7.38555086081402883957e-03
a5  =  2.89051383673415629091e-03
a6  =  1.19270763183362067845e-03
a7  =  5.10069792153511336608e-04
a8  =  2.20862790713908385557e-04
a9  =  1.08011567247583939954e-04
a10 =  2.52144565451257326939e-05
a11 =  4.48640949618915160150e-05
tc  =  1.46163214496836224576e+00
tf  = -1.21486290535849611461e-01
tt  = -3.63867699703950536541e-18
t0  =  4.83836122723810047042e-01
t1  = -1.47587722994593911752e-01
t2  =  6.46249402391333854778e-02
t3  = -3.27885410759859649565e-02
t4  =  1.79706750811820387126e-02
t5  = -1.03142241298341437450e-02
t6  =  6.10053870246291332635e-03
t7  = -3.68452016781138256760e-03
t8  =  2.25964780900612472250e-03
t9  = -1.40346469989232843813e-03
t10 =  8.81081882437654011382e-04
t11 = -5.38595305356740546715e-04
t12 =  3.15632070903625950361e-04
t13 = -3.12754168375120860518e-04
t14 =  3.35529192635519073543e-04
u0  = -7.72156649015328655494e-02
u1  =  6.32827064025093366517e-01
u2  =  1.45492250137234768737e+00
u3  =  9.77717527963372745603e-01
u4  =  2.28963728064692451092e-01
u5  =  1.33810918536787660377e-02
v1  =  2.45597793713041134822e+00
v2  =  2.12848976379893395361e+00
v3  =  7.69285150456672783825e-01
v4  =  1.04222645593369134254e-01
v5  =  3.21709242282423911810e-03
s0  = -7.72156649015328655494e-02
s1  =  2.14982415960608852501e-01
s2  =  3.25778796408930981787e-01
s3  =  1.46350472652464452805e-01
s4  =  2.66422703033638609560e-02
s5  =  1.84028451407337715652e-03
s6  =  3.19475326584100867617e-05
r1  =  1.39200533467621045958e+00
r2  =  7.21935547567138069525e-01
r3  =  1.71933865632803078993e-01
r4  =  1.86459191715652901344e-02
r5  =  7.77942496381893596434e-04
r6  =  7.32668430744625636189e-06
w0  =  4.18938533204672725052e-01
w1  =  8.33333333333329678849e-02
w2  = -2.77777777728775536470e-03
w3  =  7.93650558643019558500e-04
w4  = -5.95187557450339963135e-04
w5  =  8.36339918996282139126e-04
w6  = -1.63092934096575273989e-03

def sin_pi(x):
    x = float(x)
    e,ix = frexp(x)
    if(abs(x)<0.25):
        return -sin(pi*x)
    y = -x  ##/* x is assume negative */

## * argument reduction, make sure inexact flag not raised if input
## * is an integer
    z = floor(y)
    if(z!=y):
        y *= 0.5
        y = 2.0*(y - floor(y))  ##/* y = |x| mod 2.0 */
        n = int(y*4.0)
    else:
        if(abs(ix)>=53):
            y = zero
            n = 0  ##/* y must be even */
        else:
            if(abs(ix)<52):
                z = y+two52 ##/* exact */
            e,n=frexp(z)
            n=int(n)
            e=int(e)
            n &= 1
            y  = n
            n<<= 2

    if n == 0:
        y = sin(pi*y)
    elif (n == 1 or n == 2):
        y = cos(pi*(0.5-y))
    elif (n == 3 or n == 4):
        y = sin(pi*(one-y))
    elif (n == 5 or n == 6):
        y = -cos(pi*(y-1.5))
    else:
        y = sin(pi*(y-2.0))

    z = cos(pi*(z+1.0));
    return -y*z

def lgamma(x):
    '''return natural logarithm of gamma function of x
    raise ValueError if x is negative integer'''
    inf=NaN
    x = cf(x)

    if isnan(x):
       return NaN
    if isinf(x):
       return NaN
    e,ix = frexp(x)
    nadj = 0
    signgamp = 1

    if ((e==0.0) and (ix==0)):
        raise ValueError

    if (ix>1020):
	raise OverflowError
    
    if ((e != 0.0) and (ix<-71)):
        if (x<0):
            return -log(-x)
        else:
            return -log(x)

    if (e<0):
        if (ix>52):
            return inf ##one/zero
        t = sin_pi(x)
        if (t==zero):
            raise ValueError('gamma not defined for negative integer')
        nadj = log(pi/fabs(t*x))
        if (t<zero):
            signgamp = -1
        x = -x

    ##/* purge off 1 and 2 */
    if ((x==2.0) or (x==1.0)):
        r = 0.0
        
    ##/* for x < 2.0 */
    elif (ix<2):
        if(x<=0.9): ##/* function lgamma(x) = lgamma(x+1)-log(x) */
            r = -log(x)
            if (x>=0.7316):
                y = one-x
                z = y*y
                p1 = a0+z*(a2+z*(a4+z*(a6+z*(a8+z*a10))))
                p2 = z*(a1+z*(a3+z*(a5+z*(a7+z*(a9+z*a11)))))
                p  = y*p1+p2
                r  += (p-0.5*y)
            elif (x>=0.23164):
                y= x-(tc-one)
                z = y*y
                w = z*y
                p1 = t0+w*(t3+w*(t6+w*(t9 +w*t12))) ##	/* parallel comp */
                p2 = t1+w*(t4+w*(t7+w*(t10+w*t13)))
                p3 = t2+w*(t5+w*(t8+w*(t11+w*t14)))
                p  = z*p1-(tt-w*(p2+y*p3))
                r += (tf + p)
            else:
                y = x
                p1 = y*(u0+y*(u1+y*(u2+y*(u3+y*(u4+y*u5)))))
                p2 = one+y*(v1+y*(v2+y*(v3+y*(v4+y*v5))))
                r += (-0.5*y + p1/p2)
        else:
            r = zero
            if(x>=1.7316):
                y=2.0-x ##/* [1.7316,2] */
                z = y*y
                p1 = a0+z*(a2+z*(a4+z*(a6+z*(a8+z*a10))))
                p2 = z*(a1+z*(a3+z*(a5+z*(a7+z*(a9+z*a11)))))
                p  = y*p1+p2
                r  += (p-0.5*y)
            elif(x>=1.23164):
                y=x-tc ##/* [1.23,1.73] */
                z = y*y
                w = z*y
                p1 = t0+w*(t3+w*(t6+w*(t9 +w*t12))) ##	/* parallel comp */
                p2 = t1+w*(t4+w*(t7+w*(t10+w*t13)))
                p3 = t2+w*(t5+w*(t8+w*(t11+w*t14)))
                p  = z*p1-(tt-w*(p2+y*p3))
                r += (tf + p)
            else:
                y=x-one
                p1 = y*(u0+y*(u1+y*(u2+y*(u3+y*(u4+y*u5)))))
                p2 = one+y*(v1+y*(v2+y*(v3+y*(v4+y*v5))))
                r += (-0.5*y + p1/p2)
    ##/* x < 8.0 */
    elif(ix<4):
        i = int(x)
        t = zero
        y = x-i
        p = y*(s0+y*(s1+y*(s2+y*(s3+y*(s4+y*(s5+y*s6))))))
        q = one+y*(r1+y*(r2+y*(r3+y*(r4+y*(r5+y*r6)))))
        r = half*y+p/q
        z = one ##/* function lgamma(1+s) = log(s) + lgamma(s) */
        while (i>2):
            i -=1
            z *= (y+i)
        r += log(z)

    ##/* 8.0 <= x < 2**58 */
    elif (ix<58):
        t = log(x)
        z = one/x
        y = z*z
        w = w0+z*(w1+y*(w2+y*(w3+y*(w4+y*(w5+y*w6)))))
        r = (x-half)*(t-one)+w

    ##/* 2**58 <= x <= inf */
    else:
        r =  x*(log(x)-one)

    if (e<0):
        r = nadj - r
    return signgamp*r

def gamma(x):
    '''return gamma function of x
    raise ValueError  if x is negative integer'''
    x = cf(x)
    if isinf(x):
        return cf(())
    if x == 0.0:
        raise ValueError
    s = 1.0
    if (x<0):
        s = cos(pi*floor(x))
    return s*exp(lgamma(x))

def expm1(x):
	return exp(x)-1

class _cf_atan(cf_base):
    """Calculate the inverse tangent of x for 0 <= x <= 1,
    lazily decomposing x into tan**(partial sum of the
    Ostrogradsky series of the second kind of x)."""

    def __new__(cls, x):
        """Initialize the lazy calculation: set self.better
        to atan(0) == 0 and self.x to x."""

        self = object.__new__(cls)
        self.better = zero
        self.worse = NaN
        self.x = cf(x)
        self.add_sub = 1
        self.cache = []
        return self

    def pq(self, n):
        """Return the common partial quotients of the last partial
        sum and the penultimate partial sum of the exponents,
        extending the partial sum if necessary.

        After an exponent 1/q, the next exponent in the sum is at
        most 1/(q*(q+1)), so the accuracy of the approximation at
        least doubles with each term."""

        if n < len(self.cache):
            return self.cache[n]
        # We need to compute another term.
        while 1:
            q = self.worse.pq(n)
            if q == self.better.pq(n):
                # x always lies between self.better == tan(the
                # last partial sum) and self.worse == tan(the
                # penultimate partial sum), and tan(x) is
                # monotonic for 0 <= x <= pi/2, so when the
                # partial quotients of the two approximations
                # coincide, the actual partial quotient of
                # atan(x) will be equal to them.
                self.cache.append(q)
                return q
            # Compute self.x.pq(0), so that
            # we can compute self.x.pq(1).
            self.x.pq(0)
            assert 0 <= self.x.pq(0) <= 1
            if self.x.pq(0) == 1:
                # self.x >= 1.
                inverse_argument = 1
                next_x = binop(_cf_tan_1n(1), self.x,
                    0, 1, -1, 0, 1, 0, 0, 1)
            elif self.x.pq(1) is None:
                # self.x == 1; atan(x) has a finite alternating
                # sum representation (it's a rational number).
                self.worse = self.better
                continue
            else:
                # Here 0 < self.x < 1; we look at x.pq(1)
                # to determine the least integer k such
                # that tan(1/k) > self.x.
                inverse_argument = self.x.pq(1) + 1
                # We check whether (tan(1/k)-self.x)/(1+tan(1/k)*self.x
                # >= 0 instead of (tan(1/k) > self.x), beacuse the
                # latter expression might give a different result for
                # self.x close to tan(1/k), and we will need next_x
                # later, anyway.
                next_x = binop(_cf_tan_1n(inverse_argument), self.x,
                    0, 1, -1, 0, 1, 0, 0, 1)
                if next_x.pq(0) < 0:
                    # self.x is less than cf(0; k, 1,3*k+1, ...),
                    # but it must be greater than cf(0; k-1, ...).
                    inverse_argument -= 1
                    next_x = binop(_cf_tan_1n(inverse_argument), self.x,
                        0, 1, -1, 0, 1, 0, 0, 1)
                assert next_x.pq(0) == 0
            self.worse = self.better
            # When self.add_sub == +1, self.better += 1/inverse_argument.
            # When self.add_sub == -1, self.better -= 1/inverse_argument.
            if self.better is zero:
                self.better = cf((0, inverse_argument))
            else:
                self.better = unop(self.better,
                    inverse_argument, self.add_sub, 0, inverse_argument)
            # If we've subtracted from self.better, let's add to it
            # next time; if we've added to it, let's subtract from
            # it next time.
            self.add_sub = -self.add_sub
            # Set self.x to abs(tan(partial sum computed so far)).
            self.x = next_x
            better_pq = self.better.pq
            # Compute the partial quotients 0..n-1 of self.better,
            # so that we can examine self.better.pq(n) in the next
            # iteration of the main loop.
            for i in xrange(n):
                better_pq(i)

def atan(x):
    """Return the arc tangent of x."""
    if isinstance(x, float) and str(x)=='inf':
        return half_pi
    if isinstance(x, float) and str(x)=='-inf':
        return -half_pi
    if isinstance(x, float) and str(x)=='nan':
        return float('nan')
    x = cf(x)
    if x.pq(0) is None: return NaN
    elif x.pq(0) < 0:
        if x.pq(0) == -1:
            return -_cf_atan(-x)
        else:
            return _cf_atan(-1/x) - half_pi
    else:
        if x.pq(0) == 0:
            return _cf_atan(x)
        else:
            return half_pi - _cf_atan(1/x)

def asin(x):
    """Return the arc sine of x."""

    if isinstance(x, float) and str(x)=='-inf':
         raise ValueError,"math domain error"
    if isinstance(x, float) and str(x)=='inf':
         raise ValueError,"math domain error"
    if isinstance(x, float) and str(x)=='nan':
        return float('nan')
    x1 = cf(x)
    if x1.pq(0) is None: return NaN
    elif ((x1.pq(0) < -1) or (x1.pq(0) > +1)
    or ((x1.pq(0) == +1) and (x1.pq(1) is not None))):
        raise ValueError, 'the argument must lie in the range [-1,+1]'
    elif (x1.pq(0) == -1) and (x1.pq(1) is None):
        return -half_pi
    elif (x1.pq(0) == +1) and (x1.pq(1) is None):
        return half_pi
    else:
        # returns atan(x/sqrt(1-x**2))
        return _cf_atan(x/sqrt(binop(x1, x1, -1, 0, 0, 1, 0, 0, 0, 1)))

def acos(x):
    """Return the arc cosine of x."""

    if isinstance(x, float) and str(x)=='-inf':
         raise ValueError,"math domain error"
    if isinstance(x, float) and str(x)=='inf':
         raise ValueError,"math domain error"
    if isinstance(x, float) and str(x)=='nan':
        return float('nan')
    return half_pi - asin(x)

def erf(x):
    if isnan(x) or abs(x)==0:
        return x
    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # Save the sign of x
    sign = 1
    if x < 0:
        sign = -1
    x = abs(x)
    z=26
    if x>z or x<-z:
        return sign*1.0

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x)

    return sign*y

def erfc(x):
    return 1-erf(x)

def atan2(y, x):
    """Return the arc tangent of y/x."""

    if isinstance(x, float) and str(x)=='nan':
        return float('nan')
    if isinstance(y, float) and str(y)=='nan':
        return float('nan')
    if isinstance(y, float) and str(y)=='inf':
        if isinstance(x, float) and str(x)=='-inf':
             return (pi*3)/4
	else:
             if isinstance(x, float) and str(x)=='inf':
                 return pi/4
	     else:
                 return half_pi
    else:
        if isinstance(y, float) and str(y)=='-inf':
            if isinstance(x, float) and str(x)=='-inf':
                 return -(pi*3)/4
	    else:
	         if isinstance(x, float) and str(x)=='inf':
                       return -pi/4
	         else:
                       return -half_pi
		
        
    s=copysign(1,y)
    sx=copysign(1,x)
    y = cf(y)
    if x > 0:
        if isinstance(x, float) and str(x)=='inf':
            return 0.0
        return atan(y/x)
    elif x < 0:
        if y >= 0:
            # Continued fractions don't sport a signed zero,
            # so we always return pi for x<0, y==0.
            if isinstance(x, float) and str(x)=='-inf':
                if s>0:
                     return pi
                else:
                     return -pi

            return atan(y/x) + pi*s
        else:
            if isinstance(x, float) and str(x)=='-inf':
                return -pi
            return atan(y/x) - pi
    else:
        if y > 0:
            return half_pi
        elif y < 0:
            return -half_pi
        else:
            if sx>0:
                return zero
            if s<0:
                return -pi
            return pi

class _cf_pi(cf_base):
    """Regular continued fraction for pi."""

    def _cf_pi_generator(self):
        """Return subsequent partial quotients of pi, using
        a generalized continued fraction for 4/pi."""
    
        def gcd(x, y):
            """Return the gcd of x and y."""
            while y:
                x, y = y, x%y
            return x
    
        # c and d would equal zero at the very beginning of
        # the iteration if we had started it from scratch with
        # a,b,c,d,p,q = 4,0,1,1,1,3. By emitting the first
        # partial quotient verbatim and initializing the
        # parameters with later values, we dispose of the
        # cases c == 0 and d == 0 inside the loop.
        yield 3
        a, b, c, d, p, q = 51, 6, 7, 1, 16, 9
        while 1:
            ac, a_mod_c = divmod(a, c)
            bd, b_mod_d = divmod(b, d)
            if ac == bd:
                yield ac
                a, b, c, d = c, d, a_mod_c, b_mod_d
            else:
                a, b, c, d = p*b + q*a, a, p*d + q*c, c
                # The p's are consecutive squares;
                # the q's are consecutive odd integers.
                p += q
                q += 2
                if q&2047 == 1:
                    # Once in a while reduce a, b, c, d.
                    gcd_abcd = gcd(a, gcd(c, gcd(b, d)))
                    a //= gcd_abcd
                    b //= gcd_abcd
                    c //= gcd_abcd
                    d //= gcd_abcd

    def __new__(cls):
        """Memoize the results of _cf_pi_generator()
        via self.pq() inherited from cf_base."""

        self = object.__new__(cls)
        self.cache = []
        self.next_pq = self._cf_pi_generator().next
        return self

# Singletons for pi, pi/2 and pi/4.
pi = _cf_pi()
half_pi = pi/2
quarter_pi = pi/4


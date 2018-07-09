#!/usr/bin/python
# Python test set -- math module
import unittest
from cf import e,cf,sqrt,NaN,radians,exp,pi,digits,degrees,cos,sin,tan,atan2,erf,asin,fsum
import math
import sys
import cf as CF
from random import random, seed
from functools import wraps
def capture(f):
	from cStringIO import StringIO
	@wraps(f)
	def h(*x, **y):
		o,sys.stdout= sys.stdout,StringIO()
		w=f(*(x+(sys.stdout,)),**y)
		sys.stdout=o
		return w
	return h

seed(42)
x = cf(3*random())
y = cf(3*random())

def digit_by_digit(x, precision):
	get_digit = digits(x).next
	try:
		integer_part = get_digit()
		if integer_part < 0:
			sys.stdout.write('-')
			digit_by_digit(-x, precision)
		else:
			sys.stdout.write(str(integer_part) + '.')
	except StopIteration:
		sys.stdout.write('NaN')
		return
	try:
		for i in xrange(precision):
			sys.stdout.write(str(get_digit()))
			sys.stdout.flush()
	except StopIteration:
		pass

class test_cf(unittest.TestCase):
	@capture
	def test_cf2(self,output):
		w1='2.7182818284590452353602874713526624977572470936999595749669676277240766303535475945713821785251664274274663919320030599218174135966290435729003342952605956307381323286279434907632338298807531952510190115738341879307021540891499348841675092447614606680822648001684774118537423454424371075390777449920695517027618386062613313845830007520449338265602976067371132007093287091274437470472306969772093101416'
		ev="digit_by_digit(e, 400)"
		w=eval(ev)
		self.assertEqual(w,None)
		w=output.getvalue()
		self.assertEqual(w,w1)

	@capture
	def test_cf3(self,output):
		w1='3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865132823066470938446095505822317253594081284811174502841027019385211055596446229489549303819644288109756659334461284756482337867831652712019091456485669234603486104543266482133936072602491412737245870066063155881748815209209628292540917153643678925903600113305305488204665213841469519415116094'
		ev="digit_by_digit(pi, 400)"
		w=eval(ev)
		self.assertEqual(w,None)
		w=output.getvalue()
		self.assertEqual(w,w1)

	@capture
	def test_cf4(self,output):
		w1='1.4142135623730950488016887242096980785696718753769480731766797379907324784621070388503875343276415727350138462309122970249248360558507372126441214970999358314132226659275055927557999505011527820605714701095599716059702745345968620147285174186408891986095523292304843087143214508397626036279952514079896872533965463318088296406206152583523950547457502877599617298355752203375318570113543746034084988471' 
		ev="digit_by_digit(sqrt(2), 400)"
		w=eval(ev)
		self.assertEqual(w,None)
		w=output.getvalue()
		self.assertEqual(w,w1)

	@capture
	def test_cf5(self,output):
		w1="262537412640768743.9999999999992500725971981856888793538563373369908627075374103782106479101186073129511813461860645041930838879497538640449057287144771968148523224320391164782914886422827201311783170650104522268780144484177034696946335570768172388768100092370653951938650636275765788855822394811427691210083088665110728471062346581129818301245913283610006498266592365172617883086371078645219552815427466510961100147250"
		ev="digit_by_digit(exp(pi*sqrt(163)), 400)"
		w=eval(ev)
		self.assertEqual(w,None)
		w=output.getvalue()
		self.assertEqual(w,w1)

	def test_cf6(self):
		ev="(math.sqrt(5)-1)/2 - (sqrt(5)-1)/2"
		w=eval(ev)
		self.assertAlmostEqual(float(w),5.4321152036825058837006863671e-17)

	def test_cf7(self):
		ev="	   cf(math.sqrt(2))**2 - 2"
		w=eval(ev)
		self.assertEqual(w,2.7343234630647692806884916507e-16)

	def test_cf8(self):
		ev='"%g" % (math.sqrt(2)**2 - 2.0)'
		w=eval(ev)
		self.assertEqual(w,"4.44089e-16")

	def test_cf9(self):
		ev="(sqrt(1000001) + sqrt(1000025) + sqrt(1000031) + sqrt(1000084) + sqrt(1000087) + sqrt(1000134) + sqrt(1000158) + sqrt(1000182) + sqrt(1000198)) -(sqrt(1000002) + sqrt(1000018) + sqrt(1000042) + sqrt(1000066) + sqrt(1000113) + sqrt(1000116) + sqrt(1000169) + sqrt(1000175) + sqrt(1000199))"
		w=eval(ev)
		self.assertAlmostEqual(w,-3.3134865714985754023398832718e-37)


	def test_cf10(self):
		ev="x"
		w=eval(ev)
		self.assertEqual(w,1.9182803953736513591366019682)

	def test_cf11(self):
		ev="y"
		w=eval(ev)
		self.assertEqual(w,0.0750322656680008082119570644)

	def test_cf12(self):
		ev="-x"
		w=eval(ev)
		self.assertEqual(w,-1.9182803953736513591366019682)

	def test_cf13(self):
		ev="-y"
		w=eval(ev)
		self.assertEqual(w,-0.0750322656680008082119570644)

	def test_cf14(self):
		ev="x + y"
		w=eval(ev)
		self.assertAlmostEqual(w,1.9933126610416521673485590326)

	def test_cf15(self):
		ev="y + x"
		w=eval(ev)
		self.assertAlmostEqual(w,1.9933126610416521673485590326)

	def test_cf16(self):
		ev="x - y"
		w=eval(ev)
		self.assertAlmostEqual(w,1.8432481297056505509246449037)

	def test_cf17(self):
		ev="y - x"
		w=eval(ev)
		self.assertAlmostEqual(w,-1.8432481297056505509246449037)

	def test_cf18(self):
		ev="x * y"
		w=eval(ev)
		self.assertAlmostEqual(w,0.1439329242513934372830993413)

	def test_cf19(self):
		ev="y * x"
		w=eval(ev)
		self.assertAlmostEqual(w,0.1439329242513934372830993413)

	def test_cf20(self):
		ev="x / y"
		w=eval(ev)
		self.assertAlmostEqual(w,25.5660731859220004638626011148)

	def test_cf21(self):
		ev="y / x"
		w=eval(ev)
		self.assertAlmostEqual(w,0.0391143369076582170171653236)

	def test_cf22(self):
		ev="x % y"
		w=eval(ev)
		self.assertEqual(w,0.0424737536736311538376753560)

	def test_cf23(self):
		ev="y % x"
		w=eval(ev)
		self.assertEqual(w,0.0750322656680008082119570644)

	def test_cf24(self):
		ev="x**y"
		w=eval(ev)
		self.assertAlmostEqual(w,1.0500924475670624944476307955)

	def test_cf25(self):
		ev="y**x"
		w=eval(ev)
		self.assertAlmostEqual(w,0.0069568142062713504691627463)


	def test_cf26(self):
		ev="e < e"
		w=eval(ev)
		self.assertEqual(w,False)

	def test_cf27(self):
		ev="e <= e"
		w=eval(ev)
		self.assertEqual(w,True)

	def test_cf28(self):
		ev="e == e"
		w=eval(ev)
		self.assertEqual(w,True)

	def test_cf29(self):
		ev="e >= e"
		w=eval(ev)
		self.assertEqual(w,True)

	def test_cf30(self):
		ev="e > e"
		w=eval(ev)
		self.assertEqual(w,False)

	def test_cf31(self):
		ev="e - e"
		w=eval(ev)
		self.assertEqual(w,0.)

	def test_cf32(self):
		ev="sqrt(2)**2"
		w=eval(ev)
		self.assertEqual(w,2.)

	def test_cf33(self):
		ev="(1 - sqrt(2))*(1 + sqrt(2))"
		w=eval(ev)
		self.assertEqual(w,-1.)

	def test_cf34(self):
		ev="cf(16)**0.25"
		w=eval(ev)
		self.assertEqual(w,2.)

	def test_cf35(self):
		ev="tan(pi/4)"
		w=eval(ev)
		self.assertEqual(w,1.)

	def test_cf36(self):
		ev="[sin(x)**2 + cos(x)**2 for x in [pi*random(),pi*random(),pi*random()]]"
		w=eval(ev)
		self.assertEqual(w,[cf(1), cf(1), cf(1)])

	def test_cf37(self):
		ev="[cos(3*x)/cos(x) == cos(x)**2-3*sin(x)**2 for x in [pi*random(),pi*random(),pi*random()]]"
		w=eval(ev)
		self.assertEqual(w,[True, True, True])

	def test_cf38(self):
		ev="sqrt(5 + 2*sqrt(6)) == sqrt(2) + sqrt(3)"
		w=eval(ev)
		self.assertEqual(w,True)

	def test_cf39(self):
		ev="degrees(2*pi)"
		w=eval(ev)
		self.assertEqual(w,360.)

	def test_cf40(self):
		ev="radians(360)/pi"
		w=eval(ev)
		self.assertEqual(w,2.)


	def test_cf41(self):
		ev="NaN"
		w=eval(ev)
		self.assertEqual(str(w),str(NaN))

	def test_cf42(self):
		ev="-NaN"
		w=eval(ev)
		self.assertEqual(str(w),str(NaN))

	def test_cf43(self):
		ev="0*NaN"
		w=eval(ev)
		self.assertEqual(str(w),str(NaN))

	def test_cf44(self):
		ev="NaN*0"
		w=eval(ev)
		self.assertEqual(str(w),str(NaN))

	def test_cf45(self):
		ev="NaN-NaN"
		w=eval(ev)
		self.assertEqual(str(w),str(NaN))

	def test_cf46(self):
		ev="NaN/NaN"
		w=eval(ev)
		self.assertEqual(str(w),str(NaN))

	def test_cf47(self):
		ev="1/NaN"
		w=eval(ev)
		self.assertEqual(str(w),str(NaN))

	def test_cf48(self):
		ev="cf(1)/0"
		w=eval(ev)
		self.assertEqual(str(w),str(NaN))

	def test_cf49(self):
		ev="cf(0)/0"
		w=eval(ev)
		self.assertEqual(str(w),str(NaN))

	def test_cf50(self):
		ev="0**NaN"
		w=eval(ev)
		self.assertEqual(str(w),str(NaN))

	def test_cf51(self):
		ev="NaN**0"
		w=eval(ev)
		self.assertEqual(str(w),str(NaN))

	def test_cf52(self):
		ev="NaN**2"
		w=eval(ev)
		self.assertEqual(str(w),str(NaN))

	def test_cf53(self):
		ev="NaN**NaN"
		w=eval(ev)
		self.assertEqual(str(w),str(NaN))

	def test_e(self):
		self.assertAlmostEqual(math.e,e)
		self.assertAlmostEqual(sin(0),0)
		self.assertAlmostEqual(cos(0),1)
		self.assertAlmostEqual(exp(0),1)
		self.assertAlmostEqual(sqrt(1),1)
		self.assertAlmostEqual(tan(0),0)
		self.assertAlmostEqual(radians(0),0)
		self.assertAlmostEqual(degrees(0),0)
		self.assertAlmostEqual(pi,3.14159265358979323846264338327)

	def test_precison(self):
		x_float = 671875/1000000.0
		x_cf = cf(671875,1000000)
		for i in xrange(1, 35):
			x_float = 4*x_float*(1-x_float)
			x_cf = 4*x_cf*(1-x_cf)
			self.assertAlmostEqual(x_float,x_cf)

		for i in xrange(35, 61):
			x_float = 4*x_float*(1-x_float)
			x_cf = 4*x_cf*(1-x_cf)
			self.assertNotAlmostEqual(x_float,x_cf)

	def test_atan2(self):
		self.assertAlmostEqual(atan2(-pi,-pi),math.atan2(-math.pi,-math.pi))

	def test_erf(self):
		self.assertNotAlmostEqual(erf(1),math.erf(1))

	def test_asin(self):
		self.assertRaises(ValueError,asin,2)

	def test_str(self):
		self.assertEqual(str(cf(-1.0)),"-1.")

	def test_str0(self):
		self.assertEqual(str(cf(0.0001))[:6],"0.0001")

	def test_str1(self):
		self.assertEqual(str(+cf(0.0001))[:6],"0.0001")

	def test_str2(self):
		self.assertEqual(str(cf(0000000000000000000000000000.0000001)), '9.9999999999999995474811182588e-8')

	def test_pow(self):
		self.assertEqual(pow(cf(2),2,3),1)

	def test_radians(self):
		self.assertEqual(radians(180.0),pi)

	def test_fsum(self):
		self.assertEqual(fsum([[]]),0.0)

	def test_powinf(self):
		self.assertEqual(pow(-float('inf'),0),1.0)

	def test_quotient(self):
		self.assertEqual(cf(5)//3,1)

	def test_rmod(self):
		self.assertEqual(1.0%cf(2),1)

	def test_divmod(self):
		self.assertEqual(divmod(cf(5),3)[0],1)

	def test_rdivmod(self):
		self.assertEqual(divmod(3,cf(5))[0],0)

	def test_long(self):
		self.assertEqual(long(cf(5)),5)

	def test_repr(self):
		self.assertEqual(repr(cf(5.8888888888888)),"cf(5;1,7,1,138232032760,1,1,1,3,82)")

	def test_repr1(self):
		self.assertEqual(repr(cf(5)),"cf(5)")

	def test_repr2(self):
		self.assertEqual(repr(cf(1./1000000)),'cf(0;1000000,22098525395,3,1,1,2,1,2,2,1,1,4,4,1,1,3,..)')

	def test_repr3(self):
		self.assertEqual(repr(cf(NaN)),"cf(NaN)")

	def test_bool(self):
		self.assertRaises(ValueError,bool,NaN)

	def test_set_cf_parameter(self):
		CF.set_cf_parameter("a",123)	
		self.assertEqual(CF.a,123)

if __name__ == '__main__':
	unittest.main()


#
#   HilbertCurve class
#
"""
This is a module to convert between one dimensional distance along a
`Hilbert curve`_, :math:`h`, and N-dimensional coordinates,
:math:`(x_0, x_1, ... x_N)`.  The two important parameters are :math:`N`
(the number of dimensions, must be > 0) and :math:`p` (the number of
iterations used in constructing the Hilbert curve, must be > 0).
We consider an N-dimensional `hypercube`_ of side length :math:`2^p`.
This hypercube contains :math:`2^{N p}` unit hypercubes (:math:`2^p` along
each dimension).  The number of unit hypercubes determine the possible
discrete distances along the Hilbert curve (indexed from :math:`0` to
:math:`2^{N p} - 1`).
"""

def _binary_repr(num, width):
	"""Return a binary string representation of `num` zero padded to `width`
	bits."""
	return format(num, 'b').zfill(width)


class HilbertCurve:
	# orientation default: 
	# ------
	#  1  4
	#  2  3
	# ------
	def __init__(self, p, n):
		"""Initialize a hilbert curve with,
		Args:
			p (int): iterations to use in the hilbert curve
			n (int): number of dimensions
		"""
		if p <= 0:
			raise ValueError('p must be > 0')
		if n <= 0:
			raise ValueError('n must be > 0')
		self.p = p
		self.n = n

		# D -> 2^p, size of each dimensions (in 2D: DxD square)
		self.D = 2**p
		# L -> D^2, number of pixels
		self.L = (self.D)**2

		# maximum distance along curve
		self.max_h = 2**(self.p * self.n) - 1

		# maximum coordinate value in any dimension
		self.max_x = 2**self.p - 1

	def _hilbert_integer_to_transpose(self, h):
		"""Store a hilbert integer (`h`) as its transpose (`x`).
		Args:
			h (int): integer distance along hilbert curve
		Returns:
			x (list): transpose of h
					  (n components with values between 0 and 2**p-1)
		"""
		h_bit_str = _binary_repr(h, self.p*self.n)
		x = [int(h_bit_str[i::self.n], 2) for i in range(self.n)]
		return x

	def _transpose_to_hilbert_integer(self, x):
		"""Restore a hilbert integer (`h`) from its transpose (`x`).
		Args:
			x (list): transpose of h
					  (n components with values between 0 and 2**p-1)
		Returns:
			h (int): integer distance along hilbert curve
		"""
		x_bit_str = [_binary_repr(x[i], self.p) for i in range(self.n)]
		h = int(''.join([y[i] for i in range(self.p) for y in x_bit_str]), 2)
		return h

	#
	# orientation default 0
	#
	#   = 0       = 1
	#  ______    ______
	# | 0  3 |  | 3  0 |
	# | 1  2 |  | 2  1 |
	#
	#   = 2       = 3
	#  ______    ______
	# | 3  2 |  | 0  1 |
	# | 0  1 |  | 3  2 |
	#
	#   = 4       = 5
	#  ______    ______
	# | 2  1 |  | 1  2 |
	# | 3  0 |  | 0  3 |
	#
	#   = 6       = 7
	#  ______    ______
	# | 1  0 |  | 2  3 |
	# | 2  3 |  | 1  0 |
	#
	def coordinates_from_distance(self, h, orientation = 0):
		"""Return the coordinates for a given hilbert distance.
		Args:
			h (int): integer distance along hilbert curve
			orientation: default 0
		Returns:
			x (list): list of coordinates in space n-D !!!attention to w and h position
			[0] -> w
			[1] -> h
		"""
		if orientation != 0 and self.n != 2:
			print('orientation change unsupported for n > 2')

		if h > self.max_h:
			raise ValueError('h={} is greater than 2**(p*N)-1={}'.format(h, self.max_h))
		if h < 0:
			raise ValueError('h={} but must be > 0'.format(h))

		x = self._hilbert_integer_to_transpose(h)
		Z = 2 << (self.p-1)

		# Gray decode by H ^ (H/2)
		t = x[self.n-1] >> 1
		for i in range(self.n-1, 0, -1):
			x[i] ^= x[i-1]
		x[0] ^= t

		# Undo excess work
		Q = 2
		while Q != Z:
			P = Q - 1
			for i in range(self.n-1, -1, -1):
				if x[i] & Q:
					# invert
					x[0] ^= P
				else:
					# exchange
					t = (x[0] ^ x[i]) & P
					x[0] ^= t
					x[i] ^= t
			Q <<= 1

		# changing curve orientation
		if orientation == 1:
			x[0] = self.D-1 - x[0] #reflect single axe
			return x
		elif orientation == 2:
			x[0] = self.D-1 - x[0] 
			return x[::-1] #reverse list (transpose)
		elif orientation == 3:
			return x[::-1]
		elif orientation == 4:
			x[0] = self.D-1 - x[0]
			x[1] = self.D-1 - x[1] 
			return x
		elif orientation == 5:
			x[1] = self.D-1 - x[1] 
			return x
		elif orientation == 6:
			x[1] = self.D-1 - x[1]
			return x[::-1]
		elif orientation == 7:
			x[0] = self.D-1 - x[0]
			x[1] = self.D-1 - x[1] 
			return x[::-1]
		else:
			return x

	def distance_from_coordinates(self, x_in, orientation = 0):
		"""Return the hilbert distance for a given set of coordinates.
		Args:
			x_in (list): transpose of h
						 (n components with values between 0 and 2**p-1)
						[0] -> h
						[1] -> w
		Returns:
			h (int): integer distance along hilbert curve
		"""
		# changing curve orientation
		if orientation == 1:
			x_in = x_in[::-1]
			x_in[0] = self.D-1 - x_in[0] #reflect single axe
		elif orientation == 2:
			x_in[0] = self.D-1 - x_in[0]
			x_in = x_in
		elif orientation == 3:
			x_in = x_in
		elif orientation == 4:
			x_in = x_in[::-1]
			x_in[0] = self.D-1 - x_in[0]
			x_in[1] = self.D-1 - x_in[1]
		elif orientation == 5:
			x_in = x_in[::-1]
			x_in[1] = self.D-1 - x_in[1]
		elif orientation == 6:
			x_in[1] = self.D-1 - x_in[1]
			x_in = x_in
		elif orientation == 7:
			x_in[0] = self.D-1 - x_in[0]
			x_in[1] = self.D-1 - x_in[1]
		else:
			x_in = x_in[::-1]

		x = list(x_in)
		if len(x) != self.n:
			raise ValueError('x={} must have N={} dimensions'.format(x, self.n))

		if any(elx > self.max_x for elx in x):
			raise ValueError(
				'invalid coordinate input x={}.  one or more dimensions have a '
				'value greater than 2**p-1={}'.format(x, self.max_x))

		if any(elx < 0 for elx in x):
			raise ValueError(
				'invalid coordinate input x={}.  one or more dimensions have a '
				'value less than 0'.format(x))

		M = 1 << (self.p - 1)

		# Inverse undo excess work
		Q = M
		while Q > 1:
			P = Q - 1
			for i in range(self.n):
				if x[i] & Q:
					x[0] ^= P
				else:
					t = (x[0] ^ x[i]) & P
					x[0] ^= t
					x[i] ^= t
			Q >>= 1

		# Gray encode
		for i in range(1, self.n):
			x[i] ^= x[i-1]
		t = 0
		Q = M
		while Q > 1:
			if x[self.n-1] & Q:
				t ^= Q - 1
			Q >>= 1
		for i in range(self.n):
			x[i] ^= t

		h = self._transpose_to_hilbert_integer(x)
		
		return h
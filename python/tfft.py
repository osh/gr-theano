#!/usr/bin/env python
#
# Copyright 2014 Tim O'Shea
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#
from gnuradio import gr, blocks, audio
import numpy, theano, time
import theano.tensor as T
from theano.tensor.fourier import Fourier, fft

class fft(gr.sync_block):
    def __init__(self, N):
        gr.sync_block.__init__(self,
            name="theano_fft",
            in_sig=[numpy.complex64],
            out_sig=[numpy.complex64])

        self.set_output_multiple(N);
        self.N = N
        x = T.cmatrix("x")
        w = theano.shared(numpy.ones(self.N, dtype="complex64"), name="w")
        self.f = theano.function(
            inputs=[x],
            outputs=[T.fourier.fft(x*w, n=N, axis=1)],
            updates={},
            name = "f")

    def work(self, input_items, output_items):
        n = len(input_items[0])/self.N
        for i in range(0,n):
            inmat = numpy.vstack([input_items[0][self.N*i:self.N*(i+1)]]);
            omat = self.f(inmat);
            output_items[0][self.N*i:self.N*(i+1)] = omat[0];
        return len(output_items[0])






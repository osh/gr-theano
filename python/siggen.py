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
import numpy, theano, time
import theano.tensor as T
from gnuradio import gr, blocks, audio
class siggen(gr.sync_block):
    N = 4096
    step = T.vector("step", dtype="complex64")
    l = T.iscalar("l")
    stepd = theano.shared(numpy.zeros(N, dtype=numpy.complex64), name="stepd")
    phase = theano.shared(numpy.asarray(1+0j, dtype=numpy.complex64), name="phase")
    oo = phase * stepd;
    oos = oo[0:l]
    set_step = theano.function(
            inputs=[step],
            outputs=[],
            updates={stepd:step},
            name="set_step")
    rval = theano.function(
            inputs=[l],
            outputs=[oos],
            updates={phase:oo[l-1]*stepd[1]},
            name="rval")

    def set_f(self, f):
        self.f = f;
        stepval = numpy.pi*2*self.f/self.fs;
        iv = numpy.exp(1j*numpy.arange(0,self.N,stepval, dtype=numpy.float32) , dtype=numpy.complex64);
        self.set_step( iv );

    def __init__(self, fs, f):
        gr.sync_block.__init__(self,
            name="theano_siggen",
            in_sig=[],
            out_sig=[numpy.complex64])
        self.fs = fs
        self.set_f(f);


    def work(self, input_items, output_items):
        out = output_items[0]
        out[:] = self.rval(len(output_items[0]))[0];
        return len(output_items[0])




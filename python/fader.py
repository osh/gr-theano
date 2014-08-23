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
import numpy, theano, time, random
import theano.tensor as T
from gnuradio import gr, blocks, audio
class fader(gr.sync_block):
    # some consts
    N = 4096
    NS = 8;

    step = T.matrix("step", dtype="complex64")
    l = T.iscalar("l")
    iv = T.cvector("iv")
    stepd = theano.shared(numpy.zeros((NS,N), dtype=numpy.complex64), name="stepd")
    phase = theano.shared(numpy.asarray([1+0j]*NS, dtype=numpy.complex64), name="phase")
    oo = theano.shared(numpy.asarray([1+0j]*N, dtype=numpy.complex64), name="oo")

    # theano functions
    set_step = theano.function(
            inputs=[step],
            outputs=[],
            updates={stepd:step},
            name="set_step")
    rval = theano.function(
            inputs=[iv,l],
            outputs=[iv*oo[0:l]],
            updates={phase:phase*stepd[:,l-1]*stepd[:,1],
                     oo:T.sum(T.Rebroadcast((1,True))(phase.dimshuffle(0,'x'))*stepd,axis=0)},
            name="rval")

    def set_f(self, f):
        print "set_f %f"%(f)
        self.f = f;
 
        tones = map(lambda x: random.uniform(1,100), range(0,self.NS));
        stepval = map(lambda x: numpy.pi*2.0*x/self.fs, tones);
        iv2 = numpy.vstack( map(lambda x: numpy.exp(1j*numpy.arange(0,self.N*x,x,dtype=numpy.float32), dtype=numpy.complex64), stepval));
        self.set_step( iv2 );

    def __init__(self, fs, f):
        gr.sync_block.__init__(self,
            name="theano_fader",
            in_sig=[numpy.complex64],
            out_sig=[numpy.complex64])
        self.fs = fs
        self.set_f(f);


    def work(self, input_items, output_items):
        out = output_items[0]
        o = self.rval(input_items[0], len(output_items[0]))
        out[:] = o[0];
        return len(output_items[0])




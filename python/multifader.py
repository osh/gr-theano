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
from theano.tensor.signal import conv

class multifader(gr.sync_block):
    # some consts
    N = 4096
    NS = 8;

    TL = 8;

    step = T.matrix("step", dtype="complex64")
    l = T.iscalar("l")      # length to produce
    iv = T.cvector("iv")    # complex input vector

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
            #outputs=[oo[0:l]],
            #outputs=[0*iv[0:l]+oo[0:l]],

            
            outputs=[T.signal.conv.conv2d(iv[0:(l+TL-1)].dimshuffle('x',0), oo[0:TL].dimshuffle('x',0))],

            #outputs=[T.signal.conv.conv2d(iv[0:(l+TL-1)].dimshuffle('x',0), oo[0:TL])],
            #outputs=[0*iv[0:l]+oo[0:l]],
            #outputs=[iv[0:l]*oo[0:l]],
            #updates={phase:phase*stepd[:,l-1]*stepd[:,0],
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
            name="theano_multifader",
            in_sig=[numpy.complex64],
            out_sig=[numpy.complex64])
        self.fs = fs
        self.set_f(f);
        self.set_history(self.TL);


    def work(self, input_items, output_items):
        out = output_items[0]
        print "in len: %d"%len(input_items[0])
        print "out len: %d"%len(output_items[0])
        nout = min( len(output_items[0]), len(input_items[0])-(self.TL-1))
        o = self.rval(input_items[0], nout)
        out[:] = o[0];
        return len(output_items[0])




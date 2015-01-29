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
from theano.tensor.basic import choose

class multifader(gr.sync_block):
    # some consts
    N = 4096
    NS = 8;

    TL = 16;     # Tap length of filter
    NF = TL;    # number of fading components (equal to n taps for prototype version)

    step = T.ctensor3("step")
    l = T.iscalar("l")      # length to produce
    iv = T.cmatrix("iv")    # complex input vector

    stepd = theano.shared(numpy.zeros((NF,NS,N), dtype=numpy.complex64), name="stepd")
    taps =  theano.shared(numpy.ones((TL), dtype=numpy.complex64), name="taps");
    phase = theano.shared(numpy.asarray([[1+0j]*NS]*NF, dtype=numpy.complex64), name="phase")
    oo = theano.shared(numpy.asarray([[1+0j]*N]*NF, dtype=numpy.complex64), name="oo")

    # theano functions
    set_step = theano.function(
            inputs=[step],
            outputs=[],
            updates={stepd:step},
            name="set_step")
    rval = theano.function(
            on_unused_input="warn",
            inputs=[iv,l],
            outputs=[T.signal.conv.conv2d(iv[0:(l+TL-1)], taps.dimshuffle('x',0))],
            updates={phase:phase*stepd[:,:,l-1]*stepd[:,:,1],
                    taps: phase[:,0]},
            name="rval")

    def set_f(self, f):
        self.f = f;
 
        # initialize empty step update tensor (Num fading components, num sinusoids per fading compnent, num steps per sine)
        step_update = numpy.zeros( (self.NF, self.NS, self.N), dtype="complex64");
        
        # for each fading component
        for i in range(0,self.NF):      
            # generate random initial frequency components
            # todo: 100 is hard wired max component doppler freq? make changeable parameter?
            tones = map(lambda x: random.uniform(1,100), range(0,self.NS));
            # generate phasor step ammount for each component frequency
            stepval = map(lambda x: numpy.pi*2.0*x/self.fs, tones);
            # update step update tensor sub-matrix
            step_update[i,:,:] = numpy.vstack( map(lambda x: numpy.exp(1j*numpy.arange(0,self.N*x,x,dtype=numpy.float32), dtype=numpy.complex64), stepval));
        # pass to theano variable
        self.set_step( step_update );
                

    def __init__(self, fs, f, blocksize=1024):
        gr.sync_block.__init__(self,
            name="theano_multifader",
            in_sig=[numpy.complex64],
            out_sig=[numpy.complex64])
        self.fs = fs
        self.set_f(f);
        self.set_history(self.TL);
        self.set_output_multiple(blocksize)
        self.blocksize = blocksize

    def work(self, input_items, output_items):
        out = output_items[0]
        nout = min( len(output_items[0]), len(input_items[0])-(self.TL-1))
        nout = self.blocksize;
        o = self.rval(input_items, nout)
        out[:] = o[0];
        return nout;


if __name__ == "__main__":
    a = multifader(44100.0, 400.0, blocksize=1024)

    l = 2048
    TL = 16
    inv = [numpy.arange(0,l+TL-1,1, dtype=numpy.complex64)]
    print "in vector"
    print inv
    otv = [numpy.zeros((l), dtype=numpy.complex64)]
    a.work(inv, otv)
    print "out vector"
    print otv

    print "ok"

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


    offset_matrix = theano.shared(numpy.zeros((NF, N), dtype=numpy.int32))

#    step = T.matrix("step", dtype="complex64")
    offsets = T.imatrix("offsets");
    step = T.ctensor3("step")
    l = T.iscalar("l")      # length to produce
    iv = T.cmatrix("iv")    # complex input vector
    #iv = T.cvector("iv")    # complex input vector

    stepd = theano.shared(numpy.zeros((NF,NS,N), dtype=numpy.complex64), name="stepd")

    taps =  theano.shared(numpy.ones((TL,N+TL-1), dtype=numpy.complex64), name="taps");
    
    phase = theano.shared(numpy.asarray([1+0j]*NS, dtype=numpy.complex64), name="phase")
    oo = theano.shared(numpy.asarray([1+0j]*N, dtype=numpy.complex64), name="oo")

    # theano functions
    set_offset = theano.function(
            inputs=[offsets],
            outputs=[],
            updates={offset_matrix:offsets},
            name="set_offset")
    set_step = theano.function(
            inputs=[step],
            outputs=[],
            updates={stepd:step},
            name="set_step")
    rval = theano.function(
            on_unused_input="warn",
            inputs=[iv,l],
            #outputs=[T.signal.conv.conv2d(iv[0:(l+TL-1)].dimshuffle('x',0), taps)],
            outputs=[ T.sum( T.basic.choose(offset_matrix[0:1,:l], iv[:,:l]), 0) ],
            #outputs=[ T.sum( T.basic.choose(offset_matrix[0:1,:l], iv[:,:l]), 0) ],
            #outputs=[ T.sum( T.basic.choose(offset_matrix[0:1,:l], iv[:,:l]), 0) ],
            #outputs=[ T.sum( T.basic.choose(offset_matrix[0:1,:l], iv[:,:l]), 0) ],
            #outputs=[ T.sum( iv[0:(l+TL-1)].dimshuffle('x',0), 0)[0:l]  ],
            #outputs=[ T.sum( T.basic.choose(offset_matrix[:,0:l], iv[:,0:l]), 0) ],

            #outputs=[ T.sum( T.basic.choose(offset_matrix[:,0:l], iv[:,0:l], mode='clip'), 0) ],
            #outputs=[ T.sum( T.basic.choose(offset_matrix[:,0:l], iv[:, l].dimshuffle(0,'x') , mode='clip'), 0) ],

            #outputs=[ T.sum( T.basic.choose(offset_matrix[:,0:l].T, iv[:, l].dimshuffle(0,'x') , mode='clip'), 1) ],
            ####outputs=[ T.basic.choose(offset_matrix[:,0:l], iv[:, l].T ) ],
            #outputs=[ T.basic.choose(offset_matrix[:,0:l], iv[0:l].dimshuffle('x',0) ) ],
            #outputs=[ T.basic.choose(offset_matrix[:,0:l], iv[0:l].dimshuffle('x',0) ) ],

            #outputs=[ T.sum( T.basic.choose(offset_matrix[:,0:l], iv) * taps[:,0:l]), 0 ],
            #outputs=[ T.sum( iv[0:(l+TL-1)].dimshuffle('x',0) * taps[:,0:(l+TL-1)], 0)[0:l]  ],
#oo[0:TL].dimshuffle('x',0))],
#            updates={phase:phase*stepd[:,l-1]*stepd[:,1],
#                     oo:T.sum(T.Rebroadcast((1,True))(phase.dimshuffle(0,'x'))*stepd,axis=0)},
            name="rval")

#    rval = theano.function(
#            inputs=[iv,l],
#            #outputs=[oo[0:l]],
#            #outputs=[0*iv[0:l]+oo[0:l]],
#
#            
#            outputs=[T.signal.conv.conv2d(iv[0:(l+TL-1)].dimshuffle('x',0), oo[0:TL].dimshuffle('x',0))],
#
#            #outputs=[T.signal.conv.conv2d(iv[0:(l+TL-1)].dimshuffle('x',0), oo[0:TL])],
#            #outputs=[0*iv[0:l]+oo[0:l]],
#            #outputs=[iv[0:l]*oo[0:l]],
#            #updates={phase:phase*stepd[:,l-1]*stepd[:,0],
#            updates={phase:phase*stepd[:,l-1]*stepd[:,1],
#                     oo:T.sum(T.Rebroadcast((1,True))(phase.dimshuffle(0,'x'))*stepd,axis=0)},
#            name="rval")

    def set_f(self, f):
        print "set_f %f"%(f)
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
                

    def __init__(self, fs, f):
        gr.sync_block.__init__(self,
            name="theano_multifader",
            in_sig=[numpy.complex64],
            out_sig=[numpy.complex64])
        self.fs = fs
        self.set_f(f);
        self.set_history(self.TL);

        offsets = numpy.zeros((self.NF, self.N), numpy.int32)
        for i in range(0,self.NF):
            for j in range(0,self.N):
                #offsets[i,j] = i+j
                #offsets[i,j] = int(min(i+j, 1))
                offsets[i,j] = 1
        self.set_offset(offsets);
        print offsets
        print self.offset_matrix
    
    def work(self, input_items, output_items):
        out = output_items[0]
        #print "in len: %d"%len(input_items[0])
        #print "out len: %d"%len(output_items[0])
        nout = min( len(output_items[0]), len(input_items[0])-(self.TL-1))
#        print nout
        print "l = %d"%(nout)
        o = self.rval(input_items, nout)
        #o = self.rval(input_items[0], nout)
#        print len(o[0])
        out[:] = o[0];
        return len(output_items[0])


if __name__ == "__main__":
    a = multifader(44100.0, 400.0)

    l = 100
    TL = 16
    inv = [numpy.arange(0,l+TL-1,1, dtype=numpy.complex64)]
    print "in vector"
    print inv
    otv = [numpy.zeros((l), dtype=numpy.complex64)]
    a.work(inv, otv)
    print "out vector"
    print otv

    print "ok"

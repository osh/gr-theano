#!/usr/bin/env python
import numpy, theano, time, random
import theano.tensor as T
from theano.tensor.signal import conv

# GNU Radio block
import numpy
from gnuradio import gr, blocks, audio
class fir(gr.sync_block):
    # some consts
    x = T.matrix("x")
    y = T.matrix("y")

    def set_taps(self, taps):
        print "set_taps"
        self.b = theano.shared(numpy.vstack([taps]), name="b")
        self.set_history(taps.size)

    def __init__(self, taps):
        gr.sync_block.__init__(self,
            name="theano_fir",
            in_sig=[numpy.float32],
            out_sig=[numpy.float32])
        self.set_taps(taps)

        self.f = theano.function(
            inputs = [self.x],
            outputs=[T.signal.conv.conv2d(self.x,self.b)],
            updates={},
            name ="f")



    def work(self, input_items, output_items):
        out = output_items[0]
        o = self.f( numpy.vstack([ input_items[0] ]) );
        out[:] = o[0][0,:];
        return len(output_items[0])




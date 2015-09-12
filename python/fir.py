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
from theano.tensor.signal import conv
from gnuradio import gr, blocks, audio
class fir(gr.sync_block):
    # some consts
    x = T.fmatrix("x")
    y = T.fmatrix("y")

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




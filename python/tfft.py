#!/usr/bin/env python
from gnuradio import gr, blocks, audio
import numpy, theano, time
import theano.tensor as T

class fft(gr.sync_block):
    def __init__(self, N):
        gr.sync_block.__init__(self,
            name="theano_fft",
            in_sig=[numpy.complex64],
            out_sig=[numpy.complex64])

        self.set_output_multiple(N);
        self.N = N
        import theano.tensor as T
        import theano
        x = T.matrix("x")
        w = theano.shared(numpy.ones(self.N, dtype=theano.config.floatX), name="w")
        self.f = theano.function(
            inputs=[x],
            outputs=[T.fourier.fft(x*w, n=N, axis=1)],
            #outputs=[theano.tensor.fourier.fft(x*w, n=N, axis=1)],
            updates={},
            name = "f")

    def work(self, input_items, output_items):
        n = len(input_items[0])/N
        for i in range(0,n):
            output_items[0][N*i:N*(i+1)-1] = self.f(input_items[0][N*i:N*(i+1)-1]);
        return len(output_items[0])






#!/usr/bin/env python
from gnuradio import gr, blocks, audio
import numpy, theano, time
import theano.tensor as T
from theano.tensor.fourier import Fourier, fft

ftmp = T.fourier;

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
        #w = theano.shared(numpy.ones(self.N, dtype=theano.config.floatX), name="w")
        self.f = theano.function(
            inputs=[x],
            outputs=[T.fourier.fft(x*w, n=N, axis=1)],
            #outputs=[theano.tensor.fourier.fft(x*w, n=N, axis=1)],
            updates={},
            name = "f")

    def work(self, input_items, output_items):
        n = len(input_items[0])/self.N
        for i in range(0,n):

            inmat = numpy.vstack([input_items[0][self.N*i:self.N*(i+1)]]);
            omat = self.f(inmat);
            output_items[0][self.N*i:self.N*(i+1)] = omat[0];
#            print omat.shape;
            #output_items[0][self.N*i:self.N*(i+1)-1] = self.f(numpy.vstack([input_items[0][self.N*i:self.N*(i+1)-1]]));
            #output_items[0][self.N*i:self.N*(i+1)-1] = self.f(numpy.vstack([input_items[0][self.N*i:self.N*(i+1)-1]]));
        return len(output_items[0])






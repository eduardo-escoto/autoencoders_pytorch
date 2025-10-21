from torch.nn import Module

# Pretty basic at the moment

class Autoencoder(Module):
    def __init__(self, encoder, decoder, bottleneck):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bottleneck = bottleneck

    def encode(self, x):
        return self.encoder.encode(x)

    def decode(self, x):
        return self.decoder.decode(x)

    def forward(self, x):
        z_e = self.encode(x)
        x_d = self.decode(z_e)

        return x_d

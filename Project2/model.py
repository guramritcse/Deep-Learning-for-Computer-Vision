import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.encoder(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels//2, kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.decoder(x)

class AutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AutoEncoder, self).__init__()
        self.encoder1 = EncoderBlock(in_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(384, 128)
        self.decoder2 = DecoderBlock(192, 64)
        self.decoder1 = DecoderBlock(96, 32)
        
        self.final_layer = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        
        # Decoder with skip connections
        dec4 = self.decoder4(enc4)
        dec4 = torch.cat([enc3, dec4], dim=1)
        dec3 = self.decoder3(dec4)
        dec3 = torch.cat([enc2, dec3], dim=1)
        dec2 = self.decoder2(dec3)
        dec2 = torch.cat([enc1, dec2], dim=1)
        dec1 = self.decoder1(dec2)
        
        # Final layer
        output = self.final_layer(dec1)
        return output

if __name__ == '__main__':
    model = AutoEncoder(in_channels=3, out_channels=3)
    print(model)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as T

class Encoder(nn.Module):
    def __init__(self, encoder_layers, latent_dim):
        super().__init__()
        modules = []
        for i in range(len(encoder_layers) - 1):
            modules.append(nn.Linear(encoder_layers[i], encoder_layers[i+1]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(encoder_layers[-1], latent_dim))
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, decoder_layers, latent_dim):
        super().__init__()
        modules = []
        modules.append(nn.Linear(latent_dim, decoder_layers[0]))
        for i in range(len(decoder_layers) - 1):
            modules.append(nn.ReLU())
            modules.append(nn.Linear(decoder_layers[i], decoder_layers[i+1]))
        modules.append(nn.Sigmoid())
        self.model = nn.Sequential(*modules)

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, discriminator_layers, latent_dim):
        super().__init__()
        modules = []
        modules.append(nn.Linear(latent_dim, discriminator_layers[0]))
        modules.append(nn.ReLU())
        for i in range(len(discriminator_layers) - 2):
            modules.append(nn.Linear(discriminator_layers[i], discriminator_layers[i+1]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(discriminator_layers[-1], 1))
        modules.append(nn.Sigmoid())
        self.model = nn.Sequential(*modules)

    def forward(self, z):
        return self.model(z)

class LatentDiffusionUNet(nn.Module):
    def __init__(self, diffusion_layers, latent_dim):
        super().__init__()
        modules = []
        modules.append(nn.Linear(latent_dim, diffusion_layers[0]))
        modules.append(nn.ReLU())
        for i in range(len(diffusion_layers) - 2):
            modules.append(nn.Linear(diffusion_layers[i], diffusion_layers[i+1]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(diffusion_layers[-1], latent_dim))
        self.model = nn.Sequential(*modules)

    def forward(self, z):
        return self.model(z)

class LatentDenoisingAAE:
    def __init__(self, latent_dim, encoder_layers, decoder_layers, discriminator_layers, diffusion_layers):
        """
        Initializes the structure of the Latent Denoising Adversarial Autoencoder

        Parameters
        ----------
        latent_dim : Size of the latent dimension i.e. length of the encoded vector
        encoder_layers : A list containing the numbers in the input layer and hidden layer(s) of the Encoder
        decoder_layers : A list containing the numbers in the hidden layer(s) and output layer of the Decoder
        discriminator_layers : A list containing the numbers in the hidden layer(s) of the Discriminator
        diffusion_layers : A list containing the numbers in the hidden layer(s) of the Diffusion/Denoising Layer
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.encoder = Encoder(encoder_layers, latent_dim).to(self.device)
        self.decoder = Decoder(decoder_layers, latent_dim).to(self.device)
        self.discriminator = Discriminator(discriminator_layers, latent_dim).to(self.device)
        self.diffusion = LatentDiffusionUNet(diffusion_layers, latent_dim).to(self.device)
    
    def preprocess_for_fid(self, x):
        """
        Preprocesses the dataset to work with torchmetrics' FrechetInceptionDistance

        Parameters
        ----------
        x : PyTorch Tensor

        Returns
        -------
        Pre-processed Tensor ready to be passed to the FrechetInceptionDistance module
        """
        if x.dtype != torch.float32:
            x = x.float() / 255.0
        x = x.clamp(0, 1)
        x = T.Resize((299, 299))(x)
        x = x.expand(-1, 3, -1, -1)
        x = (x * 255).round().to(torch.uint8)
        return x
    
    def train(self, data_loader, num_epochs):
        """
        Trains the Latent Denoising Adversarial Autoencoder

        Parameters
        ----------
        data_loader : DataLoader used to fetch the training data
        num_epochs : Number of epochs for the training loop
        """
        opt_enc_dec = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=1e-3)
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3)
        opt_diff = torch.optim.Adam(self.diffusion.parameters(), lr=1e-3)

        # Training Loop
        for epoch in range(num_epochs):
            for x, _ in data_loader:
                x = x.to(self.device)
                x = x.view(x.size(0), -1)

                # Reconstruction
                z = self.encoder(x)
                x_recon = self.decoder(z)
                recon_loss = F.l1_loss(x_recon, x)

                # Adversarial
                z_real = torch.randn(x.size(0), self.latent_dim).to(self.device)
                d_real = self.discriminator(z_real)
                d_fake = self.discriminator(z.detach())
                d_loss = -torch.mean(torch.log(d_real + 1e-6) + torch.log(1 - d_fake + 1e-6))

                opt_disc.zero_grad()
                d_loss.backward()
                opt_disc.step()

                d_fake = self.discriminator(z)
                g_loss = -torch.mean(torch.log(d_fake + 1e-6))

                total_loss = recon_loss + g_loss
                opt_enc_dec.zero_grad()
                total_loss.backward()
                opt_enc_dec.step()

                # Diffusion
                z_noisy = z.detach() + 0.1 * torch.randn_like(z)
                z_denoised = self.diffusion(z_noisy)
                diffusion_loss = F.mse_loss(z_denoised, z.detach())
                opt_diff.zero_grad()
                diffusion_loss.backward()
                opt_diff.step()

            print(f"Epoch {epoch+1}, Recon: {recon_loss.item():.4f}, D: {d_loss.item():.4f}, G: {g_loss.item():.4f}, Diff: {diffusion_loss.item():.4f}")
    
    def generate_samples(self, num_samples):
        """
        Generates samples using the trained Latent Denoising AAE
        The train() method must be called before generate_samples()

        Parameters
        ----------
        num_samples : Number of samples to be generated

        Returns
        -------
        A list of samples generated by the Latent Denoising AAE
        """
        self.encoder.eval()
        self.decoder.eval()
        self.diffusion.eval()

        z_samples = torch.randn(num_samples, self.latent_dim).to(self.device)
        z_hybrid = self.diffusion(z_samples)
        x_hybrid = self.decoder(z_hybrid).view(-1, 1, 28, 28)
        x_hybrid = self.preprocess_for_fid(x_hybrid)

        return x_hybrid

    def get_fid_score(self, test_loader, fid_batch_size):
        """
        Generates a set of samples and computes the Frechet Inception Distance (FID) 
        between the generated samples and a set of test (real) samples

        Parameters
        ----------
        test_loader : DataLoader used to fetch the testing data
        fid_batch_size : Size of each batch of samples processed by FID

        Returns
        -------
        The FID Score between the generated samples and testing data
        """
        real_imgs, _ = next(iter(test_loader))
        real_imgs = real_imgs.to(self.device)
        real_imgs = self.preprocess_for_fid(real_imgs)
        fid_hybrid = FrechetInceptionDistance(feature=64).to(self.device)
        x_hybrid = self.generate_samples(real_imgs.size(0))

        for i in range(0, real_imgs.size(0), fid_batch_size):
            real_batch = real_imgs[i:i+fid_batch_size].to(self.device)
            hybrid_batch = x_hybrid[i:i+fid_batch_size].to(self.device)

            fid_hybrid.update(real_batch, real=True)
            fid_hybrid.update(hybrid_batch, real=False)
        
        fid_hybrid_score = fid_hybrid.compute()

        return fid_hybrid_score
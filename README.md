# Latent Denoising Adversarial Autoencoder

## How to run the experiment used in the paper
1. Open the `Latent Denoising AAE vs Regular AAE.ipynb` locally or in Google Colab
2. If using Google Colab, make sure the runtime type is set to T4 GPU
3. Run each cell sequentially

## How to use the Latent Denoising AAE Module
1. Include `from LatentDenoisingAAE import LatentDenoisingAAE` in the beginning of your program
2. Initialize `LatentDenoisingAAE` with the following parameters:
    * `latent_dim` : Size of the latent dimension i.e. length of the encoded vector
    * `encoder_layers` : A list containing the numbers in the input layer and hidden layer(s) of the Encoder
    * `decoder_layers` : A list containing the numbers in the hidden layer(s) and output layer of the Decoder
    * `discriminator_layers` : A list containing the numbers in the hidden layer(s) of the Discriminator
    * `diffusion_layers` : A list containing the numbers in the hidden layer(s) of the Diffusion/Denoising Layer
3. Train the model using the `train()` method and supply the following parameters:
    * `data_loader` : DataLoader used to fetch the training data
    * `num_epochs` : Number of epochs for the training loop
4. Generate samples using the `generate_samples(num_samples)` where `num_samples` is the number of samples to be generated
5. Get the Frechet Inception Distance (FID) Score between generated samples and real (test) samples using the `get_fid_score()` method and supply the following parameters:
    * `test_loader` : DataLoader used to fetch the testing data
    * `fid_batch_size` : Size of each batch of samples processed by FID
6. Reference the LDAEE_on_MNIST.py for an example on how to use the module on the MNIST dataset

## File Explanation
* `LatentDenoisingAAE.py` - Contains the Latent Denoising Adversarial Autoencoder (`LatentDenoisingAAE`) module
* `LDAAE_on_MNIST.py` - Shows how to use the `LatentDenoisingAAE` module on the MNIST dataset
* `Latent Denoising AAE vs Regular AAE.ipynb` - Runs the experiments that compare the Latent Denoising AAE to the Regular AAE and produces the results used in the paper
* `Latent Denoising AAE vs Regular AAE.pdf` - A paper/report explaining the Motivation behind the Latent Denoising AAE, along with detailing the experiments and results used to assess the performance/benefits of the novel method

## Dependencies
torch 2.6.0

torchvision 0.21.0

matplotlib 3.10.0

torchmetrics 1.7.1

numpy 2.0.2
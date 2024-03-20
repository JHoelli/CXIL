import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


class Discriminator(nn.Module):
  def __init__(self, channels_img, features_d):
    super(Discriminator, self).__init__()
    self.disc = nn.Sequential(
        #Input: N x channels_img x 32 x 32
        nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1), #16x16
        nn.LeakyReLU(0.2),
        self._block(features_d, features_d*2, 4, 2, 1), #8x8
        self._block(features_d*2, features_d*4, 4, 2, 1), #8x8
        self._block(features_d*4, features_d*8, 4, 2, 1),  #4x4
        nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0), # 1x1 representing probability TODO Kernal Size Used to be 4
    )

  def _block(self, in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.LeakyReLU(0.2)
    )

  def forward(self, x):
    return self.disc(x)



class Generator(nn.Module):
  def __init__(self, z_dim, channels_img, features_g):
    super(Generator, self).__init__()
    self.gen = nn.Sequential(
        #Input: N x z_dim x 1 x 1
        self._block(z_dim, features_g*16, 4, 1, 0), # img 4 x 4
        self._block(features_g*16, features_g*8, 4, 2, 1), # img 8x8
        self._block(features_g*8, features_g*4, 4, 2, 1), # img 16x16
        self._block(features_g*4, features_g*2, 4, 2, 1 ), #img 32x32
        nn.ConvTranspose2d(features_g*2, channels_img, kernel_size=4, stride=2, padding=1), #N x channels_img x 64 x 64
        nn.Tanh() #model aligns with normalzed images between -1 and 1
    )
  
  def _block(self, in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
  
  def forward(self, x):
    return self.gen(x)
  

def initialize_weights(model):
  for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
      nn.init.normal(m.weight.data, 0.0, 0.02)



def train_VAE(loader,loaded_critic,loaded_gen,Z_DIM = 100,lr=1e-4, num_epochs=10,critic_iterations=5, lambda_gp=10, name='temp', mode='cnn'):
    opt_critic = optim.Adam(loaded_critic.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_gen = optim.Adam(loaded_gen.parameters(), lr=lr, betas=(0.0, 0.9))

    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(loader):
            real = real#.to(device)
            new_batch_size = real.shape[0]

            # Training critic
            for _ in range(critic_iterations):
                if mode=='cnn':
                  noise = torch.randn(new_batch_size, Z_DIM, 1, 1)
                else:
                  noise = torch.randn(new_batch_size, Z_DIM)#, 1, 1)#.to(device)
                fake = loaded_gen(noise)
                #print(real.shape)
                #print(fake.shape)
                critic_real = loaded_critic(real).reshape(-1)
                critic_fake = loaded_critic(fake).reshape(-1)

                gp = gradient_penalty(loaded_critic, real, fake,mode,device='cpu')
                loss_critic = (-(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * gp)

                loaded_critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()


            # Training Generator min -D(G(z))
            output = loaded_critic(fake).reshape(-1)
            loss_gen = -(torch.mean(output))
            loaded_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()


            if batch_idx % 100 == 0:
                print(f"Epoch[{epoch+3}/{num_epochs}]  Batch[{batch_idx}/937] \n"F"Loss D: {loss_critic:.4f}, Loss G: {loss_gen:.4f}")

        #save_noise = torch.randn(batch_size, Z_DIM, 1, 1)#.to(device)
        #sample_8x8 = loaded_gen(save_noise).reshape(-1,1,28,28)
        #filepath = (f"./{epoch+20}.png")
        torch.save(loaded_gen.state_dict(), "./CXIL/Learning/models/MNIST/gen.pth")
        torch.save(loaded_critic.state_dict(), "./CXIL/Learning/models/MNIST/critic.pth" )



def gradient_penalty(critic, real, fake,mode, device="cpu"):
  #TODO is THIS CORREXT ? 
  if mode =='cnn':
    batch_size, c, H, W = real.shape
    epsilon = torch.rand((batch_size, c, 1, 1)).repeat(1, c, H, W).to(device)
  else: 
    batch_size, c = real.shape
    epsilon = torch.rand((batch_size, c))#.repeat(1, c, H, W).to(device)
     
  interpolated_images = (real * epsilon) + fake * (1 - epsilon)

  mixed_scores = critic(interpolated_images)
  gradient = torch.autograd.grad(
      inputs = interpolated_images,
      outputs = mixed_scores,
      grad_outputs = torch.ones_like(mixed_scores),
      create_graph = True,
      retain_graph = True
  )[0]

  gradient = gradient.view(gradient.shape[0], -1)
  gradient_norm = gradient.norm(2, dim=1)
  gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
  return gradient_penalty

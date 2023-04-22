import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt


#CIFAR10 Dataset
data_set = datasets.CIFAR10(root="D:\cifar-10-batches-py", download=False, transform=transforms.Compose([
    transforms.Resize(64),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))
data_loader = torch.utils.data.DataLoader(data_set, batch_size = 128, shuffle=True, num_workers=2)

#parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# noise_input dimension
noise_dimension = 100 
# real images label
real_value = 1 
# fake images label
fake_value = 0 

# Set torch random seed
rand_seed = random.randint(10, 100000)
random.seed(rand_seed)
torch.manual_seed(rand_seed)


# initializing the weights to discriminator and generator
def weights_initialize(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


#Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, pass_ip):
        disc_op = self.main(pass_ip)
        return disc_op.view(-1, 1).squeeze(1)
    

#Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(noise_dimension, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, pass_ip):
        gen_op = self.main(pass_ip)
        return gen_op
    

# Creating Generator and Discriminator and apply initial weights
discriminator = Discriminator().to(device)
generator = Generator().to(device)
discriminator.apply(weights_initialize)
generator.apply(weights_initialize)


# setup optimizer for Generator and Discriminator
optim_gen = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optim_disc = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# More parameters for training and storing values
noise = torch.randn(128, noise_dimension, 1, 1, device=device)
gen_list_LOSS = []
disc_list_LOSS = []
counter = 0
counter_list = []


# Training for Discriminator and Generator
no_epochs = 10
for epoch in range(no_epochs):
    for i, data in enumerate(data_loader, 0):
        counter += 1
        counter_list.append(counter)
        
        # Updating discriminator with real images
        real_data = data[0].to(device)
        size_batch = real_data.size(0)
        labels_tensor = torch.full((size_batch,), real_value, device = device).float()
        discriminator.zero_grad()
        disc_op = discriminator(real_data ).float()
        dis_real_error = criterion(disc_op, labels_tensor)
        dis_real_error.backward()
        dis_real_output_mean = disc_op.mean().item()

        # Updating discriminator with fake images
        labels_tensor.fill_(fake_value).float()
        noise = torch.randn(size_batch, noise_dimension, 1, 1, device=device)
        fake_data = generator(noise)
        disc_op = discriminator(fake_data.detach()).float()
        dis_fake_error = criterion(disc_op, labels_tensor)
        dis_fake_error.backward()
        dis_fake_output_mean = disc_op.mean().item()
        optim_disc.step()
        disc_final_err = dis_real_error + dis_fake_error
        disc_list_LOSS.append(disc_final_err.item())
        

        # Update Generator 
        labels_tensor.fill_(real_value).float()
        generator.zero_grad()
        gen_op = discriminator(fake_data).float()
        gen_err = criterion(gen_op, labels_tensor)
        gen_list_LOSS.append(gen_err.item())
        gen_err.backward()
        gen_op_mean = gen_op.mean().item()
        optim_gen.step()

        
        print('[%d/%d][%d/%d] DiscLoss: %.4f GenLoss: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % 
              (epoch, no_epochs, i, len(data_loader), disc_final_err.item(), 
               gen_err.item(), dis_real_output_mean, dis_fake_output_mean, gen_op_mean ))
        
    #Store images 
    fake_data = generator(noise)
    vutils.save_image(real_data,'DCganOutput/real_samples.png',normalize=True)
    vutils.save_image(fake_data.detach(),'DCganOutput/fake_samples_for_%03d_epoch.png' % (epoch), normalize=True)


#loss of the generator and the descriminator
plt.plot(counter_list, gen_list_LOSS, 'r.', label='Generator')
plt.plot(counter_list, disc_list_LOSS, 'g.', label='Discriminator')
plt.title("DCGAN Loss of Generator and Discriminator ")
plt.xlabel("Batch Number")
plt.ylabel("Binary Cross Entropy Loss")
plt.legend(loc="best")
plt.show()
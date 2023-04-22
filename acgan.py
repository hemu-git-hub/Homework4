import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt


# Get CIFAR10 Dataset
data_set = datasets.CIFAR10(root="D:\cifar-10-batches-py", download=False, transform=transforms.Compose(
    [transforms.Resize(64),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))
data_loader = torch.utils.data.DataLoader(data_set, batch_size = 128, shuffle=True, num_workers=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        
        self.main = nn.Sequential(
            nn.Conv2d(3,64,4,2,1,bias = False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(64,128,4,2,1,bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(128,256,4,2,1,bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(256,512,4,2,1,bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,True)
        )
        
        self.verify = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, 0, bias = False), 
            nn.Sigmoid()
        )
        
        self.labels = nn.Sequential(
            nn.Conv2d(512, 11, 4, 1, 0, bias = False), 
            nn.LogSoftmax(dim = 1)
        )
        
    def forward(self, passed_input):
        passed_input = self.main(passed_input)
        validity = self.verify(passed_input)
        op_label = self.labels(passed_input)
        
        # resize
        validity = validity.view(-1)
        op_label = op_label.view(-1,11)
        return validity, op_label

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        
        self.emb = nn.Embedding(10,100)
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100,512,4,1,0,bias = False),
            nn.ReLU(True),
            nn.ConvTranspose2d(512,256,4,2,1,bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256,128,4,2,1,bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,64,4,2,1,bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64,3,4,2,1,bias = False),
            nn.Tanh()
        )
        
    def forward(self, noise, inputLabels):
        embLabels = self.emb(inputLabels)
        temp = torch.mul(noise, embLabels)
        temp = temp.view(-1, 100, 1, 1)
        return self.main(temp)
    

# Create Generator and Discriminator and apply initial weights
discriminator = Discriminator().to(device)
generator = Generator().to(device)
discriminator.apply(weights_init)
generator.apply(weights_init)


# Setup optimizers 
dis_optim = optim.Adam(discriminator.parameters(), 0.0002, betas = (0.5,0.999))
gen_optim = optim.Adam(generator.parameters(), 0.0002, betas = (0.5,0.999))
criterion = nn.BCELoss()

# Parameters for training
num_epochs = 10
real_labels_tensor = 0.7 + 0.5 * torch.rand(10, device = device)
fake_labels_tensor = 0.3 * torch.rand(10, device = device)

# Variables to track training progress
counter_list = []
counter = 0
loss_list_gen = []
loss_list_disc = []


# Training algorithm for discriminator and generator
for epoch in range(0, num_epochs):
    
    # Iterate through all batches
    for index, (data, image_labels) in enumerate(data_loader, 0):
        counter += 1
        counter_list.append(counter)
        
        # make data avaialbe for cuda
        data = data.to(device)
        image_labels = image_labels.to(device)
        size_batch = data.size(0)
        labels_real = real_labels_tensor[index % 10]
        fake_labels = fake_labels_tensor[index % 10]
        class_fake_labels = 10 * torch.ones((size_batch, ), dtype = torch.long, device = device)
        
        # Periodically switch labels
        if index % 25 == 0:
            temp = labels_real
            labels_real = fake_labels
            fake_labels = temp
        
        # Train Discriminator with real data
        labels_for_validate = torch.full((size_batch, ), labels_real, device = device)
        dis_optim.zero_grad() 
        validity, op_label = discriminator(data)       
        disc_realvalidation_err = criterion(validity, labels_for_validate)            
        disc_reallabel_err = F.nll_loss(op_label, image_labels)
        disc_error_real = disc_realvalidation_err + disc_reallabel_err
        disc_error_real.backward()
        valid_mean1 = validity.mean().item()        
        
        # Train Discriminator with fake data
        disc_label_fake = torch.randint(0, 10, (size_batch, ), dtype = torch.long, device = device)
        noise = torch.randn(size_batch, 100, device = device)  
        labels_for_validate.fill_(fake_labels)
        fake_output = generator(noise, disc_label_fake)
        validity, op_label = discriminator(fake_output.detach())       
        dis_fake_valid_error = criterion(validity, labels_for_validate)
        disc_fakelabel_err = F.nll_loss(op_label, class_fake_labels)
        dis_fake_error = dis_fake_valid_error + disc_fakelabel_err
        dis_fake_error.backward()
        disc_err_final = disc_error_real + dis_fake_error
        valid_mean2 = validity.mean().item()
        dis_optim.step()
    
        # Train Generator
        labels_for_validate.fill_(1)
        labels_for_gen = torch.randint(0, 10, (size_batch, ), device = device, dtype = torch.long)
        noise = torch.randn(size_batch, 100, device = device)  
        gen_optim .zero_grad()
        fake_output = generator(noise, labels_for_gen)
        validity, op_label = discriminator(fake_output)
        gen_valid_error = criterion(validity, labels_for_validate)        
        gen_label_error = F.nll_loss(op_label, labels_for_gen)
        final_gen_error = gen_valid_error + gen_label_error
        final_gen_error.backward()
        valid_mean3 = validity.mean().item()
        gen_optim .step()
        
        
        print("[{}/{}] [{}/{}] D(x): [{:.4f}] D(G): [{:.4f}/{:.4f}] GLoss: [{:.4f}] DLoss: [{:.4f}] DLabel: [{:.4f}] "
              .format(epoch, num_epochs, index, len(data_loader), valid_mean1, valid_mean2, valid_mean3, final_gen_error, disc_err_final,
                      disc_reallabel_err+ disc_fakelabel_err + gen_label_error))
        
        # Save errors for graphing
        loss_list_disc.append(disc_err_final.cpu().detach().numpy())
        loss_list_gen.append(final_gen_error.cpu().detach().numpy())
        
    # Save images to folder
    labels = torch.arange(0,10,dtype = torch.long,device = device)
    noise = torch.randn(10,100,device = device)  
    images = generator(noise, labels)
    vutils.save_image(images.detach(),'ACGANOutput/fake_samples_epoch_%03d.png' % (epoch), normalize = True)


# Plot the loss of the generator and the descriminator
plt.plot(counter_list, loss_list_gen, 'r.', label='Generator')
plt.plot(counter_list, loss_list_disc, 'g.', label='Discriminator')
plt.title("ACGAN Loss of Discriminator and Generator")
plt.xlabel("Batch Number")
plt.ylabel("Loss (Binary Cross Entropy)")
plt.legend(loc="best")
plt.show()
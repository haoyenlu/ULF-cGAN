import torch
import torch.nn as nn
from torchsummary import summary
from torchsummaryX import summary as summaryx

from models import Generator, Discriminator

class cGAN:
    def __init__(self,seq_len,features=3,n_critic=3,lr=5e-4,
                 g_hidden=50,d_hidden=50,max_iters=1000,
                 label_dim=5,
                 saveDir=None,ckptPath=None,prefix="T01"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Train on {}".format(self.device))

    
        self.G = Generator(seq_len,features,g_hidden,label_dim=label_dim).to(self.device)
        self.D = Discriminator(seq_len,features,d_hidden,label_dim=label_dim).to(self.device)

        self.load_ckpt(ckptPath)

        self.lr = lr
        self.n_critic = n_critic

        self.g_optimizer = torch.optim.RMSprop(self.G.parameters(),lr=self.lr)
        self.d_optimizer = torch.optim.RMSprop(self.D.parameters(),lr=self.lr)

        self.seq_len = seq_len
        self.features = features
        self.label_dim = label_dim

        self.sample_size = 2
        self.max_iters = max_iters
        self.saveDir = saveDir
        self.g_hidden = g_hidden
        self.d_hidden = d_hidden
        self.prefix = prefix

    def train(self,dataloader):
        # summaryx(self.G,torch.zeros(2,self.seq_len,device=self.device),torch.zeros(2,dtype=torch.long,device=self.device))
        # summaryx(self.D,torch.zeros(2,self.features,self.seq_len,device=self.device),torch.zeros(2,dtype=torch.long,device=self.device))
        

        data = self.get_infinite_batch(dataloader)
        batch_size = 4

        
        criterion = nn.BCELoss()


        for g_iter in range(self.max_iters):
            for p in self.D.parameters():
                p.requires_grad = True
            

            self.G.train()

            for d_iter in range(self.n_critic):
                self.D.zero_grad()
                self.G.zero_grad()
                
                sequence , label = data.__next__()
                real_seq = torch.autograd.Variable(sequence).float().to(self.device)
                real_seqlabel = torch.autograd.Variable(label).long().to(self.device)

                batch_size = real_seq.size(0)

                real_label = torch.autograd.Variable(torch.Tensor(batch_size, 1).fill_(1), requires_grad=False).to(self.device)
                fake_label = torch.autograd.Variable(torch.Tensor(batch_size, 1).fill_(0), requires_grad=False).to(self.device)

                d_loss_real = criterion(self.D(real_seq,real_seqlabel),real_label)

                z = torch.randn(batch_size,1,self.seq_len).to(self.device)
                fake_seqlabel = torch.randint(low=0,high=self.label_dim,size=(batch_size,),device=self.device)

                fake = self.G(z,fake_seqlabel)  
                d_loss_fake = criterion(self.D(fake,fake_seqlabel),fake_label)

                d_loss = d_loss_fake + d_loss_real

                if self.use_spectral: 
                    sp_loss = self.spectral_loss(real_seq,fake)
                    d_loss += 0.5 * sp_loss

                d_loss.backward()

                self.d_optimizer.step()
                print(f'Discriminator iteration: {d_iter}/{self.n_critic}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')

            self.G.zero_grad()
            self.D.zero_grad()

            z = torch.randn(batch_size,1,self.seq_len).to(self.device)
            fake_seqlabel = torch.randint(low=0,high=self.label_dim,size=(batch_size,),device=self.device)
            fake = self.G(z,fake_seqlabel)
            g_loss = criterion(self.D(fake,fake_seqlabel),real_label)
            g_loss.backward()

            self.g_optimizer.step()
            print(f'Generator iteration: {g_iter}/{self.max_iters}, g_loss: {g_loss}')

            if g_iter % 50 == 0:
                self.save_model()


            torch.cuda.empty_cache()

        self.save_model()
        print("Finished Training!!")
    

    def load_ckpt(self,ckptPath):
        if ckptPath:
            print("Load Checkpoint....")
            ckpt = torch.load(ckptPath,map_location=self.device)
            self.G.load_state_dict(ckpt['G_param'])
            self.D.load_state_dict(ckpt['D_param'])


    def generate_samples(self,sample_size):
        z = torch.randn(sample_size,1,self.seq_len).to(self.device)
        fakes = self.G(z).detach().cpu().numpy()
        
        return fakes
    

    def save_model(self):
        torch.save({"G_param":self.G.state_dict(),"D_param":self.D.state_dict()},
                f"{self.saveDir}/{self.prefix}_net_G{self.g_hidden}_D{self.d_hidden}_ckpt.pth")
    

    def get_infinite_batch(self,dataloader):
        while True:
            for data in dataloader:
                yield data
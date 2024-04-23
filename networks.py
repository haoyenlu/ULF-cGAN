import torch
import torch.nn as nn
from torchsummary import summary
from torchsummaryX import summary as summaryx

from models import Generator, Discriminator

class cGAN:
    def __init__(self,seq_len,features=3,n_critic=3,lr=5e-4,
                 g_hidden=50,d_hidden=50,max_iters=1000,latent_dim=200,
                 label_dim=5, w_loss = False,
                 saveDir=None,ckptPath=None,prefix="T01"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Train on {}".format(self.device))

        self.w_loss = w_loss

    
        self.G = Generator(seq_len,features,g_hidden,label_dim=label_dim,latent_dim=latent_dim).to(self.device)
        self.D = Discriminator(seq_len,features,d_hidden,label_dim=label_dim).to(self.device)

        self.load_ckpt(ckptPath)

        self.lr = lr
        self.n_critic = n_critic

        self.g_optimizer = torch.optim.RMSprop(self.G.parameters(),lr=self.lr)
        self.d_optimizer = torch.optim.RMSprop(self.D.parameters(),lr=self.lr)

        self.seq_len = seq_len
        self.features = features
        self.label_dim = label_dim
        self.latent_dim = latent_dim

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
                z = torch.randn(batch_size,1,self.latent_dim).to(self.device)
                fake_seqlabel = torch.randint(low=0,high=self.label_dim,size=(batch_size,),device=self.device)

                fake = self.G(z,fake_seqlabel)  

                d_loss = self._get_critic_loss(self.D(fake,fake_seqlabel),self.D(real_seq,real_seqlabel))
                d_loss.backward()

                self.d_optimizer.step()
                print(f'Discriminator iteration: {d_iter}/{self.n_critic}, Discriminator Loss: {d_loss}')

            self.G.zero_grad()
            self.D.zero_grad()

            z = torch.randn(batch_size,1,self.latent_dim).to(self.device)
            fake_seqlabel = torch.randint(low=0,high=self.label_dim,size=(batch_size,),device=self.device)
            fake = self.G(z,fake_seqlabel)
            g_loss = self._get_generator_loss(self.D(fake,fake_seqlabel))
            g_loss.backward()

            self.g_optimizer.step()
            print(f'Generator iteration: {g_iter}/{self.max_iters}, g_loss: {g_loss}')

            if g_iter % 50 == 0:
                self.save_model()


            torch.cuda.empty_cache()

        self.save_model()
        print("Finished Training!!")
    

    def _get_critic_loss(self,fake_prediction,real_prediction):
        batch_size = fake_prediction.size(0)

        if self.w_loss == False: 
            real_label = torch.autograd.Variable(torch.Tensor(batch_size, 1).fill_(1), requires_grad=False).to(self.device)
            fake_label = torch.autograd.Variable(torch.Tensor(batch_size, 1).fill_(0), requires_grad=False).to(self.device)

            loss_fake = nn.functional.binary_cross_entropy_with_logits(fake_prediction,fake_label)
            loss_real = nn.functional.binary_cross_entropy_with_logits(real_prediction,real_label)
            loss_d = loss_fake + loss_real
        else:
            loss_d = fake_prediction.mean() - real_prediction.mean()
        
        return loss_d

    def _get_generator_loss(self,prediction):
        batch_size = prediction.size(0)

        if self.w_loss == False:
            real_label = torch.autograd.Variable(torch.Tensor(batch_size, 1).fill_(1), requires_grad=False).to(self.device)
            g_loss = nn.functional.binary_cross_entropy_with_logits(prediction,real_label)
        else:
            g_loss = prediction.mean()
        
        return g_loss


    def load_ckpt(self,ckptPath):
        if ckptPath:
            print("Load Checkpoint....")
            ckpt = torch.load(ckptPath,map_location=self.device)
            self.G.load_state_dict(ckpt['G_param'])
            self.D.load_state_dict(ckpt['D_param'])


    def generate_samples(self,sample_size):
        z = torch.randn(sample_size,1,self.latent_dim).to(self.device)
        label = torch.randint(low=0,high=self.label_dim,size=(sample_size,),device=self.device)
        fakes = self.G(z,label).detach().cpu().numpy()
        
        return fakes,label.cpu().numpy()
    

    def save_model(self):
        torch.save({"G_param":self.G.state_dict(),"D_param":self.D.state_dict()},
                f"{self.saveDir}/{self.prefix}_net_G{self.g_hidden}_D{self.d_hidden}_ckpt.pth")
    

    def get_infinite_batch(self,dataloader):
        while True:
            for data in dataloader:
                yield data
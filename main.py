import argparse
import numpy as np
import torch


from networks import cGAN

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--dir',help="Patient Dataset path",default=None)
        parser.add_argument('--saveDir',help="Save Checkpoint Directory",default=None)
        parser.add_argument('--saveSample',help="Save Sample Dir",default=None)
        parser.add_argument('--ckpt',help="Checkpoint path",default=None)
        parser.add_argument('--max_iter',type=int,default=1000)
        parser.add_argument('--batch_size',type=int,default=8)
        parser.add_argument('--task',help="Task to train on",default='T01')
        parser.add_argument('--sample_size',type=int,help="Sample Size",default=200)

        ''' GAN model '''
        parser.add_argument('--g_hidden',type=int,help="Generator hidden channel size",default=50)
        parser.add_argument('--d_hidden',type=int,help="Discriminator hidden channel size",default=50)
        parser.add_argument('--n_critic',type=int,help="Number of iterations for Discriminator per one Generator iterations",default=5)




        args = parser.parse_args()


        numpy_data = np.load(args.dir,allow_pickle=True).item()
        train_data = numpy_data['data']
        train_label = numpy_data['label']
        feat,seq_len = train_data[0].shape

        dataset  = []
        for data,label in zip(train_data,train_label):
                dataset.append([data,label])

        dataloader = torch.utils.data.DataLoader(dataset,args.batch_size,shuffle=True)

        print(f"Train with {args.max_iter} iterations")
        print(f"Features:{feat},Sequence Length:{seq_len}")

        model = cGAN(seq_len = seq_len, features=feat,n_critic=args.n_critic,
                g_hidden=args.g_hidden,d_hidden=args.d_hidden,max_iters=args.max_iter,
                saveDir=args.saveDir,ckptPath=args.ckpt,prefix=args.task)


        model.train(dataloader)

        sample,label = model.generate_samples(args.sample_size)
        np.save(f"{args.sampleDir}/cGAN{args.g_hidden}{args.d_hidden}_{args.task}_samples.npy",
                {'sequence':sample,'label':label})
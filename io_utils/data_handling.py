from torch.utils.data import Dataset
from skimage.transform import resize

import numpy as np

class simple_np_ds(Dataset):
    """
    Dataset encapsulating simple numpy dataset
    """


    def __init__(self, path, val_split, test_split, shuffle=True, transform=None, reduced_dataset_size=None, seed=42):
        
        self.file_names=['pi0','piplus']
        self.layer_names=['EMB1','EMB2','EMB2','TileBar0','TileBar1','TileBar2']
        self.np_file_list=[np.load(path+f+'.npz') for f in self.file_names]
        self.data_arrs={}
        for l in self.layer_names:
            self.data_arrs[l]=np.concatenate([file[l].astype(np.float32) for file in self.np_file_list],axis=0)
            
        self.labels=np.ravel(np.concatenate([file['label'] for file in self.np_file_list],axis=0))
        
        
        self.quant_names=np.asarray(['cluster_sumCellE', 
                                     'cluster_nCells', 
                                     'clusterPt', 
                                     'clusterEta', 
                                     'cluster_emProb', 
                                     'clusterPhi', 
                                     'clusterE'])
        
        self.quant_arrs={}
        for q in self.quant_names:
            self.quant_arrs[q]=np.concatenate([file[q].astype(np.float32) for file in self.np_file_list],axis=0)
        
        
     
        assert np.all(np.asarray([self.data_arrs[k].shape[0] for k in self.layer_names])==self.labels.shape[0])
        assert np.all(np.asarray([self.quant_arrs[k].shape[0] for k in self.quant_names])==self.labels.shape[0])
        
        
        
        self.transform=transform
        
        self.reduced_size = reduced_dataset_size
        
        #save prng state
        rstate=np.random.get_state()
        if seed is not None:
            np.random.seed(seed)

        indices = np.arange(len(self.labels))

        if self.reduced_size is not None:
            print("Reduced size: {}".format(self.reduced_size))
            assert len(indices)>=self.reduced_size
            indices = np.random.choice(self.labels.shape[0], reduced_dataset_size)

        #shuffle index array
        if shuffle:
            np.random.shuffle(indices)
        
        #restore the prng state
        if seed is not None:
            np.random.set_state(rstate)

        n_val = int(len(indices) * val_split)
        n_test = int(len(indices) * test_split)
        self.train_indices = indices[:-n_val-n_test]
        self.val_indices = indices[-n_test-n_val:-n_test]
        self.test_indices = indices[-n_test:]
        print('length of the training indices: {}'.format(len(self.train_indices)))
        print('length of the validation indices: {}'.format(len(self.val_indices)))
        print('length of the test indices: {}'.format(len(self.test_indices)))
        
        self.compute_scaling()
       
    def compute_scaling(self):
        self.i_clE=np.where(self.quant_names=='clusterE')[0][0]
        self.avg_clE=np.average( self.quant_arrs[self.quant_names[self.i_clE]][self.train_indices] )
        print('computed average cluster E: {}'.format(self.avg_clE))
        
    def cle_scale(self,d_current,clusterE):
        factor=clusterE/self.avg_clE
        #print('cluster E scale factor: {}'.format(factor))
        return d_current*factor
        
        
    def __getitem__(self,index):
        d_current=np.asarray([resize(self.data_arrs[l][index],
                             (128,16),preserve_range=True,
                             anti_aliasing=False,
                             anti_aliasing_sigma=None,
                             order=0) for l in self.layer_names],dtype=np.float32)
        if self.transform is None:
            return d_current,  self.labels[index]
        elif self.transform == self.cle_scale:
            #print('applying cluster energy scale')
            clusterE=self.quant_arrs['clusterE'][index]
            return self.transform(d_current,clusterE),  self.labels[index]
        else:
            return self.transform(d_current),  self.labels[index]
            



    def __len__(self):
        if self.reduced_size is None:
            return self.labels.shape[0]
        else:
            return self.reduced_size


    def __del__(self):
        for f in self.np_file_list:
            f.close()

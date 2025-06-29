import math
from typing import Dict, List, Union

import numpy as np
import torch
import torch.utils.data as tdata
import soundfile
from torchaudio import datasets
import torch.nn.functional as F

# Libri2Mix 8k wav min mix_clean

def collate_LibriMix( batches ):
    chunk_size = 6 * 8000
    length_max = 0
    for batch in batches:
        sr, mix, ( sp1, sp2 ) = batch
        if length_max < mix.size(1):
            length_max = mix.size(1)
        #print( "mix size:", mix.size())
        #print( "sp1 size:", sp1.size())
        #print( "sp2 size:", sp2.size())
        
    mixes = []
    sp1s = []
    sp2s = []
    for batch in batches:
        sp = []
        sr, mix, ( sp1, sp2 ) = batch
        #print( "mix size:", mix.size())
        pad_len = length_max - mix.size(1)
        mix = F.pad(mix[0], 
                   (0, pad_len), 
                   'constant', 
                   0)
        #mix = torch.tensor( mix )
        #print( "2 mix:", mix )
        sp1 = F.pad(sp1[0], 
                   (0, pad_len), 
                   'constant', 
                   0)
        #sp1 = torch.tensor( sp1 )
        #print( "sp1:", sp1 )
        sp2 = F.pad(sp2[0], 
                   (0, pad_len), 
                   'constant', 
                   0)
        #sp2 = torch.tensor( sp2 )
        #print( "sp2:", sp2 )
        if length_max > chunk_size:
            mix = mix[:chunk_size]
            sp1 = sp1[:chunk_size]
            sp2 = sp2[:chunk_size]
        #mix.append( mix )
        mixes.append( mix )
        #print( "1 mixes:", mixes )
        #sps.append( torch.stack( [sp1, sp2], dim = 0 ) )
        sp1s.append( sp1 )
        sp2s.append( sp2 )
    sp1s = torch.stack( sp1s )
    sp2s = torch.stack( sp2s )

        
    mixtures = torch.stack( mixes )
    sources = [sp1s, sp2s]
    sources = torch.stack( sources )
    #print( "0 size of sources:", sources.size() )
    #print( "permute size of sources:", sources.permute( 2, 0, 1 ).size() )

    #return mixtures, sources.permute( 2, 0, 1 )
    return mixtures, sources


class MyDatasets(torch.utils.data.Dataset):
    def __init__(self, mode):

        if mode == "train":
            self.dataset = datasets.LibriMix( "/mnt/ssd1/uchiyats/source_separation_data/LibriMix/data", subset = 'train-100', num_speakers = 2, sample_rate = 8000, task = 'sep_clean', mode='min')
        elif mode == "valid":
            self.dataset = datasets.LibriMix( "/mnt/ssd1/uchiyats/source_separation_data/LibriMix/data", subset = 'dev', num_speakers = 2, sample_rate = 8000, task = 'sep_clean', mode='min')
        else:
            self.dataset = datasets.LibriMix( "/mnt/ssd1/uchiyats/source_separation_data/LibriMix/data", subset = 'test', num_speakers = 2, sample_rate = 8000, task = 'sep_clean', mode='min')

        self.datanum = len(self.dataset) 

        self.mode = mode
            
    def __len__(self):
        return self.datanum
        #return self.datanum // 1600
    
    def __getitem__(self, idx):
        batch = self.dataset[idx]
        
        return batch


import json

import torch


class CustomData(torch.utils.data.Dataset):
    def __init__(self,data):
        super(CustomData,self).__init__()
        self.data = data
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self,index):
        ins = torch.tensor(self.data[index][0],dtype=torch.long)
        outs = torch.tensor(self.data[index][1],dtype= torch.long)
        return {
            "input_sequence" : ins,
            "output_sequence" : outs,
            "mask" : self.masking(outs),
            "input_length" : self.get_length(ins)
        }
    
    def masking(self,seq):
        mask = list()
        for s in seq:
            if s == 0:
                mask.append(False)
            else:
                mask.append(True)
        return torch.tensor(mask)

    def get_length(self,seq):
        return sum(seq!= 0)

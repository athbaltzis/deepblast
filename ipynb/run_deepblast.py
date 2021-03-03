#!/usr/bin/env python

import torch
from deepblast.trainer import LightningAligner
from deepblast.dataset.utils import pack_sequences
from deepblast.dataset.utils import states2alignment
import argparse
import os
import sys

parser = argparse.ArgumentParser()

#parser.add_argument("-p","--path",action='store', dest='pt_model',required=True,help='Define the path to the pretrained model',default='/Users/abaltzis/deepblast/deepblast/pretrained_models/deepblast-lstm4x.pt')
parser.add_argument("-o","--output", action='store', dest='output', help='Define an output filename',default='output.fa_aln')
parser.add_argument("-i","--input",action='store',dest='input',required=True, help='Input sequences in fasta format files')
args = parser.parse_args()
# Load the pretrained model
pt_model = '/Users/abaltzis/deepblast/deepblast/pretrained_models/deepblast-lstm4x.pt'
model = LightningAligner.load_from_checkpoint(pt_model).cpu()

seqs={}
with open(args.input) as f:
	for i in f:
		i=i.rstrip()
		if i[0]=='>':
			ID=i[1:]
			continue
		seqs[ID]=seqs.get(ID,'')+i

keys_list = list(seqs)
x = seqs[keys_list[0]]
y = seqs[keys_list[1]]
pred_alignment = model.align(x, y)

x_aligned, y_aligned = states2alignment(pred_alignment, x, y)

file = open(args.output,"w")
file.write(">%s\n%s\n>%s\n%s" % (keys_list[0], x_aligned, keys_list[1], y_aligned))
file.close()

print(x_aligned)
print(pred_alignment)
print(y_aligned)

x_ = torch.Tensor(model.tokenizer(str.encode(x))).long()
y_ = torch.Tensor(model.tokenizer(str.encode(y))).long()

seq, order = pack_sequences([x_], [y_])

score = model.aligner.score(seq, order).item()
print('Score', score)



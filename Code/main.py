import training as tr
import classify as cl

import argparse

#Set up command line argument parser
parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest='command')

#Designate 2 subparsers, one for training and one for classification
training = subparser.add_parser('training')
classify = subparser.add_parser('classify')

#Add arguments for the training subparser
training.add_argument('-i', type=str, required=True)
training.add_argument('-os', type=str, required=True)
training.add_argument('-oh', type=str, required=True)

#Add arguments for the classify subparser
classify.add_argument('-i', type=str, required=True)
classify.add_argument('-is', dest='isfile', type=str, required=True)
classify.add_argument('-ih', type=str, required=True)
classify.add_argument('-o', type=str, required=True)

args = parser.parse_args()

#Based on the command, train or classify
if args.command == 'training':
    tr.bayes(args.i, args.os, args.oh)

elif args.command == 'classify':
    cl.test(args.i, args.isfile, args.ih, args.o)
#!/usr/bin/env python

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True  # disable ROOT internal argument parser

import logging
logger = logging.getLogger("annotate_file.py")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

import numpy as np
from keras.models import load_model
import yaml
import os
import pickle
from array import array
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Apply Keras model on ROOT file")
    parser.add_argument(
        "--config-training",
        default="mt_training_config.yaml",
        help="Path to training config file")
    parser.add_argument(
        "--dir-prefix",
        type=str,
        default="mt_",
        help="Prefix of directories in ROOT file to be annotated.")
    parser.add_argument(
        "input", help="Path to input file, where response will be added.")
    parser.add_argument(
        "tag", help="Tag to be used as prefix of the annotation.")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["fold1_keras_model.h5", "fold0_keras_model.h5"],
        help=
        "Keras models to be used for the annotation. Note that these have to be booked in the reversed order [fold1*, fold0*], so that the training is independent from the application."
    )
    parser.add_argument(
        "--preprocessing",
        type=str,
        nargs="+",
        default=[
            "fold1_keras_preprocessing.pickle",
            "fold0_keras_preprocessing.pickle"
        ],
        help=
        "Data preprocessing to be used. Note that these have to be booked in the reversed order [fold1*, fold0*], so that the preprocessing is independent for the folds."
    )
    parser.add_argument(
        "--tree", default="ntuple", help="Name of trees in the directories.")
    parser.add_argument(
        "--event-branch",
        default="event",
        help=
        "Name of branch which holds the event number used to select the independent classifier."
    )
    return parser.parse_args()


def parse_config(filename):
    return yaml.load(open(filename, "r"))


def main(args, config):
    # Sanity checks
    if not os.path.exists(args.input):
        logger.fatal("Input file %s does not exist.", args.input)
        raise Exception

    logger.debug("Following mapping of classes to class numbers is used.")
    for i, class_ in enumerate(config["classes"]):
        logger.debug("%s : %s", i, class_)

    # Load Keras models and preprocessing
    classifier = [load_model(x) for x in args.models]
    preprocessing = [pickle.load(open(x, "rb")) for x in args.preprocessing]

    # Open input file
    file_ = ROOT.TFile(args.input, "UPDATE")
    if file_ == None:
        logger.fatal("File %s is not existent.", args.input)
        raise Exception

    # Loop through directories in this file and annotate tree if directory
    # starts with the set prefix.
    for key in file_.GetListOfKeys():
        # Find valid directories
        name = key.GetName()
        if name.startswith(args.dir_prefix):
            logger.debug("Process directory %s.", name)
            tree = file_.Get(os.path.join(name, args.tree))
            if tree == None:
                logger.fatal("Failed to find tree %s in directory %s.",
                             args.tree, name)
                raise Exception

            # Book branches for annotation
            values = []
            for variable in config["variables"]:
                values.append(array("f", [-999]))
                tree.SetBranchAddress(variable, values[-1])

            response_max_score = array("f", [-999])
            branch_max_score = tree.Branch("{}_max_score".format(
                args.tag), response_max_score, "{}_max_score/F".format(
                    args.tag))

            response_max_index = array("f", [-999])
            branch_max_index = tree.Branch("{}_max_index".format(
                args.tag), response_max_index, "{}_max_index/F".format(
                    args.tag))

            # Run the event loop
            for i_event in range(tree.GetEntries()):
                tree.GetEntry(i_event)

                # Get event number and compute response
                event = int(getattr(tree, args.event_branch))
                values_stacked = np.hstack(values).reshape(1, len(values))
                values_preprocessed = preprocessing[event % 2].transform(
                    values_stacked)
                response = classifier[event % 2].predict(values_preprocessed)
                response = np.squeeze(response)

                # Find max score and index
                response_max_score[0] = -999.0
                for i, r in enumerate(response):
                    if r > response_max_score[0]:
                        response_max_score[0] = r
                        response_max_index[0] = i

                # Fill branches
                branch_max_score.Fill()
                branch_max_index.Fill()

    # Write everything to file
    file_.Write()
    file_.Close()


if __name__ == "__main__":
    args = parse_arguments()
    config = parse_config(args.config_training)
    main(args, config)

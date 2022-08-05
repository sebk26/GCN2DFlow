"""
Utility functions for loading, printing and saving models and exporting samples
"""
import os
import glob
import torch
from pathlib import Path
import logging
import json
from datetime import datetime
import argparse
from torch_geometric.data import DataLoader
from shutil import copyfile
import pandas as pd
import numpy as np
from torchGeom_ptCloud import torchdataset
from normalize import MinMaxScaler


def setup_logger(name, log_file, level=logging.DEBUG, ltype="historyLog"):

    if ltype == "default":
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    elif ltype == "historyLog":
        formatter = logging.Formatter("%(message)s")
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    # stream = logging.StreamHandler(stream=None)
    # stream.setLevel(logging.CRITICAL)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    # logger.addHandler(stream)
    return logger


def gnn_model_summary(model, logger=logging):
    """Print model summary (credits: stackoverflow)"""
    model_params_list = list(model.named_parameters())
    logger.debug("-" * (25 + 25 + 20))
    line_new = "{:>25}  {:>25}  {:>15}".format(
        "Layer.Parameter", "Param Tensor Shape", "Param #"
    )
    logger.debug(line_new)
    logger.debug("-" * (25 + 25 + 20))
    for elem in model_params_list:
        p_name = elem[0]
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
        line_new = "{:>25}  {:>25} {:>15}".format(p_name, str(p_shape), str(p_count))
        logger.debug(line_new)
    logger.debug("-" * (25 + 25 + 20))
    total_params = sum([param.nelement() for param in model.parameters()])
    logger.debug("Total params:{}".format(total_params))
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.debug("Trainable params: {}".format(num_trainable_params))
    logger.debug("Non-trainable params: {}".format(total_params - num_trainable_params))


def create_Argparse(timestr=datetime.now().strftime("%Y%m%d-%H%M%S")):

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-s",
        "--source",
        help="Path to *.npz files containing \
            output of 01_GraphConv/assemble_ptcloud.py",
        type=str,
        required=True,
    )

    ap.add_argument(
        "-m",
        "--modeldir",
        help="Path to save checkpoints and \
                    modelfile containing model from save_checkpoint function",
        type=str,
        required=False,
        default="./run_{}".format(timestr),
    )

    ap.add_argument(
        "-t",
        "--type",
        help="Modeltype (e.g. SimpleNet",
        type=str,
        required=False,
        default=None,
    )

    ap.add_argument(
        "-lr",
        "--learning_rate",
        help="Learning Rate",
        type=float,
        required=False,
        default=1.0,
    )

    ap.add_argument(
        "-e", "--epochs", help="Epochs", type=int, required=False, default=4000
    )

    ap.add_argument(
        "-b", "--batch", help="Batchsize", type=int, required=False, default=1
    )

    ap.add_argument(
        "--scalerpath", help="Path to Scaler", type=str, required=False, default=None
    )

    ap.add_argument(
        "--optimizer",
        help='optimizer by string. Default "adam"',
        type=str,
        required=False,
        default="adam",
    )

    ap.add_argument(
        "--dataset",
        help="load precompiled dataset from disk",
        type=str,
        required=False,
        default=None,
    )

    ap.add_argument(
        "--sinterval", help="Saving Interval", type=int, required=False, default=50
    )
    ap.add_argument(
        "--NormInput",
        help="Normalize Input/Pos",
        action="store_true",
        required=False,
        default=False,
    )

    ap.add_argument(
        "-j",
        "--jobID",
        help="Job ID to include in directory name",
        type=str,
        required=False,
        default=None,
    )

    ap.add_argument(
        "-r",
        "--restore",
        help='Restore model from "<modeldir>/checkpoint"',
        action="store_true",
        required=False,
        default=False,
    )

    ap.add_argument(
        "-rO",
        "--restoreOptimizer",
        help='Restore optimizer from "<modeldir>/checkpoint"',
        action="store_true",
        required=False,
        default=False,
    )

    ap.add_argument(
        "--splitarr",
        nargs="+",
        help="splitarr",
        type=float,
        required=False,
        default=[0.8, 0.1, 0.1],
    )

    ap.add_argument(
        "--idents",
        help="Identifier that are concluded in sample "
        "filename. Default: \"['EquiPenta','Triangle',"
        " 'Ellipse','Square','EquiHex','Circle','EquiTri','Rectangle'].\" ",
        type=json.loads,
        required=False,
        default='["EquiPenta","Triangle","Ellipse","Square",'
        + '"EquiHex","Circle","EquiTri","Rectangle"]',
    )  # , default=cases)

    ap.add_argument(
        "--knnK",
        help="Number of nearest neighbors for knn_graph",
        type=int,
        required=False,
        default=6,
    )
    return ap


def evalArgparse(args, with_inds=False):
    for arg in vars(args):
        print("{} : {}".format(arg, getattr(args, arg)))
    if os.path.isdir(args.source):
        datadir = args.source
    else:
        raise ValueError("Supply valid Path to directory containing *.npz files!")

    logging.info("Sourcedirectory: {}".format(datadir))

    if args.modeldir[-1] == os.sep:
        args.modeldir = args.modeldir[:-1]

    # Create dir for configLogger
    if args.restore and Path(args.modeldir + os.sep + "checkpoint.pth").is_file():
        Rmodeldir = args.modeldir
        args.modeldir = "./run-restored_{}".format(
            datetime.now().strftime("%Y%m%d-%H%M%S")
        )

    # Include jobID in directory name if provided (after restore name is copied)
    if args.jobID is not None:
        args.modeldir = args.modeldir + "_{}".format(args.jobID)

    # create Modeldir so scaler can be dumped
    if not os.path.isdir(args.modeldir):
        os.mkdir(args.modeldir)

    # Config Logger
    configLog = setup_logger(
        "configLog",
        args.modeldir + os.sep + "torch-geometric.config",
        level=logging.DEBUG,
    )

    # Get Data
    if args.dataset is None:
        # Create new dataset from directory of *.npz files
        dataset1 = torchdataset(
            datadir,
            args.idents,
            splitarr=args.splitarr,
            wi=with_inds,
            mdir=args.modeldir,
            k=args.knnK,
        )

        [trainset, valiset, testset] = dataset1.data

    else:
        # Load dataset file with precompiled torchdataset
        logging.info("Load {} ...".format(args.dataset))
        dat1 = torch.load(args.dataset, map_location=torch.device("cpu"))
        [trainset, valiset, testset] = dat1["data"]
        configLog.info("Imported Dataset: {}".format(args.dataset))
        configLog.debug("Rootdir of Dataset: {}".format(dat1["rootdir"]))

        # Check that rootdir of Dataset and given flag match
        if datadir[-1] == os.sep:
            datadir = datadir[:-1]
        if dat1["rootdir"][-1] == os.sep:
            dat1["rootdir"] = dat1["rootdir"][:-1]

        assert dat1["rootdir"] == datadir

        # Check that number of points Match
        assert dat1["N"] == args.Npts

    logging.info("... Successfully Imported Datasets.")
    # Get Input and Output dimensions of Dataset
    indim, outdim = trainset[0].x.shape[-1], trainset[0].y.shape[-1]

    # Get Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get Model
    if not args.restore:
        if args.type is None:
            raise ValueError("Option --type is required if new model is created!")
        model = getModelbyName(args.type, indim, outdim, configLog)
        optimizer, args.learning_rate = initialize_optimizer(model.parameters(), args)

    # Create MinMaxScaler
    scaler = MinMaxScaler(args.modeldir)

    if args.scalerpath is not None:
        scaler.restore(args.scalerpath, configLog, dump=True)

    if args.restore:
        # Variable Rmodeldir only exists if restored
        if Path(Rmodeldir + os.sep + "checkpoint.pth").is_file():
            logging.info(
                "Restore model from last checkpoint of"
                + " modeldir {}".format(Rmodeldir)
            )
            model, optimizer, _, args.learning_rate = load_checkpoint(
                Rmodeldir + os.sep + "checkpoint.pth", device, args
            )
        if args.scalerpath is None:
            scaler.restore(
                Rmodeldir + os.sep + "MinMaxScaler.pth", configLog, dump=True
            )
            trainset = scaler.transformData(
                trainset, device=device, NormInput=args.NormInput
            )
        # Log origin of restored model
        configLog.debug("Restored Model: {}".format(Rmodeldir + os.sep + "checkpoint"))
        copyfile(
            Rmodeldir + os.sep + "history.csv", args.modeldir + os.sep + "history.csv"
        )

        startepoch = get_startepoch(glob.glob(args.modeldir + os.sep + "history.csv"))

        endepoch = startepoch + args.epochs
        configLog.debug(
            "Modeltype: {}".format(
                get_modelname(glob.glob(Rmodeldir + os.sep + "*.config"))
            )
        )
    else:
        scalerpath = args.modeldir + os.sep + "MinMaxScaler"
        # aquire dataset statistics
        trainset = scaler.transformData(
            trainset,
            True,
            scalerpath,
            configLog,
            device=device,
            NormInput=args.NormInput,
        )
        args.restore = False
        startepoch = 0
        endepoch = args.epochs

    valiset = scaler.transformData(
        valiset, device=device, NormInput=args.NormInput
    )

    trainloader = DataLoader(
        trainset, batch_size=args.batch, shuffle=True, num_workers=4
    )
    valiloader = DataLoader(valiset, batch_size=1, num_workers=4)

    for k in scaler.scaler.keys():
        print(k)
        print(scaler.scaler[k])

    configLog.debug("### New Config for Model {}".format(args.modeldir))
    configLog.debug("Source: {}".format(args.source))
    configLog.debug("Epochs: {}".format(args.epochs))
    configLog.debug("Learning Rate(LR): {}".format(args.learning_rate))
    configLog.debug("Batchsize: {}".format(args.batch))

    del trainset, valiset
    return (
        trainloader,
        valiloader,
        configLog,
        startepoch,
        endepoch,
        model,
        args.modeldir,
        device,
        optimizer,
    )


def getModelbyName(mtype, indim, outdim, configLog=logging):
    """Load different model architectures"""

    mtype = mtype.strip()

    if mtype == "GCN_2LSTM":
        from models.GCNConv_LSTM import GCN_2LSTM

        model = GCN_2LSTM(indim, outdim)
    elif mtype == "SimpleExpanse2":
        from models.SimpleExpanseNet2 import GCN

        model = GCN(indim, outdim)
    elif mtype == "SimpleExpanse6":
        from models.SimpleExpanseNet6 import GCN6

        model = GCN6(indim, outdim)
    else:
        raise NotImplementedError("Model {} is not implemented.".format(mtype))

    configLog.debug("Modeltype: {}".format(mtype))

    print(model)

    return model


def compile_checkpoint(model, optimizer, epoch):
    return {
        "model": model,
        "model_state_dict": model.state_dict(),
        "optimizer": optimizer,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }


def load_checkpoint(cpath, device, args):
    """load checkpoint and optionally restore optimizer"""
    logging.info("Load Model from checkpoint {}".format(cpath))
    checkpoint = torch.load(cpath, map_location=device)
    model = checkpoint["model"]
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer, lr = initialize_optimizer(model.parameters(), args)

    if args.restoreOptimizer:
        optimizer = checkpoint["optimizer"]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    startepoch = int(checkpoint["epoch"]) + 1
    logging.info("Checkpoint Epoch: {}".format(startepoch))

    return model, optimizer, startepoch, lr


def initialize_optimizer(params, args):
    """Initialize Optimizer"""
    if args.optimizer == "adadelta":
        return (
            torch.optim.Adadelta(params, lr=args.learning_rate, rho=0.95, eps=1e-07),
            args.learning_rate,
        )
    elif args.optimizer == "adam":
        return (
            torch.optim.Adam(params, lr=args.learning_rate, eps=1e-06),
            args.learning_rate,
        )
    else:
        raise NotImplementedError("Optimizer {} not known".format(args.optimizer))
        exit(1)


def get_startepoch(histglob):
    """
    Read history file of case to be restored and parse last epoch
    @confglob: glob.glob call to modeldir of case to be restored,
               with history.csv as target
    """

    if len(histglob) != 1:
        logging.critical(histglob)
        raise ValueError("Found mor than one historyfile in Modeldir!")
    else:
        with open(histglob[0]) as histfile:
            df = pd.read_csv(histfile, delimiter=",", header=0, skipinitialspace=True)
            df = df.iloc[1:]
            p = df.to_dict("series")

        return p["Epoch"].tail(1).values[0]


def get_modelname(confglob):
    """
    Read config file of case to be restored and parse Modeltype
    @confglob: glob.glob call to modeldir of case to be restored,
               with *.config as target
    """
    if len(confglob) != 1:
        logging.critical(confglob)
        raise ValueError("Found mor than one historyfile in Modeldir!")
    else:
        with open(confglob[0]) as conffile:
            c = conffile.readlines()

        for r in c:
            if "Modeltype:" in r:
                modelname = (r.strip().split(":"))[-1]
    return modelname


def point2VTK(outputfilepath, pointData, xcoords, ycoords, dtype=np.float32):
    """export pointcloud as vtk"""
    from pyevtk.hl import pointsToVTK

    z = np.zeros(np.shape(xcoords), dtype=dtype)

    pointsToVTK(outputfilepath, np.array(xcoords), np.array(ycoords), z, data=pointData)

"""
Evaluation of arbitrary datasets
"""
import sys
import os
import glob
import logging
import numpy as np
import torch
import time
import json
from datetime import datetime
import argparse
from torch_geometric.data import DataLoader
from util import getModelbyName, point2VTK, setup_logger
from torchGeom_ptCloud import torchdataset
from normalize import MinMaxScaler


def exportOutput(outputfilepath, INPUT, OUTPUT, y_t):
    """
    Take (rescaled) INPUT, OUTPUT, y_t and dump to vtk
    """

    pointData = {}
    pointData["x"] = np.array(INPUT[:, 0])
    pointData["y"] = np.array(INPUT[:, 1])

    if INPUT.shape[-1] == 3:
        pointData["airfoil"] = np.array(INPUT[:, -1])
    elif (INPUT.shape[-1] == 5) or (INPUT.shape[-1] == 4):
        pointData["in_xvelo"] = np.array(INPUT[:, 2])
        pointData["AoA"] = np.array(INPUT[:, 3])

    elif not (INPUT.shape[-1] == 2):
        print("Got {}".format(INPUT.shape[-1]))
        raise NotImplementedError("Unrecognized Shape!")

    pointData["xvelo"] = np.array(OUTPUT[:, 0])
    pointData["yvelo"] = np.array(OUTPUT[:, 1])
    pointData["p"] = np.array(OUTPUT[:, 2])
    pointData["rho"] = np.array(OUTPUT[:, 3])

    pointData["xvelo_t"] = np.array(y_t[:, 0])
    pointData["yvelo_t"] = np.array(y_t[:, 1])
    pointData["p_t"] = np.array(y_t[:, 2])
    pointData["rho_t"] = np.array(y_t[:, 3])

    pointData["d2xvelo"] = (pointData["xvelo"] - pointData["xvelo_t"]) ** 2
    pointData["d2yvelo"] = (pointData["yvelo"] - pointData["yvelo_t"]) ** 2
    pointData["d2p"] = (pointData["p"] - pointData["p_t"]) ** 2
    pointData["drho"] = (pointData["rho"] - pointData["rho_t"]) ** 2

    L2Norm_Vx = np.sqrt(np.sum((pointData["xvelo_t"] - pointData["xvelo"]) ** 2))
    L2Norm_Vy = np.sqrt(np.sum((pointData["yvelo_t"] - pointData["yvelo"]) ** 2))
    L2Norm_p = np.sqrt(np.sum((pointData["p_t"] - pointData["p"]) ** 2))
    L2Norm_rho = np.sqrt(np.sum((pointData["rho_t"] - pointData["rho"]) ** 2))

    point2VTK(outputfilepath, pointData, pointData["x"], pointData["y"])

    return np.array([L2Norm_Vx, L2Norm_Vy, L2Norm_p, L2Norm_rho])


def get_configs(evaldir):
    conf = glob.glob((os.sep).join([evaldir, "*config"]))
    if len(conf) != 1:
        raise TypeError('No configuration file (*.config) in "{}"'.format(evaldir))
        exit(1)
    else:
        with open(conf[0]) as conffile:
            c = conffile.readlines()

        for r in c:
            if "Modeltype" in r:
                modelname = (r.strip().split(" "))[-1]
                # Take first appearanc
                break

    return modelname


def predict(evaldir, testset, args, dumpevaldir):
    """predict single sample"""
    modelname = get_configs(evaldir)
    scaler = MinMaxScaler(evaldir)
    scaler.restore()  # Restore Scaler from evaldir

    testset = scaler.transformData(testset)
    if len(testset) < 1:
        raise ValueError("Check Datadir! No Samples imported!")
    indim, outdim = testset[0].x.shape[-1], testset[0].y.shape[-1]

    model = getModelbyName(modelname, indim, outdim)

    cpath = evaldir + "checkpoint.pth"
    logging.info("Load Model from checkpoint {}".format(cpath))
    checkpoint = torch.load(cpath, map_location=device)
    model = checkpoint["model"]
    model.load_state_dict(checkpoint["model_state_dict"])

    # ResultLogger
    Reslog = setup_logger(
        "ResultLogger",
        dumpevaldir + os.sep + "predictions.csv",
        level=logging.DEBUG,
    )

    Reslog.debug("SampleNumber, Loss, L2Norm Vx, L2Norm Vy, L2Norm p")

    print("Predicting... \n\n")
    criterion = torch.nn.MSELoss()
    testloader = DataLoader(testset, shuffle=False, batch_size=1, num_workers=1)
    numtest = len(testloader.dataset)

    L2Norms = np.zeros((4, numtest))

    t = time.time()
    test_loss = []
    model.eval()
    t = time.time()
    for step, data in enumerate(testloader):
        x = data.x.to(device, non_blocking=True)
        adj_t = data.adj_t.to(device, non_blocking=True)
        y = data.y.to(device, non_blocking=True)
        batch = data.batch.to(device)
        H = model(x, adj_t, batch)  # Forward pass.
        loss = criterion(H, y)  # Loss computation.

        x, H, y = scaler.denormalize(x, H, y)

        if hasattr(data, "params"):
            par = data.params[0]
            print(par)
            outputfilepath = (
                dumpevaldir
                + os.sep
                + "DP{}_{}_Sample{}_AoA{}_VeloXIn{:.4f}_{:.6f}".format(
                    par["dp"],
                    par["obj"],
                    step,
                    par["AoA"],
                    par["XVelo"],
                    loss.cpu().detach().numpy(),
                )
            )
        else:
            outputfilepath = (
                dumpevaldir
                + os.sep
                + "Sample{}_{:.4f}".format(step, loss.cpu().detach().numpy())
            )
        res = exportOutput(
            outputfilepath,
            x.cpu().detach().numpy(),
            H.cpu().detach().numpy(),
            y.cpu().detach().numpy(),
        )

        L2Norms[:, step] = res

        Reslog.debug(
            "{},{},{},{},{},{}".format(
                step,
                loss,
                L2Norms[0, step],
                L2Norms[1, step],
                L2Norms[2, step],
                L2Norms[3, step],
            )
        )
        test_loss.append(loss.item())

    epoch_testloss = np.mean(test_loss)  # len(list(testloader))
    logging.info(
        "Test Loss ({:.2f}s): Avg = {:.5}, Min = {:.5}, Max = {:.5}".format(
            time.time() - t, epoch_testloss, np.min(test_loss), np.max(test_loss)
        )
    )

    logging.info(
        "L2Norm Vx: Avg = {}, Max = {}, Min = {}".format(
            np.mean(L2Norms[0, :]), np.max(L2Norms[0, :]), np.min(L2Norms[0, :])
        )
    )
    logging.info(
        "L2Norm Vy: Avg = {}, Max = {}, Min = {}".format(
            np.mean(L2Norms[1, :]), np.max(L2Norms[1, :]), np.min(L2Norms[1, :])
        )
    )
    logging.info(
        "L2Norm p: Avg = {}, Max = {}, Min = {}".format(
            np.mean(L2Norms[2, :]), np.max(L2Norms[2, :]), np.min(L2Norms[2, :])
        )
    )
    logging.info(
        "L2Norm rho: Avg = {}, Max = {}, Min = {}".format(
            np.mean(L2Norms[3, :]), np.max(L2Norms[3, :]), np.min(L2Norms[3, :])
        )
    )

    logging.info("Prediction was dumped to {}".format(dumpevaldir))


if __name__ == "__main__":

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    root.addHandler(handler)

    # Get Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--modeldir",
        help="Path to model that is to \
                                    be evaluated",
        type=str,
        required=True,
    )
    ap.add_argument(
        "-d",
        "--dataset",
        help="Path to UNPREPARED dataset.",
        type=str,
        required=False,
        default="NULL",
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
        "-s",
        "--source",
        help="Path to *.npz files containing \
                output of 01_GraphConv/assemble_ptcloud.py",
        type=str,
        required=True,
    )
    ap.add_argument(
        "--loadFreshfromDataset",
        help='Restore only filenames and load files directly"',
        action="store_true",
        required=False,
        default=False,
    )

    args = ap.parse_args()

    evaldir = args.modeldir
    if not evaldir[-1] == os.sep:
        evaldir = evaldir + os.sep

    dumpevaldir = evaldir + "Predict_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    os.mkdir(dumpevaldir)

    if not args.dataset == "NULL":
        datasetpath = args.dataset
        logging.info("Load {} ...".format(datasetpath))

        dat1 = torch.load(datasetpath, map_location=torch.device("cpu"))
        if args.loadFreshfromDataset:
            logging.info(
                "Imported Validationset filenames from: {}".format(datasetpath)
            )
            d1 = torchdataset(
                args.source,
                args.idents,
                N=args.Npts,
                splitarr=[1.0],
                onlyTest=True,
                mdir=dumpevaldir,
                nameArr=[dat1["filenames"][-1]],
            )
            testset = d1.data

        else:
            [_, _, testset] = dat1["data"]
            logging.info("Imported Dataset: {}".format(datasetpath))
            logging.debug("Rootdir of Dataset: {}".format(dat1["rootdir"]))

        with open(dumpevaldir + os.sep + "filelist.txt", "w") as s:
            s.write("## SET 3 - Validation\n")
            for i, f in enumerate(dat1["filenames"][-1]):
                s.write("{}, {}\n".format(i, f))
    else:
        # Load Testset
        d1 = torchdataset(
            args.source, args.idents, splitarr=[1.0], onlyTest=True, mdir=dumpevaldir
        )
        testset = d1.data

    logging.info(
        'Evaluate Model "{}" on {} samples of validation set'.format(
            evaldir, len(testset)
        )
    )

    predict(evaldir, testset, args, dumpevaldir)

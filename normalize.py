"""
MinMax Scaler
"""
import os
import torch
import numpy as np
from torch_geometric.data import Data
import logging
from shutil import copyfile


class MinMaxScaler(object):
    """Normalizes Output to range `{[0, 1]}`
    Input to [0,1] if requested
    """

    def __init__(self, modeldir):

        self.scaler = {}
        self.modeldir = modeldir
        self.NormInput = False

    def restore(self, scalerpath=None, logger=logging, dump=False):

        if scalerpath is None:
            scalerpath = self.modeldir + "MinMaxScaler.pth"
        self.scalerpath = scalerpath
        logger.debug("MinMaxScaler from : {}".format(self.scalerpath))
        scalerdict = torch.load(self.scalerpath)

        for k, v in scalerdict["scaler"].items():
            logger.debug(k)
            self.scaler[k] = v
            logger.debug(v)
            if k == "min_in":
                self.NormInput = True
        if dump:
            copyfile(self.scalerpath, self.modeldir + os.sep + "MinMaxScaler.pth")
            self.scalerpath = self.modeldir + os.sep + "MinMaxScaler.pth"

    def transformData(
        self,
        dataset,
        new=False,
        scalerpath=None,
        logger=logging,
        NormInput=None,
        device="cpu",
    ):
        if new:
            self.create_scaler(dataset, NormInput)
            if scalerpath is not None:
                self.scalerpath = scalerpath + ".pth"
                logger.debug("New MinMaxScaler : {}".format(self.scalerpath))
                torch.save({"scaler": self.scaler}, self.scalerpath)

        if NormInput is None:
            NormInput = self.NormInput

        tDataset = []
        k = 0
        print("Got NormInput : {}".format(NormInput))
        for d in dataset:
            pos = d.pos
            x = d.x
            y = d.y

            if NormInput:
                # Only scale feature matrix
                for i in range(self.scaler["num_features"]):
                    x[:, i] = (d.x[:, i] - self.scaler["min_in"][i]) / (
                        self.scaler["max_in"][i] - self.scaler["min_in"][i]
                    )

            for i in range(self.scaler["outputDims"]):
                y[:, i] = (d.y[:, i] - self.scaler["min_out"][i]) / (
                    self.scaler["max_out"][i] - self.scaler["min_out"][i]
                )

            sdat = Data(
                pos=pos, x=x, adj_t=d.adj_t, y=y, num_nodes=d.num_nodes, device=device
            )

            if d.edge_attr is not None:
                sdat.edge_attr = d.edge_attr

            if hasattr(d, "params"):
                sdat.params = d.params

            tDataset.append(sdat)
            k += 1

        logging.info("Transformed {} samples with scaler {}".format(k, self.scalerpath))
        del dataset
        return tDataset

    def denormalize(self, X, Y, y_t, NormInput=None):

        if NormInput is None:
            NormInput = self.NormInput
        # Don't scale Position as they are provided as well in X

        if NormInput:
            for i in range(self.scaler["num_features"]):
                X[:, i] = (
                    X[:, i] * (self.scaler["max_in"][i] - self.scaler["min_in"][i])
                ) + self.scaler["min_in"][i]

        for i in range(self.scaler["outputDims"]):

            Y[:, i] = (
                Y[:, i] * (self.scaler["max_out"][i] - self.scaler["min_out"][i])
            ) + self.scaler["min_out"][i]

            y_t[:, i] = (
                y_t[:, i] * (self.scaler["max_out"][i] - self.scaler["min_out"][i])
            ) + self.scaler["min_out"][i]

        return X, Y, y_t

    def create_scaler(self, dataset, NormInput):

        self.scaler["scalertype"] = "MINMAX Scaler"
        self.scaler["dims"] = dataset[0].pos.shape[-1]
        self.scaler["num_features"] = dataset[0].x.shape[-1]
        self.scaler["outputDims"] = dataset[0].y.shape[-1]

        self.scaler["min_out"] = np.ones(self.scaler["outputDims"]) * np.inf
        self.scaler["max_out"] = np.zeros(self.scaler["outputDims"])

        if NormInput:
            self.scaler["min_in"] = np.ones(self.scaler["num_features"]) * np.inf
            self.scaler["max_in"] = np.zeros(self.scaler["num_features"])

            for d in dataset:
                for i in range(self.scaler["num_features"]):
                    if (d.x[:, i]).min() < self.scaler["min_in"][i]:
                        self.scaler["min_in"][i] = (d.x[:, i]).min()
                    if (d.x[:, i]).max() > self.scaler["max_in"][i]:
                        self.scaler["max_in"][i] = (d.x[:, i]).max()

        for d in dataset:
            for i in range(self.scaler["outputDims"]):
                if (d.y[:, i]).min() < self.scaler["min_out"][i]:
                    self.scaler["min_out"][i] = (d.y[:, i]).min()
                if (d.y[:, i]).max() > self.scaler["max_out"][i]:
                    self.scaler["max_out"][i] = (d.y[:, i]).max()

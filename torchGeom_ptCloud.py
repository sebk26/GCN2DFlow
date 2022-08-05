import os
import numpy as np
from pathlib import Path
import torch

# from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_cluster import knn_graph
import logging
from datetime import datetime


class torchdataset(object):
    def __init__(
        self,
        rootdir,
        sampleidents,
        splitarr=[0.8, 0.1, 0.1],
        wi=False,
        onlyTest=False,
        save2disk=True,
        dump=False,
        mdir=None,
        k=6,
        nameArr=[],
    ):

        self.data = self.npz2torchData(
            rootdir,
            sampleidents,
            splitarr=splitarr,
            wi=wi,
            onlyTest=onlyTest,
            save2disk=save2disk,
            dump=dump,
            mdir=mdir,
            k=k,
            name_arr=nameArr,
        )

    def npz2torchData(
        self,
        rootdir,
        sampleidents,
        splitarr=[0.8, 0.1, 0.1],
        wi=False,
        onlyTest=False,
        save2disk=True,
        dump=False,
        mdir=None,
        k=6,
        name_arr=[],
    ):
        """
        @rootdir     : directory to look recursively for *npz filelist
        @sampleidents: identifier included in filename
        @splitarr    : split into different sets [0.8,0.1,0.1] would split into
                       three sets. Must add up to 1! [0.8,0.2] also works.
        @wi          : with_inds, Take adjacency index list from mesh (boolean, default=False)
                       If false, Nearest Neighbor algorithm is applied
        @mdir        : Model directory to dump filelist for later reconstruction

        return       : list of pytorch_geometric Data objects
        """
        if len(name_arr) < 1:
            name_arr = self.get_filenames(
                rootdir, sampleidents, splitarr, onlyTest=onlyTest
            )
        global it_name
        it_name = 0
        if mdir is not None:
            logging.info(
                'Dump list of filenames to modeldir "{}"'.format(mdir + os.sep)
            )
            with open(mdir + os.sep + "filelist.txt", "w") as s:
                s.write(
                    "# Log of used files from Run on {}\n".format(
                        datetime.now().strftime("%Y%m%d-%H%M%S")
                    )
                )
                for i, set in enumerate(name_arr):
                    s.write("## SET {}\n".format(i))
                    for i, f in enumerate(set):
                        s.write("{}, {}\n".format(i, f))
        logging.info("Compile Datasets ...")

        if onlyTest:
            return [self._npz2Data(f, wi=wi, dump=dump, k=k) for f in name_arr[-1]]
        else:
            dataset = [
                [self._npz2Data(f, wi=wi, dump=dump, k=k) for f in s] for s in name_arr
            ]

            print("Dump to disk...")
            if save2disk:
                torch.save(
                    {
                        "data": dataset,
                        "filenames": name_arr,
                        "rootdir": rootdir,
                    },
                    "Dataset_{}.torch".format(datetime.now().strftime("%Y%m%d-%H%M%S")),
                )

            return dataset

    def _npz2Data(self, filename, wi=True, dump=False, k=6):
        """
        @filename : filename of *npz data files
        @wi       : with_inds, take adjacency matrix of mesh or not
        @N        :  Number of Points taken from subset. Default = 0 > Take all!
        @dump     : Dump sample to vtk
        return    : Sparse pytorch_geometric Data object
        """
        global it_name
        it_name += 1
        if (it_name % 250) == 0:
            print("Process Sample " + str(it_name))
        with np.load(filename, allow_pickle=True) as d:
            # Take complete domain (exclude obj channel (last channel))
            dp_in = torch.tensor(d["INPUT"][:, :-1], dtype=torch.float)
            N = dp_in.size()[0]
            numpts = torch.tensor(N, dtype=torch.int)
            dp_out = torch.tensor(d["OUTPUT"][:, :], dtype=torch.float)
            if hasattr(d, "params"):
                # Get Metadata dict and add sample identifier
                obj = (filename.split(os.sep)[-1]).split("_")[-1].split(".")[0]
                params = d["params"][0].item()
                params["obj"] = obj
                par = True
            elif (dp_in.shape[-1] == 5) or (dp_in.shape[-1] == 4):
                # Build Metadata dict
                params = {}
                dp = ((filename.split(os.sep)[-1]).split("_")[0]).split("dp")[-1]
                obj = (filename.split(os.sep)[-1]).split("_")[-1].split(".")[0]
                params["AoA"] = torch.mean(dp_in[:, 3]).item()
                params["XVelo"] = torch.mean(dp_in[:, 2]).item()
                params["dp"] = dp
                params["obj"] = obj
                par = True
            else:
                par = False

        # dp_in ONLY COORDINATES
        # ASSUMPTION: Only 2D cases
        if wi:
            return T.ToSparseTensor()(
                Data(
                    pos=dp_in[:, :2],
                    x=dp_in,
                    edge_index=d["indlst"],
                    y=dp_out,
                    num_nodes=numpts,
                )
            )

        else:
            if (it_name % 250) == 0:
                print("Assumption: 2D. Take first two channels.")
            edge_index = knn_graph(dp_in[:, :2], k=k, num_workers=3)
            d = Data(
                pos=dp_in[:, :2],
                x=dp_in,
                edge_index=edge_index,
                y=dp_out,
                num_nodes=numpts,
            )
            D = T.ToSparseTensor(False)(d)
            if par:
                D.params = params
        return D

    def get_filenames(
        self, rootdir, sampleidents, splitarr,  onlyTest=False
    ):
        """
        @rootdir      : directory path to ptCloud *npz files
        @sampleidents : identifier that are concluded in sample filename
        @splitarr     : split into different sets [0.8,0.1,0.1] would split into
                        three sets. Must add up to 1! [0.8,0.2] also works.
        return        : list of arbitrary number of sets of filenames
        """

        filelist = sorted(list(Path(rootdir).rglob("*.npz")))

        # Check if splitarr percents add up to 1
        if not np.sum(splitarr) == 1:
            raise ValueError("Splitarr entries have to add up to 1")
            exit(1)

        # First collect all names
        # identdict = {}
        names = []
        for ident in sampleidents:
            names.extend(
                [
                    item.as_posix()
                    for item in filelist
                    if ident in str(item.as_posix().split(os.sep)[-1])
                ]
            )

        logging.info("Found {} samples that match identifiers.".format(len(names)))
        splits = [int(len(names) * p) for p in splitarr[:-1]]

        if len(splits) > 1:
            splits[1:] = [splits[0] + k for k in splits[1:]]

        np.random.seed(42)
        inds = np.arange(len(names))

        # Randomize
        if not onlyTest:
            np.random.shuffle(inds)

        # Split into as many arrays as needed
        sets = np.split(inds, splits)

        logging.info(
            "Split into {} arrays ".format(len(splitarr))
            + "with {} samples or ".format([len(s) for s in sets])
            + "{}".format([np.round((len(s) / len(names)) * 100, 2) for s in sets])
            + " percent of all samples."
        )

        return [np.array(names)[sorted(s)] for s in sets]


def debug_dump(d_in, d_out, choice):
    global it_name
    from pyevtk.hl import pointsToVTK

    pointData = {}
    pointData["x"] = np.array(d_in[:, 0], dtype=np.float32)
    pointData["y"] = np.array(d_in[:, 1], dtype=np.float32)
    pointData["object"] = np.array(d_in[:, -1])

    choice_arr = np.zeros(np.shape(pointData["x"]), dtype=np.float32)
    choice_arr[choice] = 1.0
    pointData["choice"] = choice_arr

    pointData["xvelo"] = np.array(d_out[:, 0])
    pointData["yvelo"] = np.array(d_out[:, 1])
    pointData["p"] = np.array(d_out[:, 2])

    choice_arr2 = np.empty(np.shape(d_out))
    choice_arr2[:] = np.NaN
    choice_arr2[choice, 0] = np.array(d_out[choice, 0])
    choice_arr2[choice, 1] = np.array(d_out[choice, 1])
    choice_arr2[choice, 2] = np.array(d_out[choice, 2])

    pointData["xvelo_choice"] = np.array(choice_arr2[:, 0])
    pointData["yvelo_choice"] = np.array(choice_arr2[:, 1])
    pointData["p_choice"] = np.array(choice_arr2[:, 2])

    z = np.zeros(np.shape(pointData["x"]), dtype=np.float32)
    print("Dump debug_out{}.vtu".format(it_name))
    pointsToVTK(
        "./debug_out{}".format(it_name),
        np.array(pointData["x"]),
        np.array(pointData["y"]),
        z,
        data=pointData,
    )

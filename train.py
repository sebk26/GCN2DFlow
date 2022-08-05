"""
Main training script
"""
import os
import sys
import torch
import logging
import time
from pathlib import Path
from util import create_Argparse, evalArgparse, gnn_model_summary, setup_logger
from util import compile_checkpoint


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train(args):
    """train GCN model"""
    min_testerr = 0.0
    new_ch = False
    save_e = 0
    ch_e = 0
    end_now = False

    (
        trainloader,
        testloader,
        configLog,
        startepoch,
        endepoch,
        model,
        modeldir,
        device,
        optimizer,
    ) = evalArgparse(args)

    # HistoryLogger
    histLog = setup_logger(
        "historyLog",
        args.modeldir + os.sep + "history.csv",
        level=logging.DEBUG,
    )
    if not startepoch > 0:
        histLog.debug("Epoch, Train-Metrik, Test-Metrik, LR")

    criterion = torch.nn.MSELoss().to(device)

    configLog.debug("Optimizer : " + str(optimizer))
    configLog.debug("Loss function : " + str(criterion))
    gnn_model_summary(model, configLog)

    with open(modeldir + os.sep + "delete2terminate", "w") as s:
        s.write("# Delete this file to save & terminate the current run" + os.linesep)
    logging.info("Will use device: {}".format(device))

    numtrain = len(trainloader.dataset)
    numtest = len(testloader.dataset)
    lr = get_lr(optimizer)
    print("Initial LR: {}".format(lr))
    model.to(device)
    for e in range(startepoch, endepoch):
        logging.info("\nEpoch {}/{}".format(e + 1, endepoch))
        total_loss = 0
        model.train()
        t = time.time()
        for step, data in enumerate(trainloader):

            x = data.x.to(device, non_blocking=True)
            batch = data.batch.to(device, non_blocking=True)
            adj_t = data.adj_t.to(device)
            y = data.y.to(device, non_blocking=True)
            H = model(x, adj_t, batch)  # Forward pass.
            loss = criterion(H, y)  # Loss computation.
            optimizer.zero_grad()  # Clear gradients.
            loss.backward()  # Backward pass.

            optimizer.step()  # Update model parameters.
            total_loss += loss.item() * data.num_graphs

        epoch_trainloss = total_loss / numtrain
        logging.info(
            "Train Loss ({:.2f}s): {:.8}".format(time.time() - t, epoch_trainloss)
        )

        if e % 1000 == 0:
            test_loss = 0
            model.eval()
            t = time.time()
            for step, data in enumerate(testloader):
                x = data.x.to(device, non_blocking=True)
                adj_t = data.adj_t.to(device, non_blocking=True)
                batch = data.batch.to(device, non_blocking=True)
                y = data.y.to(device, non_blocking=True)

                H = model(x, adj_t, batch)  # Forward pass.
                loss = criterion(H, y)  # Loss computation.
                test_loss += loss.item() * data.num_graphs

            epoch_testloss = test_loss / numtest
            logging.info(
                "Test Loss ({:.2f}s): {:.8f}, Checkpoint (Epoch {}): {:.8f}".format(
                    time.time() - t, epoch_testloss, ch_e + 1, min_testerr
                )
            )

            histLog.debug(
                "{}, {}, {}, {}".format(e + 1, epoch_trainloss, epoch_testloss, lr)
            )
            if e == startepoch or epoch_testloss < min_testerr:
                min_testerr = epoch_testloss
                checkpoint = compile_checkpoint(model, optimizer, e)
                new_ch = True
                ch_e = e

        if not Path(modeldir + os.sep + "delete2terminate").is_file():
            end_now = True
            logging.warning(
                '\n"delete2terminate" was deleted. Will Save and Terminate NOW!'
            )

        if (
            e == startepoch
            or (e - save_e > args.sinterval and new_ch is True)
            or end_now
            or (e + 1 == endepoch and new_ch is True)
        ):
            ch = modeldir + os.sep + "checkpoint.pth"
            logging.info(
                "Checkpoint of Epoch {} --> {}".format(checkpoint["epoch"] + 1, ch)
            )
            torch.save(checkpoint, ch)

            new_ch = False
            save_e = ch_e + 1

        if end_now:
            break

        if lr != get_lr(optimizer):
            lr = get_lr(optimizer)
            logging.info("\n\t\t <Change in Learning Rate>\nNew LR: {}\n".format(lr))

    if Path(modeldir + os.sep + "delete2terminate").is_file():
        os.remove(modeldir + os.sep + "delete2terminate")
    logging.info("###### TRAINING ENDED ######\n\n\n")


if __name__ == "__main__":

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    ap = create_Argparse()
    args = ap.parse_args()
    root.addHandler(handler)

    train(args)

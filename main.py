import torch
import torch.nn.functional as F
import dgl.data
from network import GCN
import argparse
from utils import load_data, coarsening

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--coarsening_ratio', type=float, default=0.5)
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')
    args = parser.parse_args()

    dataset = dgl.data.CoraGraphDataset()
    g = dataset[0]

    model = GCN(g.ndata['feat'].shape[1], args.hidden, dataset.num_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    candidate, C_list, Gc_list = coarsening(g, 1 - args.coarsening_ratio, args.coarsening_method)

    coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_row, coarsen_col = load_data(
        g, candidate, C_list, Gc_list, dataset.num_classes)

    coarsen_g = dgl.DGLGraph((coarsen_row, coarsen_col))

    for e in range(args.epochs):
        model.train()
        # Forward
        logits = model(coarsen_g, coarsen_features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[coarsen_train_mask], coarsen_train_labels[coarsen_train_mask])

        # Compute accuracy on training/validation/test
        # train_acc = (pred[coarsen_train_mask] == coarsen_train_labels[coarsen_train_mask]).float().mean()
        # val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        # test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        # if best_val_acc < val_acc:
        # best_val_acc = val_acc
        # best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # if e % 5 == 0:
        # print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
        # e, loss, val_acc, best_val_acc, test_acc, best_test_acc))

        print('In epoch {}, test acc: {:.3f} '.format(e, test_acc))
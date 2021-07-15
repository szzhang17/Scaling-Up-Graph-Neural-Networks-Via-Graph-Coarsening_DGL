import torch
import torch.nn.functional as F
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from networks import APPNP
import argparse
from utils import load_data, coarsening
import dgl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--coarsening_ratio', type=float, default=0.5)
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')
    args = parser.parse_args()

    if args.dataset == 'cora':
        dataset = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        dataset = PubmedGraphDataset()
    g = dataset[0]

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    candidate, C_list, Gc_list = coarsening(g, 1 - args.coarsening_ratio, args.coarsening_method)
    coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_row, coarsen_col = load_data(
        g, candidate, C_list, Gc_list, dataset.num_classes)
    coarsen_g = dgl.graph((coarsen_row, coarsen_col))

    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    coarsen_g = dgl.remove_self_loop(coarsen_g)
    coarsen_g = dgl.add_self_loop(coarsen_g)

    if args.normalize_features:
        coarsen_features = F.normalize(coarsen_features, p=1)
        features = F.normalize(features, p=1)

    model = APPNP(g.ndata['feat'].shape[1], args.hidden, dataset.num_classes, args.K, args.alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for e in range(args.epochs):
        model.train()
        # Forward
        logits = model(coarsen_g, coarsen_features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[coarsen_train_mask], coarsen_train_labels[coarsen_train_mask])

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        # Compute accuracy on training/validation/test
        logits = model(coarsen_g, coarsen_features)
        pred = logits.argmax(1)
        val_acc = (pred[coarsen_val_mask] == coarsen_val_labels[coarsen_val_mask]).float().mean()

        logits = model(g, features)
        pred = logits.argmax(1)
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        print('In epoch {}, val acc: {:.3f}, test acc: {:.3f} '.format(e, val_acc, test_acc))
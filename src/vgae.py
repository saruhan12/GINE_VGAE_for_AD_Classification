import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.nn import Linear, Sequential, ReLU, Dropout
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GINEConv, VGAE
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import skew, kurtosis, entropy
from torch.nn.utils import clip_grad_norm_

# Encoder using GINEConv layers for edge-aware message passing
class GINEVGAEEncoder(nn.Module):
    def __init__(self, in_c, hid_c, lat_c, edge_dim, dropout_p=0.2):
        super().__init__()
        # First GINEConv maps input features to hidden dimension, using edge attributes
        self.gine1 = GINEConv(Sequential(Linear(in_c, hid_c), ReLU(), Linear(hid_c, hid_c)), edge_dim=edge_dim)
        # Dropout to prevent overfitting
        self.dropout   = Dropout(dropout_p)
        self.gine_mu     = GINEConv(Sequential(Linear(hid_c, lat_c), ReLU(), Linear(lat_c, lat_c)), edge_dim=edge_dim)
        self.gine_logstd = GINEConv(Sequential(Linear(hid_c, lat_c), ReLU(), Linear(lat_c, lat_c)), edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr):
        # Apply first GINEConv + ReLU + dropout to get hidden node features
        x = F.relu(self.gine1(x, edge_index, edge_attr))
        x = self.dropout(x)
        # Compute means and log‐std for the variational distribution
        mu     = self.gine_mu(x, edge_index, edge_attr)
        logstd = self.gine_logstd(x, edge_index, edge_attr)
        return mu, torch.clamp(logstd, -3.0, 3.0)

# VGAE models that override encode to accept edge attributes
class GINEVGAE(VGAE):
    def encode(self, x, edge_index, edge_attr):
        mu, logstd = self.encoder(x, edge_index, edge_attr)
        self.__mu__, self.__logstd__ = mu, logstd
        return mu, logstd
    
# Convert networkx graphs into PyG objects
def nx_to_pyg(nx_graphs, labels, use_edges=True):
    pyg = []
    for i, G in enumerate(nx_graphs):
        x = torch.tensor(
            [G.nodes[n]['mean_activation'] for n in G.nodes()],
            dtype=torch.float
        )
        ei = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()

        if use_edges:
            weights = []
            for u, v, d in G.edges(data=True):
                w = d.get('weight', 1.0)
                weights.append([w])
            ea = torch.tensor(weights, dtype=torch.float, device=x.device)
        else:
            ea = None

        pyg.append(Data(x=x, edge_index=ei, edge_attr=ea, y=torch.tensor([labels[i]])))
    return pyg


def safe_recon_loss(z, pos_ei, neg_ei, clamp_val=2.0):
    device = z.device

    # Positive edges
    hi, hj = pos_ei.long()
    if hi.numel() == 0:
        pos_loss = torch.tensor(0.0, device=device)
    else:
        logits_pos = (z[hi] * z[hj]).sum(dim=1).clamp(-clamp_val, clamp_val)
        pos_loss   = F.binary_cross_entropy_with_logits(
                        logits_pos,
                        torch.ones_like(logits_pos),
                        reduction='mean'
                     )

    # Negative edges
    hi_n, hj_n = neg_ei.long()
    if hi_n.numel() == 0:
        neg_loss = torch.tensor(0.0, device=device)
    else:
        logits_neg = (z[hi_n] * z[hj_n]).sum(dim=1).clamp(-clamp_val, clamp_val)
        neg_loss   = F.binary_cross_entropy_with_logits(
                        logits_neg,
                        torch.zeros_like(logits_neg),
                        reduction='mean'
                     )

    return pos_loss + neg_loss

# Pool node embeddings into a single graph‐level vector
def pool_node_embeddings(z):
    Z = z.cpu().detach().numpy()
    comps = [
        Z.mean(axis=0),
        Z.max(axis=0),
        Z.min(axis=0),
        np.median(Z, axis=0),
        Z.max(axis=0) - Z.min(axis=0),
        skew(Z, axis=0),
        kurtosis(Z, axis=0),
        np.percentile(Z,75,axis=0) - np.percentile(Z,25,axis=0),
        entropy(Z/(Z.sum(axis=0,keepdims=True)+1e-8)+1e-8, axis=0),
        np.array([np.linalg.norm(Z,axis=1).mean()])
    ]
    return np.concatenate([np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0) for c in comps])


# Pretrain GINE‐VGAE on Cora to initialize weights
def pretrain_gine_vgae_on_cora(save_path, hidden_channels=64, latent_dim=32, pretrain_epochs=50, lr=1e-3, weight_decay=1e-5, device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load Cora
    ds = Planetoid(root=".", name="Cora")
    data = ds[0].to(device)
    E = data.edge_index.size(1)
    # dummy edge_attr = 1.0 to train our model.
    data.edge_attr = torch.ones((E,1), device=device)

    # build model
    enc   = GINEVGAEEncoder(ds.num_node_features, hidden_channels, latent_dim, edge_dim=1)
    model = GINEVGAE(enc).to(device)
    opt   = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for ep in range(1, pretrain_epochs+1):
        opt.zero_grad()
        mu, logstd = model.encode(data.x, data.edge_index, data.edge_attr)
        z = model.reparametrize(mu, logstd)
        pos_e = data.edge_index
        neg_e = negative_sampling(pos_e, num_nodes=data.num_nodes, num_neg_samples=E//4).to(device)
        loss = safe_recon_loss(z, pos_e, neg_e) + (1/data.num_nodes)*model.kl_loss(mu, logstd)
        loss.backward()
        opt.step()
        print(f"[Cora Pretrain {ep:02d}] Loss {loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved pretrained weights → {save_path}")


# Full training and embedding extraction on PET graphs
def get_graph_embeddings_vgae(nx_graphs, labels, hidden_channels=64, latent_dim=32,
                             epochs=100, batch_size=1, lr=5e-5, weight_decay=1e-4, encoder_type="gine", 
                             pretrained_weights_path=None, val_frac=0.1, kl_max=0.5, early_stop_patience=10):
    use_edges = (encoder_type=="gine")
    pyg = nx_to_pyg(nx_graphs, labels, use_edges=use_edges)

    # Train/val split
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(
        pyg, test_size=val_frac,
        random_state=42, stratify=labels
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False)

    in_c    = pyg[0].x.shape[1]
    edge_dim = pyg[0].edge_attr.shape[1] if use_edges else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if encoder_type=="gine":
        enc   = GINEVGAEEncoder(in_c, hidden_channels, latent_dim, edge_dim)
        model = GINEVGAE(enc).to(device)

    # Load pretrained if given
    if pretrained_weights_path:
        checkpoint = torch.load(pretrained_weights_path, map_location=device)
        sd = model.state_dict()
        for k,v in checkpoint.items():
            if k in sd and sd[k].shape==v.shape:
                sd[k] = v
        model.load_state_dict(sd, strict=False)
        print(f"Loaded {len(checkpoint)} pretrained parameters")

    # Optimizer and scheduler
    opt   = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = ReduceLROnPlateau(opt, mode='min', factor=0.5,
                              patience=5, verbose=True)


    # Training loop with KL annealing and early stopping
    total_steps = epochs * len(train_loader)
    beta, beta_step = 0.0, kl_max / total_steps

    best_val_loss = float('inf')
    no_imp = 0

    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            batch = batch.to(device)
            x  = batch.x
            ei = batch.edge_index #edge index
            ea = batch.edge_attr if use_edges else None #edge attributes

            opt.zero_grad()
            if use_edges:
                mu, logstd = model.encode(x, ei, ea)
            else:
                mu, logstd = model.encode(x, ei)
            z = model.reparametrize(mu, logstd)
            # Sample negative edges
            pos_e = ei
            neg_e = negative_sampling(pos_e,
                                      num_nodes=x.size(0),
                                      num_neg_samples=pos_e.size(1)//4).to(device)
            pos_e = pos_e.long()
            neg_e = neg_e.long()

            # Compute loss and update
            recon = safe_recon_loss(z, pos_e, neg_e, clamp_val=2.0)
            kl    = model.kl_loss(mu, logstd) / x.size(0)
            loss  = recon + beta * kl

            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_loss += loss.item()
            beta = min(kl_max, beta + beta_step)

        avg_train = train_loss / len(train_loader)

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                if use_edges:
                    mu, logstd = model.encode(batch.x,
                                              batch.edge_index,
                                              batch.edge_attr)
                else:
                    mu, logstd = model.encode(batch.x,
                                              batch.edge_index)
                z = model.reparametrize(mu, logstd)

                pos_e = batch.edge_index
                neg_e = negative_sampling(pos_e,
                                          num_nodes=batch.num_nodes,
                                          num_neg_samples=pos_e.size(1)//4).to(device)
                pos_e = pos_e.long()
                neg_e = neg_e.long()

                val_loss += (safe_recon_loss(z, pos_e, neg_e)
                             + model.kl_loss(mu, logstd)/batch.num_nodes).item()

        avg_val = val_loss / len(val_loader)
        sched.step(avg_val)

        print(f"[Epoch {epoch:03d}] Train={avg_train:.4f}  Val={avg_val:.4f}  β={beta:.2f}  LR={opt.param_groups[0]['lr']:.2e}")

        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            no_imp = 0
            torch.save(model.state_dict(), "best_finetune.pt")
        else:
            no_imp += 1
            if no_imp >= early_stop_patience:
                print(f"⏹ Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load("best_finetune.pt", map_location=device))
    model.eval()

    embeds, labs = [], []
    for data in pyg:
        data = data.to(device)
        if use_edges:
            mu, logstd = model.encode(data.x,
                                      data.edge_index,
                                      data.edge_attr)
        else:
            mu, logstd = model.encode(data.x, data.edge_index)
        z = model.reparametrize(mu, logstd)
        embeds.append(pool_node_embeddings(z))
        labs.append(int(data.y))

    return np.vstack(embeds), np.array(labs)
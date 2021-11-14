import torch
import torch.nn as nn


class MF(nn.Module):
    def __init__(self, args, data):
        super(MF, self).__init__()
        self.n_users = data.n_users
        self.n_items = data.n_items
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.decay = args.regs

        self.embed_user = nn.Embedding(self.n_users, self.emb_dim)
        self.embed_item = nn.Embedding(self.n_items, self.emb_dim)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    # Prediction function used when evaluation
    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        users = self.embed_user(torch.tensor(users).cuda(0))
        items = torch.transpose(self.embed_item(torch.tensor(items).cuda(0)), 0, 1)
        rate_batch = torch.matmul(users, items)

        return rate_batch.cpu().detach().numpy()


class BPRMF(MF):
    def __init__(self, args, data):
        super().__init__(args, data)

    def forward(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 0.5 * torch.norm(users) ** 2 + 0.5 * torch.norm(pos_items) ** 2 + 0.5 * torch.norm(neg_items) ** 2
        regularizer = regularizer / self.batch_size

        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores))

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss


class BCEMF(MF):
    def __init__(self, args, data):
        super().__init__(args, data)

    def forward(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 0.5 * torch.norm(users) ** 2 + 0.5 * torch.norm(pos_items) ** 2 + 0.5 * torch.norm(neg_items) ** 2
        regularizer = regularizer / self.batch_size

        mf_loss = torch.mean(torch.negative(torch.log(torch.sigmoid(pos_scores) + 1e-9))
                             + torch.negative(torch.log(1 - torch.sigmoid(neg_scores) + 1e-9)))
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss
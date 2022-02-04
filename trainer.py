from tqdm import tqdm

def train_epoch(model, optimizer, train_loader, Z, X, loss_fn):
    model.train()
    losses = 0
    for idx in tqdm(train_loader): 
        z = Z[idx,:]
        x = X[idx, :]
        x_hat = model(z)[0]
        
        optimizer.zero_grad()
        
        loss = loss_fn(x, x_hat)
        loss.backward()
        
        optimizer.step()
        losses += loss.item()
    return losses / len(train_loader)

    
def val_epoch(model, Z, X, loss_fn):
    model.eval()
    X_hat = model(Z)[0]
    loss = loss_fn(X, X_hat)
    return loss.item()


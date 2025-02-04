from helpers import evaluate_accuracy
import torch

def train_word2vec(model, X, Y, vocab_size, optimiser, context_len, device, epochs=1, batch_size=32, RANDOM_SEED=42, lossi = None):
    torch.manual_seed(RANDOM_SEED) # Set manual seed

    # Send data to the device
    # X_train, y_train = X_train.to(device), y_train.to(device)
    # X_test, y_test = X_test.to(device), y_test.to(device)

    # Create batches from dataset
    dataset = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True, 
                                            drop_last=True)
    
    print(f"Training for {epochs} epochs with batch size {batch_size}, batches: {len(dataloader)}")
    
    # Loop through the data
    for epoch in range(epochs):
        ### Training
        total_loss = 0  # Track loss across batches
        
        for batch_idx, batch in enumerate(dataloader):
            # Unpack batch
            X_batch, positive_samples = batch
            # Generate negative samples
            negative_samples = torch.randint(0, vocab_size, (X_batch.size(0), context_len -1))

            if batch_idx == 0:
                print(f"Batch shapes - X: {X_batch.shape}, Pos: {positive_samples.shape}, Neg: {negative_samples.shape}")
                print(f"Max indices - X: {X_batch.max()}, Pos: {positive_samples.max()}, Neg: {negative_samples.max()}")
                print(f"Vocab size: {vocab_size}")

            # 1. Forward pass (now returns loss directly)
            loss = model(X_batch, positive_samples, negative_samples)  # Model computes loss internally
            
            # 2. Zero the gradients
            optimiser.zero_grad()
            
            # 3. Loss backwards
            loss.backward()
            
            # 4. Step the optimiser
            optimiser.step()
            
            total_loss += loss.item()

            # Track loss
            model.loss_history.append(loss.item())
            
            accuracy = evaluate_accuracy(model, X, Y, vocab_size)

            if batch_idx % 20 == 0:
                print(f"Epoch {epoch + 1} | Batch {batch_idx + 1} | Loss: {loss.item():.5f} | Accuracy: {accuracy:.4f}")
        
    return model.loss_history
        # print(f"Epoch {epoch + 1} | Loss: {loss.item():.5f} | Accuracy: {accuracy:.4f}")

        # print(f"Epoch {epoch + 1} | Batch {batch_idx + 1} | Loss: {loss.item():.5f}")
        # Print out what's happening every 100 epochs
        # if epoch % 100 == 0:
        #     avg_loss = total_loss / len(dataloader)
        #     print(f"Epoch: {epoch} | Average Loss: {avg_loss:.5f}")

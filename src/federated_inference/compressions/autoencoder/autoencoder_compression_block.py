import os
import torch
import torch.nn as nn
import torch.optim as optim


from federated_inference.compressions.autoencoder.models import Encoder, Decoder, Autoencoder

def AutoEncoderCompressionBlock():

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize models
        self.encoder = Encoder().to(self.device)
        self.decoder = Decoder().to(self.device)
        self.autoencoder = Autoencoder(self.encoder, self.decoder).to(self.device)

        # Loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.001)

    def train(self, train_loader):    
        for epoch in range(20):
            self.autoencoder.train()
            running_loss = 0.0
            for images, _ in train_loader:
                images = images.to(self.device)

                outputs = self.autoencoder(images)
                loss = self.criterion(outputs, images)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            print(f"Epoch [{epoch+1}/10], Loss: {running_loss/len(train_loader):.4f}")

    def test(self, test_loader):
        # Evaluation
        self.autoencoder.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.device)
                outputs = self.autoencoder(images)
                loss = self.criterion(outputs, images)
                test_loss += loss.item()
        print(f"Test Loss: {test_loss/len(test_loader):.4f}")

    def save(self):
        # Save models
        torch.save(self.encoder.state_dict(), "encoder_model.pth")
        torch.save(self.decoder.state_dict(), "decoder_model.pth")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # File size function
    def get_file_size(file_path):
        size = os.path.getsize(file_path)
        for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024


    encoder_model_size = get_file_size("encoder_model.pth")
    decoder_model_size = get_file_size("decoder_model.pth")
    print("Encoder model size:", encoder_model_size)
    print("Decoder model size:", decoder_model_size)

    # Load models
    loaded_encoder = Encoder().to(device)
    loaded_decoder = Decoder().to(device)
    loaded_encoder.load_state_dict(torch.load("encoder_model.pth"))
    loaded_decoder.load_state_dict(torch.load("decoder_model.pth"))
    loaded_encoder.eval()
    loaded_decoder.eval()

    # Use loaded models for prediction
    sample_images, _ = next(iter(test_loader))
    sample_images = sample_images.to(device)
    with torch.no_grad():
        encoded_imgs = loaded_encoder(sample_images)
        decoded_imgs = loaded_decoder(encoded_imgs)

    # Visualization
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Original image
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(sample_images[i].cpu().view(28, 28), cmap='gray')
        ax.axis('off')

        # Reconstructed image
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].cpu().view(28, 28), cmap='gray')
        ax.axis('off')
    plt.show()


import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim
from overcoming_catastrophic_forgetting_in_autoencoders.elastic_weight_consolidation import ElasticWeightConsolidation
 
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=10):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([40, 64, 64]),  # Assuming the input images are 32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            
            nn.Conv2d(40, 40, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([40, 32, 32]),  # After max pooling, the size is halved
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Flatten(),
            nn.LayerNorm([10240]),  # After flattening, the size is 10240
            nn.ReLU(),
            nn.Linear(int(10240), latent_dim),
             nn.LayerNorm([latent_dim]),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, int(10240)),
            nn.LayerNorm([10240]),
            nn.ReLU(),
            nn.Unflatten(1, (40, 16, 16)),
            nn.LayerNorm([40, 16, 16]),
            nn.ConvTranspose2d(40, 40, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1),
            nn.LayerNorm([40, 32, 32]),
            nn.ReLU(),
            nn.ConvTranspose2d(40, 1, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1),
            nn.LayerNorm([1, 64, 64]),  # Assuming the output images are 64x64
        )
         
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
 


class VideoEmbedder(ElasticWeightConsolidation):
    def __init__(self, latent_dim=64, batch_size=40):
        super(VideoEmbedder, self).__init__()
        self.model = Autoencoder(latent_dim)
        self.latent_dim = latent_dim  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.batch_size = batch_size

    def training_loop(self, num_epochs: int):
        
        for epoch in range(num_epochs):
            for data in self.dataloader:
                input_batch = data[0]
                self.optimizer.zero_grad()
                output = self.model(input_batch)
                loss = self.ewc_loss() + self.criterion(output, input_batch)
                loss.backward()
                self.optimizer.step()


        self.register_ewc_params()        
        return epoch, loss

    def save_model(self, path=None):
        torch.save(self.model.state_dict(), f"{path}/{self.name}_model_{self.latent_dim}.pt")

    def load_model(self, path=None, name=None, latent_dim=None):
        print(f"Loaded model: {path}/{self.name}_model_{self.latent_dim}.pt")
        self.model.load_state_dict(torch.load(f"{path}/{self.name}_model_{self.latent_dim}.pt"))
        self.model.eval()
    
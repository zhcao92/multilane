# model.py

import torch
import torch.nn as nn
import math

class MLP(nn.Module):
    def __init__(self, input_dim=7):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 2)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming Normal Initialization for ReLU
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # No activation for regression outputs
        return x

class TransformerModel(nn.Module):
    def __init__(self, embed_dim=8, num_heads=2, num_layers=2, dropout=0.0, max_agents=10):
        """
        Transformer-based regression model with flexible number of surrounding agents.

        Parameters:
            embed_dim (int): Dimension of the embedding.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of Transformer encoder layers.
            dropout (float): Dropout rate.
            max_agents (int): Maximum number of surrounding agents to handle.
        """
        super(TransformerModel, self).__init__()
        self.embed_dim = embed_dim
        self.max_agents = max_agents
        self.num_tokens = 2 + self.max_agents  # Ego Car, Speed Limit, Agents

        # Input projections
        self.ego_proj = nn.Linear(3, embed_dim)  # Ego Car: x, y, speed (x is always 0)
        self.agent_proj = nn.Linear(3, embed_dim)  # Surrounding Agent: x_rel, y, speed
        self.speed_limit_proj = nn.Linear(1, embed_dim)  # Speed Limit

        # Positional Encoding
        # self.positional_encoding = self._generate_positional_encoding(self.num_tokens, embed_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(8, 2)  # Outputs: accel and dt
        )

        self.initialize_weights()

    def _generate_positional_encoding(self, seq_length, embed_dim):
        """
        Generate fixed positional encoding using sine and cosine functions.

        Parameters:
            seq_length (int): Maximum sequence length.
            embed_dim (int): Dimension of the embedding.

        Returns:
            Tensor: Positional encoding tensor of shape (1, seq_length, embed_dim)
        """
        pe = torch.zeros(seq_length, embed_dim)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, seq_length, embed_dim)
        return nn.Parameter(pe, requires_grad=False)

    def initialize_weights(self):
        # Initialize input projections
        nn.init.kaiming_normal_(self.ego_proj.weight, nonlinearity='relu')
        nn.init.constant_(self.ego_proj.bias, 0)

        nn.init.kaiming_normal_(self.agent_proj.weight, nonlinearity='relu')
        nn.init.constant_(self.agent_proj.bias, 0)

        nn.init.kaiming_normal_(self.speed_limit_proj.weight, nonlinearity='relu')
        nn.init.constant_(self.speed_limit_proj.bias, 0)

        # Initialize regression head
        for m in self.regressor:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, ego, agents, speed_limit, agent_mask):
        """
        Forward pass of the Transformer model.

        Parameters:
            ego (Tensor): Ego car features (batch_size, 3)
            agents (Tensor): Surrounding agents features (batch_size, max_agents, 3)
            speed_limit (Tensor): Speed limit feature (batch_size, 1)
            agent_mask (Tensor): Attention mask for agents (batch_size, max_agents)

        Returns:
            Tensor: Output tensor of shape (batch_size, 2)
        """
        batch_size = ego.size(0)

        # Project inputs
        # ego_emb = torch.relu(self.ego_proj(ego))  # (batch_size, embed_dim)
        # speed_limit_emb = torch.relu(self.speed_limit_proj(speed_limit))  # (batch_size, embed_dim)

        ego_emb = self.ego_proj(ego)  # (batch_size, embed_dim)
        speed_limit_emb = self.speed_limit_proj(speed_limit)  # (batch_size, embed_dim)

        # Project agents
        # agents_emb = torch.relu(self.agent_proj(agents))  # (batch_size, max_agents, embed_dim)
        agents_emb = self.agent_proj(agents)  # (batch_size, max_agents, embed_dim)

        # Concatenate all embeddings: [Ego Car, Agents..., Speed Limit]
        # Shape: (batch_size, 2 + max_agents, embed_dim)
        x = torch.cat([ego_emb.unsqueeze(1), agents_emb, speed_limit_emb.unsqueeze(1)], dim=1)

        # Add positional encoding
        # pe = self.positional_encoding[:, :x.size(1), :]  # (1, seq_length, embed_dim)
        # x = x + pe  # Broadcasting over batch_size

        # Prepare for Transformer: (seq_length, batch_size, embed_dim)
        x = x.permute(1, 0, 2)

        # Create src_key_padding_mask: True for padding tokens
        # First token: Ego Car (not padding)
        # Last token: Speed Limit (not padding)
        # Agents: padding based on agent_mask
        # Shape: (batch_size, seq_length)
        # Initialize with False (no padding)
        src_key_padding_mask = torch.zeros(batch_size, x.size(0), dtype=torch.bool, device=x.device)
        # Agents start from index 1 to max_agents
        if self.max_agents > 0:
            src_key_padding_mask[:, 1:self.max_agents + 1] = ~agent_mask  # True where padding
        # src_key_padding_mask = ~src_key_padding_mask  # Invert mask
        # Transformer Encoder
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)  # (seq_length, batch_size, embed_dim)

        # Aggregate output: Use Ego Car token's output
        x = x[0, :, :]  # (batch_size, embed_dim)

        # Regression Head
        out = self.regressor(x)  # (batch_size, 2)

        return out


class L2LModel(nn.Module):
    def __init__(self, embed_dim=8, num_heads=2, num_layers=2, dropout=0.0, max_agents=10):
        """
        Transformer-based regression model with flexible number of surrounding agents.

        Parameters:
            embed_dim (int): Dimension of the embedding.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of Transformer encoder layers.
            dropout (float): Dropout rate.
            max_agents (int): Maximum number of surrounding agents to handle.
        """
        super(L2LModel, self).__init__()
        self.embed_dim = embed_dim
        self.max_agents = max_agents
        self.num_tokens = 2 + self.max_agents  # Ego Car, Speed Limit, Agents

        # Input projections
        self.ego_proj = nn.Linear(3, embed_dim)  # Ego Car: x, y, speed (x is always 0)
        self.agent_proj = nn.Linear(3, embed_dim)  # Surrounding Agent: x_rel, y, speed
        self.speed_limit_proj = nn.Linear(1, embed_dim)  # Speed Limit

        # Positional Encoding
        # self.positional_encoding = self._generate_positional_encoding(self.num_tokens, embed_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Regression Head
        self.regressor_agent = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(8, 2)  # Outputs: accel and dt
        )

        self.regressor_speed_limit = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(8, 2)  # Outputs: accel and dt
        )

        self.initialize_weights()

    def _generate_positional_encoding(self, seq_length, embed_dim):
        """
        Generate fixed positional encoding using sine and cosine functions.

        Parameters:
            seq_length (int): Maximum sequence length.
            embed_dim (int): Dimension of the embedding.

        Returns:
            Tensor: Positional encoding tensor of shape (1, seq_length, embed_dim)
        """
        pe = torch.zeros(seq_length, embed_dim)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, seq_length, embed_dim)
        return nn.Parameter(pe, requires_grad=False)

    def initialize_weights(self):
        # Initialize input projections
        nn.init.kaiming_normal_(self.ego_proj.weight, nonlinearity='relu')
        nn.init.constant_(self.ego_proj.bias, 0)

        nn.init.kaiming_normal_(self.agent_proj.weight, nonlinearity='relu')
        nn.init.constant_(self.agent_proj.bias, 0)

        nn.init.kaiming_normal_(self.speed_limit_proj.weight, nonlinearity='relu')
        nn.init.constant_(self.speed_limit_proj.bias, 0)

        # Initialize regression head
        for m in self.regressor_agent:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for m in self.regressor_speed_limit:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, ego, agents, speed_limit, agent_mask):
        """
        Forward pass of the Transformer model.

        Parameters:
            ego (Tensor): Ego car features (batch_size, 3)
            agents (Tensor): Surrounding agents features (batch_size, max_agents, 3)
            speed_limit (Tensor): Speed limit feature (batch_size, 1)
            agent_mask (Tensor): Attention mask for agents (batch_size, max_agents)

        Returns:
            Tensor: Output tensor of shape (batch_size, 2)
        """
        batch_size = ego.size(0)

        # Project inputs
        # ego_emb = torch.relu(self.ego_proj(ego))  # (batch_size, embed_dim)
        # speed_limit_emb = torch.relu(self.speed_limit_proj(speed_limit))  # (batch_size, embed_dim)

        ego_emb = self.ego_proj(ego)  # (batch_size, embed_dim)
        speed_limit_emb = self.speed_limit_proj(speed_limit)  # (batch_size, embed_dim)

        # Project agents
        # agents_emb = torch.relu(self.agent_proj(agents))  # (batch_size, max_agents, embed_dim)
        agents_emb = self.agent_proj(agents)  # (batch_size, max_agents, embed_dim)

        # Concatenate all embeddings: [Ego Car, Agents..., Speed Limit]
        # Shape: (batch_size, 2 + max_agents, embed_dim)
        x = torch.cat([ego_emb.unsqueeze(1), agents_emb, speed_limit_emb.unsqueeze(1)], dim=1)

        # Add positional encoding
        # pe = self.positional_encoding[:, :x.size(1), :]  # (1, seq_length, embed_dim)
        # x = x + pe  # Broadcasting over batch_size

        # Prepare for Transformer: (seq_length, batch_size, embed_dim)
        x = x.permute(1, 0, 2)

        # Create src_key_padding_mask: True for padding tokens
        # First token: Ego Car (not padding)
        # Last token: Speed Limit (not padding)
        # Agents: padding based on agent_mask
        # Shape: (batch_size, seq_length)
        # Initialize with False (no padding)
        src_key_padding_mask = torch.zeros(batch_size, x.size(0), dtype=torch.bool, device=x.device)
        # Agents start from index 1 to max_agents
        if self.max_agents > 0:
            src_key_padding_mask[:, 1:self.max_agents + 1] = ~agent_mask  # True where padding
        # src_key_padding_mask = ~src_key_padding_mask  # Invert mask
        # Transformer Encoder
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)  # (seq_length, batch_size, embed_dim)

        # First Regression Head: Use speed limit's token output
        out1 = self.regressor_speed_limit(x[0, :, :])
        out1[:,1] = 0

        # Second Regression Head: Use surrounding_car token's output, if agent_mask is not all False
        # Determine which samples have at least one valid agent
        agent_present = torch.any(agent_mask, dim=1)  # Shape: (batch_size,)

        # Apply the regression head to all samples
        out2 = self.regressor_agent(x[1, :, :])  # Shape: (batch_size, 2)

        # Set out2 to [0, 0] for samples where no agents are present
        out2 = torch.where(
            agent_present.unsqueeze(1),  # Condition: Shape (batch_size, 1)
            out2,                        # If True: Use the regressor's output
            torch.zeros_like(out2)       # If False: Set to [0, 0]
        )

        out2[:,1] = 0

        out = torch.tensor([1,0], device=x.device).repeat(batch_size, 1)-out1-out2
        # out = torch.zeros(batch_size, 2, device=x.device)-out1-out2

        return out, {out1, out2}
    


def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_sigmoid(logits, tau=1.0):
    # Sample Gumbel noise
    gumbel_noise = sample_gumbel(logits.size())
    # Apply Gumbel-Sigmoid approximation
    y = logits + gumbel_noise
    return torch.sigmoid(y / tau)

class L2LModel_GSP(nn.Module):
    def __init__(self, embed_dim=8, num_heads=2, num_layers=2, dropout=0.0, max_agents=10):
        """
        Transformer-based regression model with flexible number of surrounding agents.

        Parameters:
            embed_dim (int): Dimension of the embedding.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of Transformer encoder layers.
            dropout (float): Dropout rate.
            max_agents (int): Maximum number of surrounding agents to handle.
        """
        super(L2LModel_GSP, self).__init__()
        self.embed_dim = embed_dim
        self.max_agents = max_agents
        self.num_tokens = 2 + self.max_agents  # Ego Car, Speed Limit, Agents

        # Input projections
        self.ego_proj = nn.Linear(3, embed_dim)  # Ego Car: x, y, speed (x is always 0)
        self.agent_proj = nn.Linear(3, embed_dim)  # Surrounding Agent: x_rel, y, speed
        self.speed_limit_proj = nn.Linear(1, embed_dim)  # Speed Limit

        # Positional Encoding
        # self.positional_encoding = self._generate_positional_encoding(self.num_tokens, embed_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Regression Head
        self.regressor_agent = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(8, 3)  # Outputs: mu sigma alpha
        )

        self.binary_regressor = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(8, 1)  # Outputs: mu sigma alpha
        )

        self.regressor_speed_limit = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(8, 2)  # Outputs: speed_value, alpha
        )

        self.initialize_weights()

    def _generate_positional_encoding(self, seq_length, embed_dim):
        """
        Generate fixed positional encoding using sine and cosine functions.

        Parameters:
            seq_length (int): Maximum sequence length.
            embed_dim (int): Dimension of the embedding.

        Returns:
            Tensor: Positional encoding tensor of shape (1, seq_length, embed_dim)
        """
        pe = torch.zeros(seq_length, embed_dim)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, seq_length, embed_dim)
        return nn.Parameter(pe, requires_grad=False)

    def initialize_weights(self):
        # Initialize input projections
        nn.init.kaiming_normal_(self.ego_proj.weight, nonlinearity='relu')
        nn.init.constant_(self.ego_proj.bias, 0)

        nn.init.kaiming_normal_(self.agent_proj.weight, nonlinearity='relu')
        nn.init.constant_(self.agent_proj.bias, 0)

        nn.init.kaiming_normal_(self.speed_limit_proj.weight, nonlinearity='relu')
        nn.init.constant_(self.speed_limit_proj.bias, 0)

        # Initialize regression head
        for m in self.regressor_agent:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for m in self.regressor_speed_limit:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, ego, agents, speed_limit, agent_mask):
        """
        Forward pass of the Transformer model.

        Parameters:
            ego (Tensor): Ego car features (batch_size, 3)
            agents (Tensor): Surrounding agents features (batch_size, max_agents, 3)
            speed_limit (Tensor): Speed limit feature (batch_size, 1)
            agent_mask (Tensor): Attention mask for agents (batch_size, max_agents)

        Returns:
            Tensor: Output tensor of shape (batch_size, 2)
        """
        batch_size = ego.size(0)

        # Project inputs
        # ego_emb = torch.relu(self.ego_proj(ego))  # (batch_size, embed_dim)
        # speed_limit_emb = torch.relu(self.speed_limit_proj(speed_limit))  # (batch_size, embed_dim)

        ego_emb = self.ego_proj(ego)  # (batch_size, embed_dim)
        speed_limit_emb = self.speed_limit_proj(speed_limit)  # (batch_size, embed_dim)

        # Project agents
        # agents_emb = torch.relu(self.agent_proj(agents))  # (batch_size, max_agents, embed_dim)
        agents_emb = self.agent_proj(agents)  # (batch_size, max_agents, embed_dim)

        # Concatenate all embeddings: [Ego Car, Agents..., Speed Limit]
        # Shape: (batch_size, 2 + max_agents, embed_dim)
        x = torch.cat([ego_emb.unsqueeze(1), agents_emb, speed_limit_emb.unsqueeze(1)], dim=1)

        # Add positional encoding
        # pe = self.positional_encoding[:, :x.size(1), :]  # (1, seq_length, embed_dim)
        # x = x + pe  # Broadcasting over batch_size

        # Prepare for Transformer: (seq_length, batch_size, embed_dim)
        x = x.permute(1, 0, 2)

        # Create src_key_padding_mask: True for padding tokens
        # First token: Ego Car (not padding)
        # Last token: Speed Limit (not padding)
        # Agents: padding based on agent_mask
        # Shape: (batch_size, seq_length)
        # Initialize with False (no padding)
        src_key_padding_mask = torch.zeros(batch_size, x.size(0), dtype=torch.bool, device=x.device)
        # Agents start from index 1 to max_agents
        if self.max_agents > 0:
            src_key_padding_mask[:, 1:self.max_agents + 1] = ~agent_mask  # True where padding
            
        # Transformer Encoder
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)  # (seq_length, batch_size, embed_dim)

        gaussian_agent = self.regressor_agent(x[1, :, :]) # Shape: (batch_size, 3)
        mu_agent, sigma_agent, alpha_agent = torch.split(gaussian_agent, 1, dim=1)

        speed_out = self.regressor_speed_limit(x[-1, :, :]) # Shape: (batch_size, 2)
        speed_value, alpha_speed = torch.split(speed_out, 1, dim=1)

        # binary_regressor_out = self.binary_regressor(x[0, :, :]) # Shape: (batch_size, 1)
        # speed_limit_binary = gumbel_sigmoid(binary_regressor_out, tau=0.5)

        # Ensure alpha_speed, alpha_agent is non-negative
        alpha_speed =torch.sigmoid(alpha_speed)
        alpha_agent = 10 * torch.sigmoid(alpha_agent)

        # Calculate the gaussian value at the ego car's position
        ego_position = torch.zeros(batch_size, 2, device=x.device)
        ego_position[:, 0] = ego[:, 0]  # x
        ego_position[:, 1] = ego[:, 1]  # y

        ego_speed = torch.zeros(batch_size, 1, device=x.device)
        ego_speed = ego[:, 2]  # Speed of the ego car
        ego_speed = ego_speed.unsqueeze(1)

        desired_speed = speed_limit
        desired_speed_value = self._accel_for_desired_speed(desired_speed, ego_speed)

        # Add the position of surrounding agents to mu_agent
        agent_positions = agents[:, 0:1, 0:1]  # Extract x_rel positions of agents
        agent_positions = agent_positions.squeeze(2)

        sigma_agent = 100 * torch.sigmoid(sigma_agent)

        mu_agent = agent_positions

        # Calculate Gaussian value for agent at ego car's position
        gaussian_value_agent = torch.exp(-0.5 * ((ego_position[:,0].unsqueeze(1) - mu_agent) / sigma_agent).pow(2).sum(dim=1, keepdim=True))

        agent_present = torch.any(agent_mask, dim=1)  # Shape: (batch_size,)

        gaussian_value_agent = torch.where(
            agent_present.unsqueeze(1),  # Condition: Shape (batch_size, 1)
            gaussian_value_agent,                        # If True: Use the regressor's output
            torch.zeros_like(gaussian_value_agent)       # If False: Set to 0
        )

        # Calculate the Differential of Gaussian value for agent at ego car's position
        # diff_gaussian_value_agent = -((ego_position[:, 0].unsqueeze(1) - mu_agent) / (sigma_agent ** 2)) * gaussian_value_agent

        # Calculate the cdf of Gaussian value for agent at ego car's position
        # cdf_gaussian_value_agent = 0.5 * (1 + torch.erf((ego_position[:, 0].unsqueeze(1) - mu_agent) / (sigma_agent * math.sqrt(2))))

        out = torch.zeros(batch_size, 2, device=x.device)

        out[:,0] = (desired_speed_value - alpha_agent * gaussian_value_agent).squeeze()

        out[:,1] = 0
        
        return out, {speed_limit_binary, speed_value}
    
    def _accel_for_desired_speed(self, desired_speed, current_speed):
        """
        Calculate the acceleration required to reach the desired speed.

        Parameters:
            desired_speed (Tensor): Desired speed value.
            current_speed (Tensor): Current speed value.

        Returns:
            Tensor: Acceleration value.
        """
        # Calculate the acceleration using IDM model
        # Reference: https://en.wikipedia.org/wiki/Intelligent_driver_model
        # IDM Parameters
        a = 1
        delta = 4.0

        return a * (1 - (current_speed / desired_speed) ** delta)
    

    def _accel_for_desired_speed_liner(self, desired_speed, current_speed):
        """
        Calculate the acceleration required to reach the desired speed.

        Parameters:
            desired_speed (Tensor): Desired speed value.
            current_speed (Tensor): Current speed value.

        Returns:
            Tensor: Acceleration value.
        """

        return (desired_speed-current_speed)

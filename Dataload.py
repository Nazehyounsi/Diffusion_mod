import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

folder_path = 'C:/Users/NEZIH YOUNSI/Desktop/Hcapriori_input/Observaton_Context_Tuples'


def is_valid_chunk(chunk):
    if not chunk[0]:  # Check if the chunk[0] is an empty list
        return False
    for event in chunk[0]:
        if event[0] is float or event[1] is None or event[2] is None:
            return False
    return True

def load_data_from_folder(folder_path):
    all_data = []  # This list will hold all our chunks (both observations and actions) from all files.

    # Iterate over each file in the directory
    for file in os.listdir(folder_path):
        # If the file ends with '.txt', we process it
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), 'r') as f:
                # Read the lines and filter out any empty ones.
                lines = f.readlines()
                non_empty_lines = [line.strip() for line in lines if line.strip() != ""]

                # Transform the non-empty line strings into actual list of tuples.
                chunks = [eval(line) for line in non_empty_lines]

                # Extract observation, action and chunk descriptor
                observation_chunks = [chunk[:-1] for chunk in chunks[::2]]  # get all tuples except the last one
                action_chunks = chunks[1::2]  # extract every second element starting from 1
                chunk_descriptors = [chunk[-1] for chunk in chunks[::2]]

                # Replace None values by -1 in chunk descriptor
                for i in range(len(chunk_descriptors)):
                    event = list(chunk_descriptors[i])
                    if event[2] is None:
                        event[2] = -1
                    if event[1] is None:
                        event[1] = -1
                    chunk_descriptors[i] = tuple(event)

                # Extend the all_data list with the observation, action, and chunk descriptor
                all_data.extend(list(zip(observation_chunks, action_chunks, chunk_descriptors)))

    return all_data  # Return the master list containing chunks from all files


def preprocess_data(data):
    filtered_data = [chunk for chunk in data if is_valid_chunk(chunk)]
    data = filtered_data
    for chunk in data:

        # Observation vector
        # Standardizing the starting time
        min_start_time = min([event[1] for event in chunk[0]])
        for i in range(len(chunk[0])):
            event = list(chunk[0][i])
            event[1] -= min_start_time  # Adjust starting time based on the first event in the chunk
            event[1] = round(event[1], 3)
            chunk[0][i] = tuple(event)

        # Normalizing the duration
        max_duration = max([event[2] for event in chunk[0]])
        for i in range(len(chunk[0])):
            event = list(chunk[0][i])
            event[2] /= max_duration  # Normalize the duration based on the maximum duration in the chunk
            event[2] = round(event[2], 3)
            chunk[0][i] = tuple(event)

            # Action vector
            # Standardizing the starting time
            min_start_time = min([event[1] for event in chunk[1]])
            for i in range(len(chunk[1])):
                event = list(chunk[1][i])
                event[1] -= min_start_time  # Adjust starting time based on the first event in the chunk
                event[1] = round(event[1], 3)
                chunk[1][i] = tuple(event)

            # Normalizing the duration
            max_duration = max([event[2] for event in chunk[1]])
            for i in range(len(chunk[1])):
                event = list(chunk[1][i])
                event[2] /= max_duration  # Normalize the duration based on the maximum duration in the chunk
                event[2] = round(event[2], 3)
                chunk[1][i] = tuple(event)

    return data

class MotivationalInterviewingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        observation, action, chunk_descriptor = self.data[idx]

        # Convert tuples into tensors for PyTorch
        return torch.tensor(observation, dtype=torch.float32), torch.tensor(action, dtype=torch.float32),torch.tensor(chunk_descriptor, dtype=torch.float32)
        #return observation, action, chunk_descriptor


#embedding of observation (and action ?)

class EventEmbedder(nn.Module):
    def __init__(self, num_event_types, event_embedding_dim, continuous_embedding_dim, event_weights=None):
        super(EventEmbedder, self).__init__()
        self.event_embedding = nn.Embedding(num_event_types, event_embedding_dim)
        self.duration_embedding = nn.Linear(1, continuous_embedding_dim)
        self.start_time_embedding = nn.Linear(1, continuous_embedding_dim)

        input_dim = event_embedding_dim + 2 * continuous_embedding_dim
        self.interaction_lstm = nn.LSTM(input_dim, 16, batch_first=True)

        # Initialize event type weights
        if event_weights is None:
            # If no weights are provided, initialize as ones (i.e., no change in importance).
            self.event_weights = nn.Parameter(torch.ones(num_event_types), requires_grad=False)
        else:
            self.event_weights = nn.Parameter(torch.tensor(event_weights), requires_grad=False)

    def forward(self, x):

        event_type_indices = x[:, :, 0].long()
        event_type_embed = self.event_embedding(event_type_indices)
        duration_embed = self.duration_embedding(x[:, :, 1].unsqueeze(-1))
        start_time_embed = self.start_time_embedding(x[:, :, 2].unsqueeze(-1))

        #if add weights on events
        event_type_weights = self.event_weights[event_type_indices]
        event_type_embed = event_type_embed * event_type_weights.unsqueeze(-1)

        concatenated_embeddings = torch.cat([event_type_embed, duration_embed, start_time_embed], dim=-1)
        output, _ = self.interaction_lstm(concatenated_embeddings)
        return output

data = load_data_from_folder(folder_path)
data = preprocess_data(data)
dataset = MotivationalInterviewingDataset(data)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


num_event_types = 42
event_embedding_dim = 16
continuous_embedding_dim = 3
#event_weights = vector of dim num event types
embedder = EventEmbedder(num_event_types, event_embedding_dim, continuous_embedding_dim, event_weights=None)

#for epoch ?
#for observations, actions, chunk_descriptors in dataloader:

    # Convert observations to tensors and pass through the embedder
    #observations_tensor = torch.tensor(observations, dtype=torch.float32)
    #embedded_observations = embedder(observations)

    #convert chunk desk into tensor (done already)
    #concat the embeddeb obs to chunk descriptor
    #convert action into tensor then embed action
    # The next steps would involve passing embedded_observations through the diffusion model,
    # computing the loss, backpropagating, and updating the model weights.
    # For example:
    # outputs = diffusion_model(embedded_observations)
    # loss = criterion(outputs, actions)


#model to add (diffusion)
# for epoch in range(num_epochs):
#     for observations, actions in dataloader:
#         # Pass observations through your model
#         outputs = model(observations)
#
#         # Compute loss between outputs and actions
#         loss = criterion(outputs, actions)
#
#         # Backpropagation and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()


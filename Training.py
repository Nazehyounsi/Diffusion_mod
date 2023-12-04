
from itertools import product
import pickle
import numpy as np
import os
import torch
from sklearn.neighbors import KernelDensity
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import mean_squared_error

from Models import Model_mlp_diff,  Model_Cond_Diffusion, EventEmbedder

import wandb

os.environ["WANDB_MODE"] = "offline"

DATASET_PATH = "dataset"
SAVE_DATA_DIR = "output"  # for models/data

EXPERIMENTS = [
    {
        "exp_name": "diffusion",
        "model_type": "diffusion",
        "drop_prob": 0.0,
    },
]

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Diffusion Model training 1.0",

    # track hyperparameters and run metadata
    config={
        "learning_rate": 1e-4,
        "architecture": "Basic Original Option 0",
        "dataset": "AnnoMI",
        "epochs": 100,
    }
)
EXTRA_DIFFUSION_STEPS = [0, 2, 4, 8, 16, 32]
GUIDE_WEIGHTS = [0.0, 4.0, 8.0]

n_epoch = 100
lrate = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_hidden = 512
batch_size = 32
n_T = 50
net_type = "transformer"
#event_embedder
num_event_types = 43
event_embedding_dim = 16
continuous_embedding_dim = 3
#timesiren
embed_output_dim = 16

def is_valid_chunk(chunk):
    if not chunk[0]:  # Check if the chunk[0] is an empty list
        return False
    for event in chunk[0]:
        if event[0] is float or event[1] is None or event[2] is None:
            return False
    return True

def load_data_from_folder(folder_path):
    all_data = []  # This list will hold all our chunks (both observations and actions) from all files.
    total_lines = 0
    # Iterate over each file in the directory
    for file in os.listdir(folder_path):
        # If the file ends with '.txt', we process it
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), 'r') as f:
                # Read the lines and filter out any empty ones.
                lines = f.readlines()
                total_lines += len(lines)
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

# Assuming the functions 'load_data_from_folder' and 'preprocess_data' are defined as in your provided code.

class MyCustomDataset(Dataset):
    def __init__(self, folder_path, train_or_test="train", train_prop=0.90):
        # Load and preprocess data
        raw_data = load_data_from_folder(folder_path)
        self.raw_data_length = len(raw_data)
        processed_data = preprocess_data(raw_data)

        # Split the data into training and testing based on train_prop
        n_train = int(len(processed_data) * train_prop)
        if train_or_test == "train":
            self.data = processed_data[:n_train]
        elif train_or_test == "test":
            self.data = processed_data[n_train:]
        else:
            raise ValueError("train_or_test should be either 'train' or 'test'")

        # Transform data into the new format
        self.transformed_data = []
        last_action = None  # Initialize last action as None
        for observation, action, chunk_descriptor in self.data:
            for i in range(len(action)):
                if i == 0 and last_action is not None:
                    prev_action = last_action
                else:
                    prev_action = action[i - 1] if i > 0 else [0] * len(
                        action[0])  # Use zero vector if no previous action

                x = observation + [prev_action]
                y = action[i]
                self.transformed_data.append((x, y, chunk_descriptor))

            last_action = action[-1]  # Save the last action of the current sequence

    def __len__(self):
        # Return the number of transformed data points
        return len(self.transformed_data)

    def __getitem__(self, idx):
        # Retrieve the transformed data point at the given index
        x, y, chunk_descriptor = self.transformed_data[idx]

        # Convert lists into tensors for PyTorch
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        chunk_descriptor_tensor = torch.tensor(chunk_descriptor, dtype=torch.float32)

        return x_tensor, y_tensor, chunk_descriptor_tensor

    def collate_fn(batch):
        # Unzip the batch into separate lists
        x_data, y_data, chunk_descriptors = zip(*batch)

        # Separate observations and previous actions
        observations = [x[:-1] for x in x_data]  # All elements except the last one
        prev_actions = [x[-1] for x in x_data]  # Only the last element

        # Pad the observation sequences
        observations_padded = torch.nn.utils.rnn.pad_sequence([obs.clone().detach() for obs in observations],
                                                              batch_first=True, padding_value=0)

        #prev_actions_tensor = torch.stack([torch.tensor(pa) for pa in prev_actions])

        # Convert previous actions to tensor
        prev_actions_tensor = torch.stack([pa.clone().detach() for pa in prev_actions])

        # Concatenate the padded observations with the previous actions
        x_tensors = torch.cat([observations_padded, prev_actions_tensor.unsqueeze(1)], dim=1)

        # Convert y_data to tensor
        y_tensors = torch.stack([y.clone().detach() for y in y_data])

        # Convert chunk_descriptors to tensor
        chunk_descriptors_tensor = torch.stack([cd.clone().detach() for cd in chunk_descriptors])

        return x_tensors, y_tensors, chunk_descriptors_tensor


def train_claw(experiment, n_epoch, lrate, device, n_hidden, batch_size, n_T, net_type, EXTRA_DIFFUSION_STEPS, GUIDE_WEIGHTS):
    # Unpack experiment settings
    exp_name = experiment["exp_name"]
    model_type = experiment["model_type"]
    drop_prob = experiment["drop_prob"]

    torch.autograd.set_detect_anomaly(True)

    # get datasets set up
    tf = transforms.Compose([])

    # Update the dataset path here (dataset for local run)
    folder_path = 'C:/Users/NEZIH YOUNSI/Desktop/Hcapriori_input/Observaton_Context_Tuples'

    # #Dataset for gpu
    # folder_path = '~/Observaton_Context_Tuples'
    # expanded_folder_path = os.path.expanduser(folder_path)
    # folder_path = expanded_folder_path



    # Use MyCustomDataset instead of ClawCustomDataset
    torch_data_train = MyCustomDataset(folder_path, train_or_test="train", train_prop=0.90)
    test_dataset = MyCustomDataset(folder_path, train_or_test="test")
    dataload_train = DataLoader(torch_data_train, batch_size=batch_size, shuffle=False, collate_fn=MyCustomDataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Calculate the total number of batches
    total_batches = len(dataload_train)
    print(f"Total number of batches: {total_batches}")
    print(f"Number of samples in raw data: {torch_data_train.raw_data_length}")


    event_embedder = EventEmbedder(num_event_types, event_embedding_dim, continuous_embedding_dim,embed_output_dim, event_weights=None)

   # Determine the shape of input and output tensors
    sample_observation, sample_action,_ = torch_data_train[0]
    input_shape = sample_observation.shape
    output_dim = sample_action.shape[0]

    x_dim = input_shape
    #torch_data_train.image_all.shape[1:]
    y_dim = output_dim
    #torch_data_train.action_all.shape[1]
    t_dim = 1
    # create model
    if model_type == "diffusion":
        #ici qu'on appel le model_mlp_diff fusionné a mon embedding model
        nn_model = Model_mlp_diff(
            event_embedder, y_dim, net_type="transformer").to(device)
        model = Model_Cond_Diffusion(
            nn_model,
            betas=(1e-4, 0.02),
            n_T=n_T,
            device=device,
            x_dim=x_dim,
            y_dim=y_dim,
            drop_prob=drop_prob,
            guide_w=0.0,
        )
    else:
        raise NotImplementedError

    # Count the number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lrate)

    for ep in tqdm(range(n_epoch), desc="Epoch"):
        results_ep = [ep]
        model.train()

        # lrate decay
        optim.param_groups[0]["lr"] = lrate * ((np.cos((ep / n_epoch) * np.pi) + 1) / 2)

        # train loop
        pbar = tqdm(dataload_train)
        loss_ep, n_batch = 0, 0
        for x_batch, y_batch, chunk_descriptor in pbar:
            #need to concat the chunk descriptor after the first test and see its impact
            x_batch = x_batch.type(torch.FloatTensor).to(device) #obs
            y_batch = y_batch.type(torch.FloatTensor).to(device) #targets

            loss = model.loss_on_batch(x_batch, y_batch)
            optim.zero_grad()
            loss.backward()
            loss_ep += loss.detach().item()
            n_batch += 1
            pbar.set_description(f"train loss: {loss_ep / n_batch:.4f}")
            wandb.log({"loss": loss})
            optim.step()
        results_ep.append(loss_ep / n_batch)

    model.eval()
    test_results = []
    #idxs = [14, 2, 0, 9, 5, 35, 16]
    extra_diffusion_steps = EXTRA_DIFFUSION_STEPS if exp_name == "diffusion" else [0]
    use_kde = [False, True] if exp_name == "diffusion" else [False]
    guide_weight_list = GUIDE_WEIGHTS if exp_name == "cfg" else [None]
    #idxs_data = [[] for _ in range(len(idxs))]
    for extra_diffusion_step, guide_weight, use_kde in product(extra_diffusion_steps, guide_weight_list, use_kde):
        if extra_diffusion_step != 0 and use_kde:
            continue
        for x_batch, y_batch, _ in test_dataloader:
            x_eval = x_batch.to(device)

            for j in range(6 if not use_kde else 300):
                x_eval_ = x_eval.repeat(50, 1, 1, 1)
                with torch.no_grad():  # Use torch.no_grad() for evaluation
                    if exp_name == "cfg":
                        model.guide_w = guide_weight
                    if model_type != "diffusion":
                        y_pred_ = model.sample(x_eval_).detach().cpu().numpy()
                    else:
                        if extra_diffusion_step == 0:
                            y_pred_ = model.sample(x_eval_, extract_embedding=True).detach().cpu().numpy()
                            if use_kde:
                                # kde
                                torch_obs_many = x_eval_
                                action_pred_many = model.sample(torch_obs_many).cpu().numpy()
                                # fit kde to the sampled actions
                                kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(action_pred_many)
                                # choose the max likelihood one
                                log_density = kde.score_samples(action_pred_many)
                                idx = np.argmax(log_density)
                                y_pred_ = action_pred_many[idx][None, :]
                        else:
                            y_pred_ = model.sample_extra(x_eval_,
                                                         extra_steps=extra_diffusion_step).detach().cpu().numpy()
                if j == 0:
                    y_pred = y_pred_
                else:
                    y_pred = np.concatenate([y_pred, y_pred_])

            # Store or process the predictions as needed
            test_results.append(y_pred)

        # Save data as a pickle
        true_exp_name = exp_name
        if extra_diffusion_step != 0:
            true_exp_name = f"{exp_name}_extra-diffusion_{extra_diffusion_step}"
        if use_kde:
            true_exp_name = f"{exp_name}_kde"
        if guide_weight is not None:
            true_exp_name = f"{exp_name}_guide-weight_{guide_weight}"
        with open(os.path.join(SAVE_DATA_DIR, f"{true_exp_name}.pkl"), "wb") as f:
            pickle.dump(idxs_data, f)

    # def evaluate_model(test_dataloader, model, device):
    #     model.eval()
    #     total_mse = 0.0
    #     total_samples = 0
    #
    #     with torch.no_grad():
    #         for x_batch, y_batch, _ in test_dataloader:
    #             x_batch = x_batch.to(device)
    #             y_batch = y_batch.to(device)
    #
    #             # Use the model's sampling method to generate predictions
    #             y_pred = model.sample(x_batch)  # Adjust this line based on your model's sampling method
    #
    #             # Calculate MSE for the batch
    #             mse = mean_squared_error(y_batch.cpu().numpy(), y_pred.cpu().numpy())
    #             total_mse += mse * x_batch.size(0)
    #             total_samples += x_batch.size(0)
    #
    #     # Calculate average MSE over all batches
    #     avg_mse = total_mse / total_samples
    #     return avg_mse
    #

    # AFTER TRAINING LOOP CALL THE EVAL METHODE
    # avg_mse = evaluate_model(test_dataloader, model, device)
    # print(f"Average MSE on Test Set: {avg_mse}")


if __name__ == "__main__":
    os.makedirs(SAVE_DATA_DIR, exist_ok=True)
    for experiment in EXPERIMENTS:
        train_claw(experiment, n_epoch, lrate, device, n_hidden, batch_size, n_T, net_type, EXTRA_DIFFUSION_STEPS, GUIDE_WEIGHTS)

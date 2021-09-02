import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import utils

# Define training data loader
test_dataset = utils.G2NetDataSet(main_folder='/slowfs/datasets/g2net-gravitational-wave-detection',
                                  set_type='test')
test_loader = DataLoader(test_dataset)

# Load the model
model = torch.load('models/logit_v1.pt')

# Inference
with open('output.csv', 'w') as output_file:
    output_file.write("id,target\n")
    progress_bar = tqdm(test_loader)
    for data in progress_bar:
        inputs, _, filename = data
        outputs = model(inputs)
        output_file.write("%s,%f\n" % (filename[0], outputs[0]))
        progress_bar.set_postfix({'last_target': float(outputs)})



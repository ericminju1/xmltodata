from model.demucs4.demucs.htdemucs import HTDemucs
from data.reorg.dataset_list import load_data
import torch
from glob import glob 
import wandb

wandb.init(project="MDDSPd", entity="user")

model = HTDemucs(sources=['inst1','inst2'], audio_channels=1, samplerate=16000)

sample_rate = 16000
sample_timelength = 5.0

warmup = True

model_num = '0'
batch_size = 16

train_dataset = load_data('URMP')   # ('MDDSP')
val_dataset = load_data('URMP', 'test')   # ('MDDSP', 'test')

device = 'cuda'

from torch_audiomentations import Compose, Gain, PitchShift, ApplyImpulseResponse

# Initialize augmentation callable
apply_augmentation = Compose(
    transforms=[
        Gain(
            min_gain_in_db=-5.0,
            max_gain_in_db=0.0,
            p=0.3,
        ),
    ]
)

separate_augmentation0 = Compose(
    transforms=[
        Gain(
            min_gain_in_db=-3.0,
            max_gain_in_db=3.0,
            p=0.3,
        ),
    ]
)

from torch.utils.data import DataLoader

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dataset_length = len(train_dataset)
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchmetrics import PermutationInvariantTraining
from torchmetrics.functional import scale_invariant_signal_noise_ratio

criterion = PermutationInvariantTraining(scale_invariant_signal_noise_ratio, 'max').to(device)
init_lr = 5e-4
optimizer = optim.Adam(model.parameters(), lr=init_lr)

# ckpt = torch.load(f'model{model_num}_pre/best.pkl', map_location='cpu')
# ckpt_optim = torch.load(f'model3/best_optim.pkl', map_location='cuda')
# model.load_state_dict(ckpt.state_dict())
# optimizer.load_state_dict(ckpt_optim.state_dict())
# optimizer = ckpt_optim
model = model.to(device)

scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

num_epochs = 150
update_epochs = [30,60,90,120]

torch.save(model, f'model{model_num}/01.pkl')
best_val_loss = 1e10
steps = 0

for epoch in range(num_epochs):  # loop over the dataset multiple times
    # training step
    model.train()
    running_loss = 0.0
    total_loss = 0.0
    max_norm = 5
    
    for i, data in enumerate(trainloader, 0):
        steps += 1
        if (steps < 300) and warmup:
            # learning rate warmup
            lr = init_lr*steps/300
            for g in optimizer.param_groups:
                g['lr'] = lr
                
        # get the inputs; data is a list of [inputs, labels]
        labels = data/(data.mean(axis=[1,2])+1e-16).unsqueeze(1).unsqueeze(2)
        labels = apply_augmentation(labels)
        labels = torch.cat([separate_augmentation0(labels[:,0:1,:]), 
                            labels[:,1:2,:]], axis=1)
        
        inputs = labels.sum(axis=1).unsqueeze(1)
        
        labels = labels.to(device)
        inputs = inputs.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs).squeeze(dim=2)
        loss = -criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        total_loss += loss.item()
        
        if i % 50 == 49:    # print every 2000 mini-batches
            print(f'\r[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 5:.5f}')
            wandb.log({"loss": running_loss/5, 
                       "learning_rate": optimizer.param_groups[0]['lr'], 
                       "steps": steps, "epoch": epoch+1})
            running_loss = 0.0
        elif i % 5 == 4:    # print every 100 mini-batches
            print(f'\r[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 5:.5f}', end="")
            wandb.log({"loss": running_loss/5, 
                       "learning_rate": optimizer.param_groups[0]['lr'], 
                       "steps": steps, "epoch": epoch+1})
            running_loss = 0.
        
                
    #after training each epoch
    print(f'\r[{epoch + 1}, {i + 1:5d}] loss: {total_loss / (i+1):.5f}')
    wandb.log({"total_loss": total_loss/(i+1), "steps":steps, "epoch": epoch+1})
    
    # validation step
    model.eval()
    val_loss = 0.0
    total = 0
    
    running_val_loss = 0.0
    with torch.no_grad():
        for data in valloader:
            # labels = data * scale
            labels = data/(data.mean(axis=[1,2])+1e-16).unsqueeze(1).unsqueeze(2)
            inputs = labels.sum(axis=1)
            # labels = data/(data.mean(axis=[1,2])+1e-16).unsqueeze(1).unsqueeze(2)
            # labels = apply_augmentation(labels)
            # labels = torch.cat([separate_augmentation0(labels[:,0:1,:]),
            #                     labels[:,1:2,:]], axis=1)
            
            labels = labels.to(device)
            inputs = inputs.to(device).unsqueeze(1)
            
            outputs = model(inputs).squeeze(dim=2)
            
            val_loss = -criterion(outputs, labels)
            running_val_loss += val_loss.item()*inputs.shape[0]
            total += inputs.shape[0]
    
    print(f'\rval [{epoch + 1}, {total + 1:5d}] loss: {running_val_loss / total:.5f}')
    wandb.log({"val_loss": running_val_loss/total,
               "learning_rate": optimizer.param_groups[0]['lr'], 
               "steps": steps, "epoch": epoch+1})
        
    if running_val_loss <= best_val_loss:
        torch.save(model, f'model{model_num}/best.pkl')
        torch.save(optimizer, f'model{model_num}/best_optim.pkl')
        best_val_loss = running_val_loss
        
    if epoch in update_epochs:
        print('Updating Learning Rate')
        scheduler.step()
        with open(f"model{model_num}.txt", "a") as file:
            file.write(f"Updating Learning Rate at epoch {epoch + 1} to {optimizer.param_groups[0]['lr']}\n")
            
torch.save(model, f'model{model_num}/final.pkl')
torch.save(optimizer, f'model{model_num}/final_optim.pkl')

print('Finished Training')

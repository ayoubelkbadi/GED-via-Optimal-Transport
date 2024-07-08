# GED Computation with Optimal Transport
Source codes and appendices of 'Computing Approximate Graph Edit Distance via Optimal Transport'

## Requirements
All codes are implemented in python3.9

```
dgl                       1.0.2
pot                       0.8.2
networkx                  3.1
numpy                     1.26.4
scipy                     1.12.0
pytorch                   2.2.2+cpu
pyg                       2.5.2 
torchvision               0.17.2
texttable                 1.6.4
tqdm                      4.65.0
```
## Code Running
### Datasets
The datasets we use are AIDS, Linux and IMDB from [GEDGNN](https://github.com/ChengzhiPiao/GEDGNN/tree/master)

### Training
An example of training GEDIOT on AIDS for 20 epochs
```
python src/main.py --model-name GEDIOT --dataset AIDS --model-epoch-start 0 --model-epoch-end 20 --model-train 1
```
The parameter `model-name` can be replaced by `GedGNN`, `TaGSim`, `GPN` and `SimGNN`
### Testing
An example of testing GEDIOT, GEDHOT and GEDGW on AIDS. GEDIOT and GEDHOT use the 20-th epoch model
```
python src/main.py --model-name GEDIOT --dataset AIDS --model-epoch-start 20 --model-epoch-end 20 --model-train 1 --path
python src/main.py --model-name GEDHOT --dataset AIDS --model-epoch-start 20 --model-epoch-end 20 --model-train 1 --GW --path
python src/main.py --model-name GEDGW --dataset AIDS --GW --path
```

# Sparse Measure

Sparse Measure is a companion toy example for our tutorial ["Hands-on Bayesian Neural Networks - A Tutorial for Deep Learning Users"](https://doi.org/10.1109/MCI.2022.3155327). It aims at showing how different learning strategies can be implemented for a BNN.

## Dependancies

The code depends on: 

- numpy (tested with version 1.19.2), 
- scipy (tested with version 1.3.1), 
- pytorch (tested with version 1.8.1),
- matplotlib (tested with version 3.1.1),

and one library from the base python distribution: argparse.

It has been tested with python 3.6.9.

## Usage

The project is split into multiple files:

- dataset.py implement the primitives to generate the dataset.
- experiment.py contain a small script to run the experiment which generated figure 16 in the paper
- viModel.py implement the variational inference layers and model we are using.
- viExperiment.py is the script running the actual experiment. It can be called with the -h option to get a contextual help message:

	python viExperiment.py -h

## Citation

If you use our code in your project please cite our tutorial:

	@ARTICLE{9756596,
	author={Jospin, Laurent Valentin and Laga, Hamid and Boussaid, Farid and Buntine, Wray and Bennamoun, Mohammed},
	journal={IEEE Computational Intelligence Magazine}, 
	title={Hands-On Bayesian Neural Networks—A Tutorial for Deep Learning Users}, 
	year={2022},
	volume={17},
	number={2},
	pages={29-48},
	doi={10.1109/MCI.2022.3155327}
	}

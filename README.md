# Sparse Measure

Sparse Measure is a companion toy example for our tutorial "Hands-on Bayesian Neural Networks - a Tutorial for Deep Learning Users". It aims at showing how different learning strategies can be implemented for a BNN.

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

	@article{DBLP:journals/corr/abs-2007-06823,
	author    = {Laurent Valentin Jospin and
				Wray L. Buntine and
				Farid Boussa{\"{\i}}d and
				Hamid Laga and
				Mohammed Bennamoun},
	title     = {Hands-on Bayesian Neural Networks - a Tutorial for Deep Learning Users},
	journal   = {CoRR},
	volume    = {abs/2007.06823},
	year      = {2020},
	url       = {https://arxiv.org/abs/2007.06823},
	archivePrefix = {arXiv},
	eprint    = {2007.06823},
	timestamp = {Tue, 21 Jul 2020 12:53:33 +0200},
	biburl    = {https://dblp.org/rec/journals/corr/abs-2007-06823.bib},
	bibsource = {dblp computer science bibliography, https://dblp.org}
	}

(The final peer reviewed paper still under revision and should be available soon) 

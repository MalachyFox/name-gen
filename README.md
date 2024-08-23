# ML Interns Intro 2023

This repo contains introductory materials and instructions aimed at getting you
familiar with the concepts needed to apply and research deep learning.

## Before you start

It's a good idea to commit your work to version control (in our case git), so
you don't lose any previous work. You can clone this repository, make a new
branch (`git checkout -b my-branch-name`), and commit your work there.

A lot of what you'll be working on in this introduction will be exploratory, so
I recommend doing your work in Jupyter notebooks.

It's common to need to install different versions of Python libraries for
different projects, so rather than install them all into your global system
environment, you will want to use some tool for isolating environments such as:

- [venv](https://docs.python.org/3/library/venv.html)
  - Most readily available option, but least support for creating reproducible
    environments, and the version of Python itself will be fixed.
- [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)
  - Gives you the option of installing different versions of Python.
  - Note that you can use conda just for creating environments but continue to
    use pip for install packages into those environments.
- [docker](https://docs.docker.com/get-docker/)
  - Most heavyweight option. You can build up almost an entirely separate
    operating system.

In the case of Docker, you'll need to write a Dockerfile to define your
environment. For venv and conda, it's also good practice for reproducibility to
create a `requirements.txt` file with a list of any Python packages you
installed.

## Getting started with PyTorch

If you've never used PyTorch before, start by following this [quick start
tutorial](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html).

## Training a language model

A lot of the machine learning at Myrtle is focused on language based tasks (e.g.
speech recognition, speech synthesis). In this introduction, we will take a look
at language models, which assign probabilities to sentences (or more generally
utterances), usually by assigning a probability for a word (or more generally a
token, which could be a character, or a subword) given all the words preceding it.

[This video](https://www.youtube.com/watch?v=TCH_1BHY58I) (accompanying code
[here](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part2_mlp.ipynb))
by Andreij Karpathy walks through a simple neural language model based on [this
paper](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf). Start by
making sure you understand it and can reproduce it. There are a lot of things
that can go wrong when train a machine learning model, and many of them are not
obvious, so it's always good to start from a known working baseline! Once you
have it working, try the following exercises:

- Train your model on a different dataset. There are many datasets available on
  [Hugging Face](https://huggingface.co/datasets) such as the works of
  Shakespeare, Enron emails, Yelp reviews, etc. Any large body of text should be
  possible to train on!
- Search over some hyperparameters. This could be model size (number of layers,
  size of each layer), context length, learning rate, etc. Draw some plots of
  the results using `matplotlib`.
- It's useful to be able to see the results of your training in real time, and
  to be able to compare them to past training experiments. Add
  [tensorboard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)
  integration to do this. At a minimum, enable tracking of training and
  validation losses.
- Read about RNNs and LSTMs. There are some good resources here:
  - [Andreij Karpathy's blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
  - [Chris Olah's
    blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
  - [Fast.ai](https://github.com/fastai/fastbook/blob/master/12_nlp_dive.ipynb)
  Replace your MLP model with an LSTM based model. Can you train it to be better
  than your MLP model?
- Language models are often trained because they use self-supervised learning,
  where the only training data you need is unlabelled text. Compare this to the
  supervised learning used for a text classification task, where training data
  needs to have been labelled, usually by a human. Labelled data is much harder
  to come by. An effective way to train a model when there is limited training
  data is to first traing the model on a somewhat related task where more data
  is available before finetuning on the downstream task. This is called a
  transfer learning.

  First train a model on a text classification task (again, there are many
  datasets available on Hugging Face).

  Can you use transfer learning to do any better than your baseline?

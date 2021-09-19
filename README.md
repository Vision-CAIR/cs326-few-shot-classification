## CS 326 Low Resource Deep Learning
### Coding assignment #2
In this assignment you are asked to implement 3 algorithms for few-shot learning and compare their performance between each other:
- an unpretrained baseline
- a pretrained baseline
- Prototypical Networks
- MAML

This assignment assumes that you are familiar with these algorithms and pytorch.
A GPU card is not strictly required, but highly recommended.
Otherwise, experimenting and debugging will take you a lot of time.

The unpretrained baseline is already implemented for you and you can launch it by calling
```
CUDA_VISIBLE_DEVICES=0 python run_experiment.py -m unpretrained_baseline
```

### Instructions
Please refer to the [instructions file](instructions/instructions.pdf).

### Code structure
Method hyperparameters are supposed to be stored in `utils/config.py`.
The main entry-point is `run_experiment.py` which launches experiments.
The most "diffucult" file in the template is `utils/data.py` which loads data and generates training episodes.
This episode generation is quite cumbersome, but it shouldn't be necessary to dive too deep into it.
Each training episode consists on `ds_train` and `ds_test` which are used to train the model on a given task.
`ds_test` is used for outer loop update only, hence MAML is the only method that uses it.

While implementing MAML, you may face the problem when using `nn.Module` since we need to be able to perform a forward pass with parameter values that are different from what a module currently has.
That's why we provided `models/pure_layers.py` for you and `PureLeNet` class in `models/lenet.py`.
You may like to read [this article](https://sjmielke.com/jax-purify.htm) to get the idea of pure functions in deep learning.
You are also _encouraged_ to check open-source MAML implementations to get the idea of what the problem with `nn.Module` is and why we need to reinvent this wheel.

`trainers/trainer.py` is responsible for training/evaluating baselines.
Since there is a lot of code sharing, ProtoNet and MAML trainers inherit from it and override several things.

Other files should be self-explanatory.

You can search for `TODO` keyword to see what pieces are missing.

### Contributing
Create an issue or send a PR if you found a bug or want to propose an update.

<!-- ### Copyright and licensing
Copyright is by Ivan Skorokhodov, Jun Chen and Mohamed Elhoseiny @ KAUST university.
We will be happy if you'll use this assignment in your own course and will be even more happy if you'll specify the source.
For a version with the solution, email iskorokhodov@gmail.com. -->

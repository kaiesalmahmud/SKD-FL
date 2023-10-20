# Adaptive Semi-supervised Federated Learning with Selective Knowledge Distillation

This work builds on the work of [FedMD](https://arxiv.org/abs/1910.03581)

### Hypothesis 01 : Introducing poisonous nodes in the federated network can have negative impact on collaborative learning

#### Experiemnt: Introduce varying number of poisoned node in all four environments of FedMD an observe the impact on collaborative learning.

#### Results:

FEMNIST Balanced:

![Poison variation: FEMNIST Balanced](thesis-fig/poison-var/femnist-balanced.png)

FEMNIST Imbalanced:

![Poison variation: FEMNIST Imbalanced](thesis-fig/poison-var/femnist-imbalanced.png)

CIFAR Balanced:

![Poison variation: CIFAR Balanced](thesis-fig/poison-var/cifar-balanced.png)

CIFAR Imbalanced:

![Poison variation: CIFAR Imbalanced](thesis-fig/poison-var/cifar-imbalanced.png)


### Hypothesis 02 : Selective Knowledge Distillation (SKD) can minimize the impact of poisoned nodes in collaborative learning

Framework:

![Framework](thesis-fig/structure.png)


#### Experiemnt: We test SKD algorithm on all four environements with 40% nodes poisoned

#### Results:

FEMNIST Balanced:

![SKD on Supervised Setting: FEMNIST Balanced](thesis-fig/skd/femnist-balanced.png)

FEMNIST Imbalanced:

![SKD on Supervised Setting: FEMNIST Imbalanced](thesis-fig/skd/femnist-imbalanced.png)

CIFAR Balanced:

![SKD on Supervised Setting: CIFAR Balanced](thesis-fig/skd/cifar-balanced.png)

CIFAR Imbalanced:

![SKD on Supervised Setting: CIFAR Imbalanced](thesis-fig/skd/cifar-imbalanced.png)

### Hypothesis 03 : Selective Knowledge Distillation (SKD) will also work on semi-supervised setting

Semi-supervised learning methodology:

![Semi-supervised Algorithm](thesis-fig/semi-flow.png)


#### Experiemnt: We test SKD algorithm on all four environements with 40% nodes poisoned

#### Results:

FEMNIST Balanced:

![SKD on Semi-supervised Setting: FEMNIST Balanced](thesis-fig/ssfd/femnist-balanced.png)

FEMNIST Imbalanced:

![SKD on Semi-supervised Setting: FEMNIST Imbalanced](thesis-fig/ssfd/femnist-imbalanced.png)

CIFAR Balanced:

![SKD on Semi-supervised Setting: CIFAR Balanced](thesis-fig/ssfd/cifar-balanced.png)

CIFAR Imbalanced:

![SKD on Semi-supervised Setting: CIFAR Imbalanced](thesis-fig/ssfd/cifar-imbalanced.png)
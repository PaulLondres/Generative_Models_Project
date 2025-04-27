# Generative Models Project

This repository contains the data and code for the Generative Models project, part of the second semester of the MVA program.

## Project Overview

The goal of this project is to explore and study the method proposed in https://arxiv.org/abs/2201.11793. 
We also wanted to compare it with a conditionnal diffusion model, train specifically for deblurring.

## Directory Structure

- `data/`: Contains the datasets used for training and evaluating the generative models.
- `src/`: Contains the source code for the project divided in half, one for the conditionnal and one for the unconditionnal model

## Results with DDRM unconditionned model
Here are some examples of deblurring with the unconditionned model : 
![alt text](https://github.com/PaulLondres/Generative_Models_Project/blob/main/figures/other_examples.png?raw=true)

Deblurring results when we vary the amount of noise of the inverse problem : 
![alt text](https://github.com/PaulLondres/Generative_Models_Project/blob/main/figures/imgs_sigs_tight.png?raw=true)

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.


# SparseMate

A work in progress research project in applying modern mechanistic interpretability techniques designed for LLMs to transformer chess models.

Techniques include using Sparse Autoencoders to develop decompositions of features in the residual stream, and linear probing to validate findings. Future work may include integrating SAEs and linear probes in the hope of enforcing human-interpretable features, expanding linear probing to produce general linear subspaces, and analyzing the effects of relative chess piece placement on embeddings.

Repository includes modified code from Jenner et al. [@jenner2024evidence] and by Marks et al. [@marks2024dictionary_learning].

Additional models and data from Jenner et al. [@jenner2024evidence] and from Ruoss et al. [@ruoss2024amortized].

Note: Project has grown organically over time. Enter this repository at your own risk!

## Citations

```bibtex
@misc{jenner2024evidence,
  title={Evidence of Learned Look-Ahead in a Chess-Playing Neural Network}, 
  author={Erik Jenner and Shreyas Kapur and Vasil Georgiev and Cameron Allen and Scott Emmons and Stuart Russell},
  year={2024},
  eprint={2406.00877},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@misc{marks2024dictionary_learning,
  title = {dictionary_learning},
  author = {Samuel Marks, Adam Karvonen, and Aaron Mueller},
  year = {2024},
  howpublished = {\url{https://github.com/saprmarks/dictionary_learning}},
}

@inproceedings{ruoss2024amortized,
  author       = {Anian Ruoss and Gr{\'{e}}goire Del{\'{e}}tang and Sourabh Medapati and Jordi Grau{-}Moya and Li Kevin Wenliang and Elliot Catt and John Reid and Cannada A. Lewis and Joel Veness and Tim Genewein},
  title        = {Amortized Planning with Large-Scale Transformers: A Case Study on Chess},
  booktitle    = {NeurIPS},
  year         = {2024}
}

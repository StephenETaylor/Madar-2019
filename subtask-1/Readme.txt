Both the tortuous and neuralNetwork classifiers use kenlm language models,
but the code for building the models is found in the tortuous directory.
Language model data is prepared by the Glm6 tool in the tortuous directory,
and passed manually to the neuralNetwork code.

I *think* I built the kenlm *lmplz* and *build_binary* tools from the 
build instructions at https://kheafield.com/code/kenlm.



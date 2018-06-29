# HRED
The implementation of Hierarchical Recurrent Encoder Decoder network proposed by Serban et, al. (Serban, Iulian Vlad, et al. "Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models." AAAI. Vol. 16. 2016)

You need to prepare your own data files which consists of three text files: dialog_train.txt, dialog_valid.txt, rg_vocab.txt.

The format of dialog_train.txt file and dialog_valid.txt file is:
q1\ta1\tq2\ta2\n
q1\ta1\tq2\ta2\n
...

For each line, there are several utterances (like q1, a1 here) and you can split them by '\t'. For each utterance, I have already conducted word tokenization and you can split each utterance by whitespace to get each word token.

To train the model, you need to firstly use prepare_context_RG_data.py to generate the .tfrecords files. Then you can use train.py to train the model.



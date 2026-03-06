# lm-electrostatics

Plan:

first write out the matrix equations for GPT2 from the beginning list of <input_id> vectors to embedding matrix to attention transformations to ffn to a recursive iteration over multiple layers and all the way up to the last layer and the final weight matrix multiplication to get the prob distribution to generate the next token, i.e., P(x_{t+1}|P(x_{1:t})). I want clearly defined matrix parameters, their dimensions, etc. Be very thorough because the conventions you choose here will stay with you forever. 
For example, it will be something like 
P(x_{t+1}|P(x_{1:t})) = softmax[W Layer_L(Layer_{L-1}(Layer_{L-2}(....(Embedding(one_hot(input_id_seq))))))]

save your equations into a file called `gpt2_equations.md`

And then there will be individual equations like Layer_i(X) = X + ....ffn...MHA...

I might be wrong in certain things but you should be in perfect alignment with the gpt2 model and all their choices

Assuming the following 

S: Sequence Length 
B: Batch Size
H: Hidden Dimension / Model Dimension 
N: Number of Attention Heads
L: Number of layers

Assume that the batch size is 1 here, and any kind of layernorm is represented by appropriate parameters- something like X_l --> LayerNorm(X_l), where X_l is a H x S tensor that comes out of every layer and goes straight into the next layer.

Now assume that X_l is the tensor coming out of each layer and there are L layers.

I can flatten X_l (output of layer l) and X_0 (input to layer 1) and define X_l (H x S, 1) as a vector function of X_0 (H x S, 1). I want to decompose X_l(X_0) into a conservative and non-conservative part. I know that H x S is very high, but you will use pytorch and give me divergences of X_l(X_0) numerically. I want to see how this divergence grows as a function of a given input sequence and as a function of layers l and for various sequences (take a few sequences in and out of gpt2's training corpus). I will ask you to do more things later, but let us start with this.


I have done poetry init (look at my toml) and do the necessary poetry add to get started.


load a pre-trained language model (for now take a random transformer model like gpt2 with scaled down params, i.e., S, H, N, L are so small that one can use a CPU) for testing 

your code roughly will have a function to calculate divergences and a main() loop that takes in 5 sentences from gpt2's training data and 5 sentences outside gpt2's training data and prints out the divergences. You should also calculate the perplexity and plot divergence versus perplexity.

My grand goal is to derive a mathematical relation between perplexity and divergence, but you shouldn't worry about it until I tell you. 

the eventual goal is to investigate the structure of the vector fields of the outputs of the final layer as a function of the inputs of the first layer (output of the initial embedding I mean), and split vector fields into conservative and non-conservative components, and come up with notions such as charge and electrostatic potential (assuming the non-conservative part is small). But I will tell you all this in detail later, don't get distracted by these things. 

Note: The default device should be auto so that I can test it here and then run it on a cpu. 

ok now execute all the steps and generate code for me. Be careful about the autodiffs because these are non-trivial and typically people take grads wrt params and not the input of the bottom most layers. I mean the gradients flowing to the weights of all the layers will show up in the gradients using chain rule but please be thorough and follow the equations you write in the `gpt2_equations.md` clearly.
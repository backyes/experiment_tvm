from mog_lstm import MogrifierLSTMCell
import torch
import torch.nn as nn

# https://github.com/fawazsammani/mogrifier-lstm-pytorch
        
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, mogrify_steps, vocab_size, tie_weights, dropout):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.mogrifier_lstm_layer1 = MogrifierLSTMCell(input_size, hidden_size, mogrify_steps)
        self.mogrifier_lstm_layer2 = MogrifierLSTMCell(hidden_size, hidden_size, mogrify_steps)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.drop = nn.Dropout(dropout)
        if tie_weights:
            self.fc.weight = self.embedding.weight
        
    def forward(self, seq, max_len = 3):
        
        embed = self.embedding(seq)
        batch_size = seq.shape[0]
        h1,c1 = [torch.zeros(batch_size,self.hidden_size), torch.zeros(batch_size,self.hidden_size)]
        h2,c2 = [torch.zeros(batch_size,self.hidden_size), torch.zeros(batch_size,self.hidden_size)]
        hidden_states = []
        outputs = []
        for step in range(max_len):
            x = self.drop(embed[:, step])
            h1,c1 = self.mogrifier_lstm_layer1(x, (h1, c1))
            h2,c2 = self.mogrifier_lstm_layer2(h1, (h2, c2))
            out = self.fc(self.drop(h2))
            hidden_states.append(h2.unsqueeze(1))
            outputs.append(out.unsqueeze(1))
            

        hidden_states = torch.cat(hidden_states, dim = 1)   # (batch_size, max_len, hidden_size)
        outputs = torch.cat(outputs, dim = 1)               # (batch_size, max_len, vocab_size)
        
        return outputs, hidden_states 

input_size = 512
hidden_size = 512
vocab_size = 30
batch_size = 4
lr = 3e-3
mogrify_steps = 5        # 5 steps give optimal performance according to the paper
dropout = 0.5            # for simplicity: input dropout and output_dropout are 0.5. See appendix B in the paper for exact values
tie_weights = True       # in the paper, embedding weights and output weights are tied
betas = (0, 0.999)       # in the paper the momentum term in Adam is ignored
weight_decay = 2.5e-4    # weight decay is around this value, see appendix B in the paper
clip_norm = 10           # paper uses cip_norm of 10

model = Model(input_size, hidden_size, mogrify_steps, vocab_size, tie_weights, dropout)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=1e-08, weight_decay=weight_decay)

# seq of shape (batch_size, max_words)
seq = torch.LongTensor([[ 8, 29, 18,  1, 17,  3, 26,  6, 26,  5],
                        [ 8, 28, 15, 12, 13,  2, 26, 16, 20,  0],
                        [15,  4, 27, 14, 29, 28, 14,  1,  0,  0],
                        [20, 22, 29, 22, 23, 29,  0,  0,  0,  0]])

outputs, hidden_states = model(seq)
print(outputs.shape)
print(hidden_states.shape)


#--trace, error, FIX with calling model.eval()
#/Users/shwangyanfei/opt/miniconda3/lib/python3.9/site-packages/torch/jit/_trace.py:992: TracerWarning: Output nr 1. of the traced function does not match the corresponding output of the Python function. Detailed error:
#Tensor-likes are not close!
#
#Mismatched elements: 1200 / 1200 (100.0%)
#Greatest absolute difference: 6.078236743807793 at index (3, 8, 5) (up to 1e-05 allowed)
#Greatest relative difference: 451.0588994422156 at index (0, 0, 24) (up to 1e-05 allowed)
#  _check_trace(
#/Users/shwangyanfei/opt/miniconda3/lib/python3.9/site-packages/torch/jit/_trace.py:992: TracerWarning: Output nr 2. of the traced function does not match the corresponding output of the Python function. Detailed error:
#Tensor-likes are not close!
#
#Mismatched elements: 20477 / 20480 (100.0%)
#Greatest absolute difference: 0.1517551690340042 at index (1, 7, 320) (up to 1e-05 allowed)
#Greatest relative difference: 6029.9744357672125 at index (2, 5, 225) (up to 1e-05 allowed)
#  _check_trace(

model.eval()
with torch.no_grad():
	jit_model = torch.jit.trace(model, seq)
#print(jit_model)
#print(jit_model.code)
print(jit_model.graph)
# output, means lstm is unrolling by max_steps, check mogrifier_lstm_layer1 nodes, see graph_lstm_unrolling.log

torch._C._jit_pass_inline(jit_model.graph)
print(jit_model.graph)

# 将模型序列化
jit_model.save('jit_model.pth')
# 加载序列化后的模型
jit_model = torch.jit.load('jit_model.pth')

import torch
import numpy as np


class ConvLSTMRLTestModel(torch.nn.Module):
  def __init__(self, num_acts):

class ConvLSTMLayer(torch.nn.Module):
  def __init__(self, num_ins, num_hids):
    super().__init__()

    # Number of inputs into net when accounting for hidden state being concatenated to input
    cat_ins = num_ins + num_hids

    self.conv_f = torch.nn.Conv2d(cat_ins, num_hids, 1)
    self.conv_u = torch.nn.Conv2d(cat_ins, num_hids, 1)
    self.conv_o = torch.nn.Conv2d(cat_ins, num_hids, 1)
    self.conv_c = torch.nn.Conv2d(cat_ins, num_hids, 1)
    
    


  def forward(self, x):
    '''
      Takes tensor of shape (L,N,C,H,W) with:
         L: timesteps
         N: batch size
         C: number of channels
         H: height
         W: width
    '''
    prev_out = torch.zeros_like(x[0])
    prev_state = torch.zeros_like(x[0])
    all_outs = []
    all_states = []
    for i in range(x.shape[0]):
      prev_out, prev_state = self.forward_step(x[i], prev_out, prev_state)
      all_outs.append(prev_out.unsqueeze(0))
      all_states.append(prev_state.unsqueeze(0))
    return torch.cat(all_outs, dim=0), torch.cat(all_states, dim=0)
    

  def forward_step(self, x_t, out_prev, state_prev):
    # concat along channel dim (it's rank 4 tensor)
    x = torch.cat([x_t, h_prev], dim=1)

    # Forget gate
    f_t = F.sigmoid(self.conv_f(x))

    # update gate
    u_t = F.sigmoid(self.conv_u(x))

    # Output gate
    o_t = F.sigmoid(self.conv_o(x))

    # Candidate state
    cs_t = F.tanh(self.conv_c(x))

    # New state
    state = f_t * state_prev + u_t * cs_t

    # Output
    out = o_t * F.tanh(state)

    return (out, s_t)



class TestModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.l1 = torch.nn.Linear(32, 64)
    self.l2 = torch.nn.Linear(64,128)



class ModelWrapper:
  def __init__(self, model):
    self.model = model

  def forward(x):
    return self.model.forward(x)
  

  def get_params(self):
    flat_params = []
    for p in self.model.parameters():
      flat_params.append(p.flatten())
    return torch.cat(flat_params,dim=0)

  def set_params(self, param_vec):
    start = 0
    for p in self.model.parameters():
      param_shape = p.data.shape
      num_params = np.prod(list(param_shape))
      p.data = torch.Tensor(param_vec[start:start+num_params]).reshape_as(p.data)
      start += num_params


if __name__=="__main__":
  m = ConvLSTMLayer(3, 256)

  w = ModelWrapper(m)

  params = w.get_params()

  print(list(w.model.parameters())[0])

  w.set_params(params + params)

  print(list(w.model.parameters())[0])
 


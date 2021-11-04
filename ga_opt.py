import numpy as np


class GeneticAlgorithm:
  def __init__(self, num_params, num_pop=100, child_mut_prob=0.2, max_mut_port=0.1, max_mut_mag=0.1,
                tournament_size=100):
    self.num_params = num_params
    self.num_pop = num_pop
    self.pops = [np.random.rand(num_params) - 0.5 for _ in range(num_pop)]
    self.parent_probs = []
    self.best = None

    self.child_mut_prob = child_mut_prob
    self.max_params_to_mut = int(np.ceil(max_mut_port * num_params))
    self.max_mut_mag = max_mut_mag
    self.tournament_size = tournament_size
    

  def step(self, eval_func):
    fits = []
    for pop in self.pops:
      fits.append(eval_func(pop))

    pops_ordered = [pop for pop,fit in sorted(zip(self.pops, fits), reverse=True, key=lambda p_f: p_f[1])]
    self.best = pops_ordered[0]

    new_pops = []
    for i in range(self.num_pop):
      p1 = self.select_parent(pops_ordered)
      p2 = self.select_parent(pops_ordered)
      child = self.cross(p1,p2)
      if np.random.rand() < self.child_mut_prob:
        child = self.mutate(child)
      new_pops.append(child)

    self.pops = new_pops

  def select_parent(self, cand_list):
    # Using tournament selection
    # Since candidates are sorted by fitness selecting the min index is the same as selecting highest fit
    index = np.min(np.random.choice(range(len(cand_list)), size=self.tournament_size))
    return cand_list[index]


  def cross(self, p1, p2):
    cross_points = list(np.random.choice(range(self.num_params), size=int(np.ceil(self.num_params / 1000))))
    cross_points.sort()

    child = []
    p_copy = p1
    p_skip = p2
    for start, cp in zip([0] + cross_points, cross_points + [self.num_params]):
      child.append(p_copy[start:cp])
      p_copy, p_skip = p_skip, p_copy

    return np.concatenate(child)
      
  
  def mutate(self, pop):
    if self.max_params_to_mut > 1:
      num_to_mut = np.random.randint(1, self.max_params_to_mut)
    else:
      num_to_mut = 1

    ind_to_mut = np.random.choice(range(self.num_params), size=num_to_mut)

    for i in ind_to_mut:
      pop[i] = pop[i] + (2 * np.random.rand() - 1) * self.max_mut_mag
    return pop      


  def getBest(self):
    return self.best



def  test_func(x):
  return sum(map(lambda x: -(x-1)**2, x))




if __name__=="__main__":
  ga = GeneticAlgorithm(4000, 100)

  ga.step(test_func)

  print(ga.getBest())

  for i in range(10000):
    ga.step(test_func)
    if i % 20 == 0:
      print(ga.getBest())


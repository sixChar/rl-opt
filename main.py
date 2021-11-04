import retro


games = [
          'SuperMarioWorld2-Snes',
          'BeamRider-Atari2600',
          'UpNDown-Atari2600',
          'Enduro-Atari2600',
          'RoadRunner-Atari2600',
          'TimePilot-Atari2600',
          'Boxing-Atari2600',
          'Breakout-Atari2600',
          'Bowling-Atari2600',
          'Frostbite-Atari2600',
          'Zaxxon-Atari2600',
          'Hero-Atari2600',
          'JourneyEscape-Atari2600',
          'Robotank-Atari2600',
          'ElevatorAction-Atari2600',
          'Phoenix-Atari2600',
          'Qbert-Atari2600',
          'AirRaid-Atari2600',
          'Jamesbond-Atari2600',
          'ChopperCommand-Atari2600',
          'Kangaroo-Atari2600',
          'FishingDerby-Atari2600',
          'Freeway-Atari2600',
          'MontezumaRevenge-Atari2600',
          'Alien-Atari2600',
          'CrazyClimber-Atari2600',
          'StarGunner-Atari2600',
          'Solaris-Atari2600',
          'PrivateEye-Atari2600'
]



ROMS_FOLDER = '/home/colm/roms/'


def run_game(game):
  env = retro.make(game=game)
  obs_shape = env.observation_space.shape
  act_shape = env.action_space.shape

  env.close()
  '''
  obs = env.reset()
  while True:
    obs, reward, done, info = env.step(env.actions_space.sample())
    env.render()
    if done:
      obs = env.reset
  env.close()
  '''

if __name__=="__main__":
  run_game(games[0])
  run_game(games[1])


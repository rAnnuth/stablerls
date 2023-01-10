def default_controller(self,outputs):
  ### 
  # Control Reference
  ### 
  # Input 0 muss liste sein
  Voltage = [330 if self.time < 5 else 680] 
  # Input 1 muss liste sein
  Load = [50 if self.time < 6 else 4000] 
  
  ### 
  # Control Error 
  ### 

  ### 
  # Plant Input
  ### 
  for idx, x in enumerate(self.input):
    if idx == 0:
      self.fmu.setReal([x.valueReference], Load)
    elif idx == 4:
      self.fmu.setReal([x.valueReference], Voltage)
    else:
      #print('{} not implemented as input. Setting to 0.\n'.format(x.name))
      self.fmu.setReal([x.valueReference],[0])
  # store in class
  self.inputs += [[self.time] + Load + [0] + [0] + [0] + Voltage]

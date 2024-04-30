# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:09:27 2024

@author: Jordan
"""

import networkx as nx
from Auxillary.DPAH import DPAH

#%% testing and building networks

G_DPAH = DPAH(N = 10,
      fm=0.5, 
      d=0.05, 
      plo_M=2.5, 
      plo_m=2.5, 
      h_MM=0, 
      h_mm=0, 
      verbose=False)

nx.draw(G_DPAH)
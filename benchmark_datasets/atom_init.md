# Element Embedding Format
## Structure of atom_init
Dictionary -> key = str: atomic number, value = list: property encoding with values [0,1] of length 92 (no. of total categories)

Properties embedded:
- ```embedding[Z][0]``` = if lanthanide/actinide (1 if yes, 0 if no)
  - 1 category
- ```embedding[Z][1:19]``` = onehot(group_number): Atomic Group Number 1-18
  - 18 categories
- ```embedding[Z][19:26]``` = onehot(period_number): Period 1-7
  - 7 categories
- ```embedding[Z][26:36]``` = 10 bins of EN (0.5-4.0)
  - 10 categories
- ```embedding[Z][36:46]``` = 10 bins of Covalent Radius (25 - 250 pm)
  - 10 categories
- ```embedding[Z][46:58]``` = onehot(no. of valence electrons): 1-12
  - 12 categories
- ```embedding[Z][58:68]``` = 10 log-scaled bins of First ionization energy (1.3 - 3.3 eV)
  - 10 categories
- ```embedding[Z][68:78]``` = 10 log-scaled bins of electron affinity (-3 - 3.7 eV)
  - 10 categories
- ```embedding[Z][78:82]``` = onehot(block): enumerate(s, p, d, f)
  - 4 categories
- ```embedding[Z][82:92]``` = 10 log-scaled bin atomic volume (1.5-4.3 cm^3/mol)
  - 10 categories
 

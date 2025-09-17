# pdp_analyser

Interactive tools for analysing potentiodynamic polarisation (PDP) data.  
This package provides functions for plotting and interactively extracting electrochemical parameters such as passive current density, passivation potential, pitting potential, and repassivation potential.
"analysis.py" provides an example script for: calling the excel file containing data, defining sample surface area, producing an overall plot and interactively determining electrochemical parameters

---

## Features
- Plot forward/reverse polarisation scans with log-scaled current density.
- Interactive selection of passive and pitting regions using mouse clicks.
- Automatic calculation of:
  - Passive current density
  - Passivation potential
  - Pitting potential
  - Repassivation potential
- Clean plots with matplotlib integration.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/pdp_analyser.git
cd pdp_analyser
pip install -e .

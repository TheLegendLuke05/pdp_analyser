# analysis.py

from pdp_analyser import overall_plot, variable_extractor

# Path to your Excel file
filepath = r"C:\Users\ansel\OneDrive - University of Leeds\ICorr_Data\conv_3.5wt_pH7_1.xlsx"

#Area of sample (cm^2)
area = 4.9

#Call functions
overall_plot(filepath, area)

variable_extractor(filepath, area)
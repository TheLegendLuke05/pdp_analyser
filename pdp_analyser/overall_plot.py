def overall_plot(filepath, area):

    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    """
    Loads polarisation data from Excel, splits it at max voltage,
    and plots current vs. voltage in two segments.
    """
    # Load data
    df = pd.read_excel(filepath, usecols=[0, 1])
    df.columns = ['Voltage', 'Current']
    df['Current'] = df['Current'].abs()

    # Convert units
    df['Voltage_mV'] = df['Voltage']*1000
    df['CurrentDensity_uA_cm2'] = df['Current'] * 1000 / area

    # Find max voltage index
    max_voltage_index = df['Voltage'].idxmax()

    # Split data
    df_before = df.loc[:max_voltage_index]
    df_after = df.loc[max_voltage_index + 1:]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(df_before['CurrentDensity_uA_cm2'], df_before['Voltage_mV'], color='blue', label='Forward Scan')
    plt.scatter(df_after['CurrentDensity_uA_cm2'], df_after['Voltage_mV'], color='red', label='Reverse Scan')

    plt.xlabel('Current Density [µA/cm²]')
    plt.ylabel('Voltage [mV vs Ag/AgCl]')
    plt.title('Polarisation Curve')
    plt.grid(True)
    plt.xscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)

    pass
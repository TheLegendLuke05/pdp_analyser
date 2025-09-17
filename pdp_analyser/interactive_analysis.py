def variable_extractor(filepath, area):
    """
    Allows user to select the passive region and pitting region on the Voltage vs log(Current) plot.
    Fits straight lines to each, overlays them, calculates passive current, pitting potential, and repassivation potential.
    """

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    # Load and process data
    df = pd.read_excel(filepath, usecols=[0, 1])
    df.columns = ['Voltage', 'Current']
    df['Voltage'] *= 1000
    df['Current'] = df['Current'].abs()
    df['Current'] = (df['Current'] / area) * 1000

    # Find max voltage index and slice
    max_voltage_index = df['Voltage'].idxmax()
    df_before = df.loc[:max_voltage_index].copy()
    df_after = df.loc[max_voltage_index:].copy()

    #Find OCP and Transition Potential
    def get_middle_voltage_of_min_current(df):
        min_current = df['Current'].min()
        matching_rows = df[df['Current'] == min_current]
        middle_idx = matching_rows.index[len(matching_rows) // 2]
        return df.loc[middle_idx, 'Voltage']
    
    ocp_voltage = get_middle_voltage_of_min_current(df_before)
    transition_voltage = get_middle_voltage_of_min_current(df_after)


    # Drop nonpositive currents and take log
    df_before = df_before[df_before['Current'] > 0]
    df_after = df_after[df_after['Current'] > 0]

    # Plot both regions
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df_before['Current'], df_before['Voltage'], s=15, label='Forward Scan', color='paleturquoise')
    ax.scatter(df_after['Current'], df_after['Voltage'], s=15, label='Reverse Scan', color='darksalmon')
    ax.set_xlabel('Current Density [µA/cm²]')
    ax.set_xscale('log')
    ax.set_ylabel('Voltage [mV vs Ag/AgCl]')
    ax.grid(True)
    plt.tight_layout()

    #Plot OCP and Transition
    ax.scatter(df_before['Current'].min(), ocp_voltage, color='blue', s=60, marker='x', label=f'Corrosion Potential = {ocp_voltage:.3f} mV')
    ax.scatter(df_after['Current'].min(), transition_voltage, color='red', s=60, marker='x', label=f'Transition Potential = {transition_voltage:.3f} mV')    

    def get_range_by_clicks(df, region):
        x1, y1 = region[0]
        x2, y2 = region[1]
        distances = ((np.log10(df['Current']) - np.log10(x1))**2 + (df['Voltage'] - y1)**2,
                    (np.log10(df['Current']) - np.log10(x2))**2 + (df['Voltage'] - y2)**2)
        idx1 = np.argmin(distances[0])
        idx2 = np.argmin(distances[1])
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        return df.iloc[idx1:idx2 + 1]

    # Select Passive Region
    ax.set_title('Select 2 points defining the PASSIVE region')
    print("🧊 Select 2 points defining the PASSIVE region (left to right)")
    passive_points = plt.ginput(2, timeout=-1)
    df_passive = get_range_by_clicks(df_before, passive_points)
    passive_current_avg = df_passive['Current'].mean()
    print(f"📉 Average passive current density = {passive_current_avg:.2f} µA/cm²")

    # Fit Passive
    fit_passive = np.polyfit(np.log10(df_passive['Current']), df_passive['Voltage'], 1)
    x_linear = np.logspace(np.log10(df_passive['Current'].min()), np.log10(df_passive['Current'].max()), 100)
    y_fit = fit_passive[0] * np.log10(x_linear) + fit_passive[1]
    ax.plot(x_linear, y_fit, color='springgreen', label='Passive Fit', linewidth=2.5)

    #Add Passivation Current Density
    log_passive_current_avg = np.log10(passive_current_avg)
    voltage_at_passive_current = fit_passive[0] * log_passive_current_avg + fit_passive[1]
    ax.plot(passive_current_avg, voltage_at_passive_current, 'xg', 
            label=f'Passive Current Density = {passive_current_avg:.2f} µA/cm²')
    
    # Find index range of passive region
    x1, y1 = passive_points[0]
    x2, y2 = passive_points[1]
    idxs = df_before.index[((df_before['Current'] >= min(x1, x2)) &
                            (df_before['Current'] <= max(x1, x2)) &
                            (df_before['Voltage'] >= min(y1, y2)) &
                            (df_before['Voltage'] <= max(y1, y2)))]
    idx_start, idx_end = idxs.min(), idxs.max()

    # Iteratively expand lower bound while R^2 >= 0.9
    from sklearn.metrics import r2_score
    best_idx_start = idx_start
    while best_idx_start > 0:
        df_test = df_before.loc[best_idx_start:idx_end]
        X = np.log10(df_test['Current'])
        y = df_test['Voltage']
        fit = np.polyfit(X, y, 1)
        r2 = r2_score(y, np.polyval(fit, X))
        if r2 < 0.95:
            break
        best_idx_start -= 1

    # Get passivation potential
    passivation_potential = df_before.loc[best_idx_start + 1, 'Voltage']
    current_at_passivation = df_before.loc[best_idx_start + 1, 'Current']
    ax.plot(current_at_passivation, passivation_potential, 'xg',
               label=f'Passivation Potential = {passivation_potential:.3f} mV')
    print(f"🧪 Passivation Potential = {passivation_potential:.3f} mV")

    # Show fitted line extended
    ax.plot([x2, 10**((passivation_potential-fit_passive[1])/fit_passive[0])], [(fit_passive[0]*np.log10(x2) + fit_passive[1]), passivation_potential], 
            linestyle='--', color='springgreen', linewidth=2.5)



    plt.draw()

    # Select Pitting Region
    ax.set_title('Select 2 points defining the PITTING region')
    print("🔥 Select 2 points defining the PITTING region (left to right)")
    pitting_points = plt.ginput(2, timeout=-1)
    df_pitting = get_range_by_clicks(df_before, pitting_points)

    # Fit Pitting
    fit_pitting = np.polyfit(np.log10(df_pitting['Current']), df_pitting['Voltage'], 1)
    x_pitting = np.logspace(np.log10(df_pitting['Current'].min()), np.log10(df_pitting['Current'].max()), 100)
    y_pitting = fit_pitting[0] * np.log10(x_pitting) + fit_pitting[1]

    def find_intersection(m1, b1, m2, b2):
        if m1 == m2:
            return None
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
        return x, y

    pitting_intersect = find_intersection(fit_passive[0], fit_passive[1], fit_pitting[0], fit_pitting[1])

    # Final Plot
    ax.plot(x_pitting, y_pitting, color='gold', label='Pitting Fit', linewidth=2.5)
    if pitting_intersect:
        x_int, y_int = pitting_intersect
        ax.plot(10**x_int, y_int, 'kx', label=f'Pitting Potential = {y_int:.3f} mV')

        # Repassivation Line (no label)
        ax.axvline(x=10**x_int, linestyle='--', color='black')

        # Repassivation Potential
        df_after_interp = df_after.copy()
        min_current_index = df_after_interp['Current'].idxmin()
        min_current_voltage = df_after_interp.loc[min_current_index, 'Voltage']
        df_after_interp = df_after_interp[df_after_interp['Voltage'] >= min_current_voltage]
        df_after_interp = df_after_interp.sort_values(by='Current')
        repassivation_voltage = np.interp(x_int, np.log10(df_after_interp['Current']), df_after_interp['Voltage'])
        ax.plot(10**x_int, repassivation_voltage, 'kx', label=f'Repassivation Potential = {repassivation_voltage:.3f} mV')
        print(f"🧯 Repassivation Potential = {repassivation_voltage:.3f} mV")

    ax.legend()
    ax.set_title('Final Plot with Fits and Intercept')
    plt.draw()
    plt.show(block=True)

    return passive_current_avg, pitting_intersect, repassivation_voltage


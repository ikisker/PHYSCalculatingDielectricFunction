import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MaxNLocator
from tkinter import filedialog  # Add this import
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
import os
import warnings

#want to use files marked .tr for import
#tab delimited text file
#ignore data above 3 Thz
# .3ft hav ahad the forier transform already aplied and are capeed to the usable veiv

#add thicness difference perameter of reference an thin film mesure, this causes phase ofset that
# is 2pi*delta / wavelength

# Define initial material parameters for MAPbI3 (from Table 1)
w_TO = [63, 32]
g_TO = [20, 9]
w_LO = [133, 40]
g_LO = [30, 11]
eps_inf = 3.94# high frequency dielectric constant for MAPI
# eps_inf = 4.49# high frequency dielectric constant for MAPI

# Add parameters for lead iodide (PbI2)
# w_TO_pbi2 = [63]  # TO phonon frequency for PbI2
# g_TO_pbi2 = [5]   # Damping constant for TO phonon: guess
# w_LO_pbi2 = [133] # LO phonon frequency for PbI2
# g_LO_pbi2 = [7]   # Damping constant for LO phonon: guess


# Optimization Results (no file imported):
# PbI2 Thickness: 0.1910
# w_TO: 51.8133
# g_TO: 8.6410
# w_LO: 125.7573
# g_LO: 48.7117
# eps_inf: 4.1203
# Magnitude Optimization Cost: 0.0205


# Add parameters for lead iodide (PbI2)
# w_TO_pbi2 = [51.8133]  # TO phonon frequency for PbI2
# g_TO_pbi2 = [8.6410]   # Damping constant for TO phonon: guess
# w_LO_pbi2 = [125.7573] # LO phonon frequency for PbI2
# g_LO_pbi2 = [48.7117]   # Damping constant for LO phonon: guess
# eps_inf_pbi2 =  4.1203 # High-frequency dielectric constant for PbI2


#vals goten from P2 Fir
# Add parameters for lead iodide (PbI2)
w_TO_pbi2 = [54.05]  # TO phonon frequency for PbI2
g_TO_pbi2 = [6.76]   # Damping constant for TO phonon: guess
w_LO_pbi2 = [117.57] # LO phonon frequency for PbI2
g_LO_pbi2 = [58.11]   # Damping constant for LO phonon: guess
eps_inf_pbi2 =  4.19 # High-frequency dielectric constant for PbI2

#higher frequncy 
#larger damping ceofficient


eps_static = 33.5
eps = eps_inf
eps_pb2 = eps_inf_pbi2
film_thickness_mapi = 0.5  # units of Micrometre
film_thickness_pbi2 = 0.1  # units of Micrometre
tau = None
sigma = None
muNought = 1.25663706e-6
C = 2.99792458e8
n_s = 2 #index of refraction of substrate

#strenght peramiter for MAPI 13.5

# Add thickness_difference parameter to global variables
thickness_difference = 0.0  # units of Micrometre

# Add near the top of the file with other imports
warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered in divide')

# Add near the top with other global variables
show_simulation = True  # Toggle variable for simulation visibility

# Function to evaluate the dielectric function expression
def evaluate_dielectric_function(freq, w_TO, g_TO, w_LO, g_LO, eps_inf):
    eps = eps_inf
    for n in range(len(w_TO)):
        eps *= (w_LO[n]**2 - freq**2 + 1j*freq*g_LO[n]) / (w_TO[n]**2 - freq**2 + 1j*freq*g_TO[n])
    return eps

#function to calculate conductivity
def calculate_conductivity(eps, freq, thickness):
    sigma = 1j * freq * (eps) * (thickness * 1e-6)
    return sigma

#add conductivity of lead iodide and meythal ammonium lead iodide to then calculate the transmission

#function to calculate transmission
def calculate_transmission(sigma):
    ada = (muNought * C) / (n_s +1) # will want to brake this out for speed
    # print(ada)
    tau = 1 / (1 + ( ada * (sigma)))  
    tau = tau**2
    return tau

#calculate the magnitude of the transmission
def calculate_magnitude(tau):
    return np.abs(tau)

#function to calculate the phase of the transmission
def calculate_phase(tau, freq):
    # Add small epsilon to avoid division by zero
    freq = np.where(freq == 0, 1e-10, freq)  # Replace zeros with small number
    # Calculate wavelength in micrometers (c/f)
    wavelength = (C / (freq * 1e10)) * 1e6  # Convert to micrometers
    # Add phase offset due to thickness difference
    phase_offset = 2 * np.pi * thickness_difference / wavelength
    # Calculate total phase (original phase + offset)
    total_phase = np.angle(tau, deg=True) + np.degrees(phase_offset)
    return total_phase


def full_calculation(freq, w_TO, g_TO, w_LO, g_LO, eps_inf, w_TO_pbi2, g_TO_pbi2, w_LO_pbi2, g_LO_pbi2, eps_inf_pbi2):
    # Calculate dielectric function for MAPbI3
    eps1 = evaluate_dielectric_function(freq, w_TO, g_TO, w_LO, g_LO, eps_inf)
    
    # Calculate dielectric function for PbI2
    eps2 = evaluate_dielectric_function(freq, w_TO_pbi2, g_TO_pbi2, w_LO_pbi2, g_LO_pbi2, eps_inf_pbi2)

    # Calculate conductivity for both layers
    sigma1 = calculate_conductivity(eps1, freq, film_thickness_mapi)
    sigma2 = calculate_conductivity(eps2, freq, film_thickness_pbi2)

    # Combine conductivities
    sigma_total = sigma1 + sigma2

    # Calculate transmission using the total conductivity
    tau = calculate_transmission(sigma_total)

    mag = calculate_magnitude(tau)
    phase = calculate_phase(tau, freq)

    return mag, phase

    


# Function to create a slider with an entry box
def create_slider(parent, label_text, from_value, to_value, initial_value):
    slider_frame = ttk.Frame(parent)
    slider_frame.pack(pady=5)
    
    slider_label = ttk.Label(slider_frame, text=label_text)
    slider_label.pack(side=tk.LEFT)
    
    # Update the slider command to trigger on any movement
    slider = ttk.Scale(slider_frame, from_=from_value, to=to_value, value=initial_value, 
                      orient=tk.HORIZONTAL, 
                      command=lambda val: update_entry_and_plot(slider, entry, val))
    slider.pack(side=tk.LEFT, padx=5)
    
    entry = ttk.Entry(slider_frame, width=5)
    entry.insert(0, str(initial_value))
    entry.pack(side=tk.LEFT)
    
    # Add immediate update on entry changes
    entry.bind('<KeyRelease>', lambda event: update_slider_and_plot(slider, entry))
    entry.bind('<Return>', lambda event: update_slider_and_plot(slider, entry))
    entry.bind('<FocusOut>', lambda event: update_slider_and_plot(slider, entry))
    
    return slider, entry

# Function to update the entry box and call the command when the slider value changes
def update_entry_and_plot(slider, entry, value):
    entry.delete(0, tk.END)
    entry.insert(0, f"{float(value):.2f}")
    # Ensure plot updates immediately
    root.after(10, update_plot)  # Small delay to ensure UI remains responsive

# Function to update the slider and call the command when the entry box value changes
def update_slider_and_plot(slider, entry):
    try:
        new_value = float(entry.get())
        slider.set(new_value)
        # Ensure plot updates immediately
        root.after(10, update_plot)  # Small delay to ensure UI remains responsive
    except ValueError:
        pass

#function to do an auto fit of the imported data adjusting MAPI film thicness, PBI2 thicness and thicness difference to minimize chi squared value
#by taking small steps and seeing if the fit gets better or worse

#adaptive box, once i find best point, if in center draw new smaller box, if on edge draw new box with edge on point
#chi squared perecentage improvement 

# Function to do an auto fit of the imported data
def auto_fit():
    global film_thickness_mapi, film_thickness_pbi2, thickness_difference
    best_chi_squared = float('inf')
    best_params = (film_thickness_mapi, film_thickness_pbi2, thickness_difference)

    # Define step sizes for adjustments
    step_size = 0.01
    for delta_mapi in np.arange(-0.1, 0.1, step_size):
        for delta_pbi2 in np.arange(-0.1, 0.1, step_size):
            for delta_thickness in np.arange(-0.1, 0.1, step_size):
                # Adjust parameters
                current_mapi = film_thickness_mapi + delta_mapi
                current_pbi2 = film_thickness_pbi2 + delta_pbi2
                current_thickness_diff = thickness_difference + delta_thickness
                
                # Perform full calculation with adjusted parameters
                mag, phase = full_calculation(imported_data['freq'], w_TO, g_TO, w_LO, g_LO, eps_inf, 
                                              w_TO_pbi2, g_TO_pbi2, w_LO_pbi2, g_LO_pbi2, eps_inf_pbi2)
                
                # Calculate chi-squared
                chi_squared = calculate_chi_squared(mag, imported_data['mag'])
                
                # Update best parameters if current chi-squared is lower
                if chi_squared < best_chi_squared:
                    best_chi_squared = chi_squared
                    best_params = (current_mapi, current_pbi2, current_thickness_diff)

    # Update global variables with best parameters
    film_thickness_mapi, film_thickness_pbi2, thickness_difference = best_params
    
    # Update the plot with the best fit parameters
    update_plot()  

    # Optionally, display the best chi-squared value to the user
    print(f"Best Chi-squared: {best_chi_squared:.4f}")
    print(f"Best Parameters: MAPI Thickness = {film_thickness_mapi:.4f}, PbI2 Thickness = {film_thickness_pbi2:.4f}, Thickness Difference = {thickness_difference:.4f}")



def least_squares_auto_fit():
    """
    Two-step optimization:
    1. Optimize magnitude by adjusting MAPI and PbI2 thicknesses
    2. Optimize phase by adjusting only thickness difference
    """
    global film_thickness_mapi, film_thickness_pbi2, thickness_difference
    
    def optimize_magnitude(params):
        """
        Objective function for magnitude optimization.
        
        Args:
            params (array): [mapi_thickness, pbi2_thickness]
        
        Returns:
            array: Difference between simulated and experimental magnitude
        """
        global film_thickness_mapi, film_thickness_pbi2
        
        # Unpack parameters
        current_mapi, current_pbi2 = params
        
        # Store original values
        original_mapi = film_thickness_mapi
        original_pbi2 = film_thickness_pbi2
        
        # Update thicknesses
        film_thickness_mapi = current_mapi
        film_thickness_pbi2 = current_pbi2
        
        # Calculate magnitude
        mag, _ = full_calculation(
            imported_data['freq'], 
            w_TO, g_TO, w_LO, g_LO, eps_inf,
            w_TO_pbi2, g_TO_pbi2, w_LO_pbi2, g_LO_pbi2, eps_inf_pbi2
        )
        
        # Restore original values
        film_thickness_mapi = original_mapi
        film_thickness_pbi2 = original_pbi2
        
        return mag - imported_data['mag']
    
    def optimize_phase(thickness_diff):
        """
        Objective function for phase optimization.
        Only optimizes thickness_difference parameter.
        
        Args:
            thickness_diff (float): Thickness difference value
        
        Returns:
            array: Difference between simulated and experimental phase
        """
        global thickness_difference
        
        # Store original value
        original_thickness_diff = thickness_difference
        
        # Update thickness difference
        thickness_difference = thickness_diff[0]  # Need [0] as least_squares expects array-like input
        
        # Calculate phase
        _, phase = full_calculation(
            imported_data['freq'], 
            w_TO, g_TO, w_LO, g_LO, eps_inf,
            w_TO_pbi2, g_TO_pbi2, w_LO_pbi2, g_LO_pbi2, eps_inf_pbi2
        )
        
        # Restore original value
        thickness_difference = original_thickness_diff
        
        return phase - imported_data['phase']
    
    try:
        # Step 1: Optimize magnitude
        initial_params_mag = [film_thickness_mapi, film_thickness_pbi2]
        bounds_mag = ([0, 0], [10, 10])
        
        result_mag = least_squares(
            optimize_magnitude, 
            initial_params_mag,
            bounds=bounds_mag,
            method='trf'
        )
        
        # Update thickness parameters with optimized values
        film_thickness_mapi, film_thickness_pbi2 = result_mag.x
        
        # Step 2: Optimize phase (thickness difference only)
        initial_params_phase = [thickness_difference]
        bounds_phase = ([-100], [100])
        
        result_phase = least_squares(
            optimize_phase,
            initial_params_phase,
            bounds=bounds_phase,
            method='trf'
        )
        
        # Update thickness difference with optimized value
        thickness_difference = result_phase.x[0]
        
        # Print optimization results with filename
        if 'filename' in imported_data and imported_data['filename']:
            print(f"\nOptimization Results for {imported_data['filename']}:")
        else:
            print("\nOptimization Results (no file imported):")
        print(f"MAPI Thickness: {1000 * film_thickness_mapi:.4f}")
        print(f"PbI2 Thickness: {1000 * film_thickness_pbi2:.4f}")
        print(f"Thickness Difference: {thickness_difference:.4f}")
        # print(f"Magnitude Optimization Cost: {result_mag.cost:.4f}")
        # print(f"Phase Optimization Cost: {result_phase.cost:.4f}")
        print(f"total thickness: {(film_thickness_pbi2 + film_thickness_mapi):.4f} ")
        print(f"{1000 * film_thickness_mapi:.4f}" + "," + f"{1000 * film_thickness_pbi2:.4f}")

        # Update the sliders with the new values
        slider_film_thickness_mapi.set(film_thickness_mapi)
        slider_film_thickness_pbi2.set(film_thickness_pbi2)
        slider_thickness_difference.set(thickness_difference)
        
        # Update the plot
        update_plot()
    
    except Exception as e:
        print(f"Optimization failed: {e}")


# Modify the update_plot function
def update_plot():
    global x_axis_range, film_thickness_mapi, film_thickness_pbi2, thickness_difference
    w_TO = [slider_w_TO_1.get(), slider_w_TO_2.get()]
    g_TO = [slider_g_TO_1.get(), slider_g_TO_2.get()]
    w_LO = [slider_w_LO_1.get(), slider_w_LO_2.get()]
    g_LO = [slider_g_LO_1.get(), slider_g_LO_2.get()]
    eps_inf = slider_eps_inf.get()
    eps_static = slider_eps_static.get()
    x_axis_range = slider_x_axis.get()
    film_thickness_mapi = slider_film_thickness_mapi.get()
    film_thickness_pbi2 = slider_film_thickness_pbi2.get()
    w_TO_pbi2 = [slider_w_TO_pbi2.get()]
    g_TO_pbi2 = [slider_g_TO_pbi2.get()]
    w_LO_pbi2 = [slider_w_LO_pbi2.get()]
    g_LO_pbi2 = [slider_g_LO_pbi2.get()]
    eps_inf_pbi2 = slider_eps_inf_pbi2.get()
    thickness_difference = slider_thickness_difference.get()
    plot_dielectric_function(w_TO, g_TO, w_LO, g_LO, eps_inf, eps_static, w_TO_pbi2, g_TO_pbi2, w_LO_pbi2, g_LO_pbi2, eps_inf_pbi2)

# Create the main window
root = tk.Tk()
root.title("Dielectric Function Simulation for MAPbI3")
root.geometry("1200x800")

# Create two frames, one for sliders and one for the plots
slider_frame = ttk.Frame(root)
slider_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

plot_frame = ttk.Frame(root)
plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

# Add a frame for buttons
button_frame = ttk.Frame(slider_frame)
button_frame.pack(pady=5)

# Add import button and data storage
imported_data = {'freq': None, 'mag': None, 'phase': None}

def import_data():
    file_path = filedialog.askopenfilename(filetypes=[("3ftr files", "*.3ftr"),("All files", "*.*")])
    if file_path:
        try:
            # Read tab-delimited data using pandas
            data = pd.read_csv(file_path, delimiter='\t', skiprows=0)
            # Extract frequency and magnitude data from columns
            imported_data['freq'] = data.iloc[:, 0].values * 33.356  # Convert THz to cm^-1
            imported_data['mag'] = data.iloc[:, 1].values
            imported_data['phase'] = data.iloc[:, 2].values * 57.3  # Get phase data from third column convert to degrees
            imported_data['filename'] = file_path.split('/')[-1]  # Store just the filename
            print(f"Imported file: {imported_data['filename']}")  # Add this line for debugging
            update_plot()
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to load data: {str(e)}")

import_button = ttk.Button(button_frame, text="Import Data", command=import_data)
import_button.pack(side=tk.LEFT, padx=5)

# Add after the import_button creation
def fit_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        # Get all .3ftr files in the folder
        files = [f for f in os.listdir(folder_path) if f.endswith('.3ftr')]
        
        print(f"\nProcessing {len(files)} files from folder: {folder_path}")
        print("=" * 50)
        
        # Process each file
        for file in files:
            full_path = os.path.join(folder_path, file)
            try:
                # Read the file
                data = pd.read_csv(full_path, delimiter='\t', skiprows=0)
                
                # Store the data
                imported_data['freq'] = data.iloc[:, 0].values * 33.356
                imported_data['mag'] = data.iloc[:, 1].values
                imported_data['phase'] = data.iloc[:, 2].values * 57.3
                imported_data['filename'] = file
                
                # Perform the optimization
                least_squares_auto_fit()
                
            except Exception as e:
                print(f"\nOptimization failed for file {file}")
                print(f"Error: {str(e)}")
                print("-" * 50)
        
        print("\nFolder processing complete")
        print("=" * 50)

# Add the Fit Folder button
fit_folder_button = ttk.Button(button_frame, text="Fit Folder", command=fit_folder)
fit_folder_button.pack(side=tk.LEFT, padx=5)

# Create sliders with entry boxes using the create_slider function
slider_w_TO_1, entry_w_TO_1 = create_slider(slider_frame, "w_TO_1", 0, 100, w_TO[0])
slider_w_TO_2, entry_w_TO_2 = create_slider(slider_frame, "w_TO_2", 0, 100, w_TO[1])
slider_g_TO_1, entry_g_TO_1 = create_slider(slider_frame, "g_TO_1", 0, 100, g_TO[0])
slider_g_TO_2, entry_g_TO_2 = create_slider(slider_frame, "g_TO_2", 0, 100, g_TO[1])
slider_w_LO_1, entry_w_LO_1 = create_slider(slider_frame, "w_LO_1", 0, 300, w_LO[0])
slider_w_LO_2, entry_w_LO_2 = create_slider(slider_frame, "w_LO_2", 0, 300, w_LO[1])
slider_g_LO_1, entry_g_LO_1 = create_slider(slider_frame, "g_LO_1", 0, 100, g_LO[0])
slider_g_LO_2, entry_g_LO_2 = create_slider(slider_frame, "g_LO_2", 0, 100, g_LO[1])
slider_eps_inf, entry_eps_inf = create_slider(slider_frame, "eps_inf", 0, 10, eps_inf)
slider_eps_static, entry_eps_static = create_slider(slider_frame, "eps_static", 0, 100, eps_static)

# Add sliders for PbI2 parameters
slider_w_TO_pbi2, entry_w_TO_pbi2 = create_slider(slider_frame, "w_TO_PbI2", 0, 100, w_TO_pbi2[0])
slider_g_TO_pbi2, entry_g_TO_pbi2 = create_slider(slider_frame, "g_TO_PbI2", 0, 100, g_TO_pbi2[0])
slider_w_LO_pbi2, entry_w_LO_pbi2 = create_slider(slider_frame, "w_LO_PbI2", 0, 300, w_LO_pbi2[0])
slider_g_LO_pbi2, entry_g_LO_pbi2 = create_slider(slider_frame, "g_LO_PbI2", 0, 100, g_LO_pbi2[0])
slider_eps_inf_pbi2, entry_eps_inf_pbi2 = create_slider(slider_frame, "eps_inf_PbI2", 0, 10, eps_inf_pbi2)

# Add to the slider creation section
slider_x_axis, _ = create_slider(slider_frame, "X-axis range (THz)", 0, 10, 3)
slider_film_thickness_mapi, _ = create_slider(slider_frame, "MAPIFilm Thickness (Micrometre)", 0, 1, film_thickness_mapi)
slider_film_thickness_pbi2, _ = create_slider(slider_frame, "PbI2 Thickness (Micrometre)", 0, 1, film_thickness_pbi2)
slider_thickness_difference, _ = create_slider(slider_frame, "Thickness Difference (Micrometre)", -10, 10, thickness_difference)

# Create a figure and canvas for plotting
fig = Figure(figsize=(8, 8), dpi=100)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Function to convert cm^-1 to THz
def cm1_to_thz(freq_cm1):
    return freq_cm1 * 0.03

# Function to calculate chi-squared
def calculate_chi_squared(simulated, experimental):
    return np.sum((simulated - experimental) ** 2)

# Modify the plot_dielectric_function function
def plot_dielectric_function(w_TO, g_TO, w_LO, g_LO, eps_inf, eps_static, w_TO_pbi2, g_TO_pbi2, w_LO_pbi2, g_LO_pbi2, eps_inf_pbi2):
    freq_cm1 = np.linspace(0, 300, 1000)
    freq_thz = cm1_to_thz(freq_cm1)

    # Calculate combined transmission
    tau_combined, phase = full_calculation(freq_cm1, w_TO, g_TO, w_LO, g_LO, eps_inf, 
                                         w_TO_pbi2, g_TO_pbi2, w_LO_pbi2, g_LO_pbi2, eps_inf_pbi2)

    # Calculate MAPI only transmission
    eps_mapi = evaluate_dielectric_function(freq_cm1, w_TO, g_TO, w_LO, g_LO, eps_inf)
    sigma_mapi = calculate_conductivity(eps_mapi, freq_cm1, film_thickness_mapi)
    tau_mapi = calculate_transmission(sigma_mapi)

    # Calculate PbI2 only transmission
    eps_pbi2 = evaluate_dielectric_function(freq_cm1, w_TO_pbi2, g_TO_pbi2, w_LO_pbi2, g_LO_pbi2, eps_inf_pbi2)
    sigma_pbi2 = calculate_conductivity(eps_pbi2, freq_cm1, film_thickness_pbi2)
    tau_pbi2 = calculate_transmission(sigma_pbi2)

    ax1.clear()
    ax2.clear()

    # Plot simulated data only if show_simulation is True
    if show_simulation:
        ax1.plot(freq_thz, np.abs(tau_combined), label='Combined', linewidth=2)
        ax1.plot(freq_thz, np.abs(tau_mapi), '--', label='MAPI', alpha=0.7)
        ax1.plot(freq_thz, np.abs(tau_pbi2), '--', label='PbI2', alpha=0.7)
        ax2.plot(freq_thz, phase, label='Simulated')
    
    # Plot imported data if available
    if imported_data['freq'] is not None:
        imported_freq_thz = imported_data['freq'] * 0.03  # Convert cm^-1 to THz
        ax1.scatter(imported_freq_thz, imported_data['mag'], color='red', s=10, label='Experimental')
        ax2.scatter(imported_freq_thz, imported_data['phase'], color='red', s=10, label='Experimental')

        # Interpolate the experimental data to match the simulated frequency points
        interp_func_mag = interp1d(imported_freq_thz, imported_data['mag'], bounds_error=False, fill_value="extrapolate")
        interp_func_phase = interp1d(imported_freq_thz, imported_data['phase'], bounds_error=False, fill_value="extrapolate")

        # Get interpolated values
        interpolated_mag = interp_func_mag(freq_thz)
        interpolated_phase = interp_func_phase(freq_thz)
        
        # Calculate chi-squared
        if imported_data['freq'] is not None and len(imported_data['freq']) > 0:
            # Determine the range of the imported data
            min_freq = np.min(imported_data['freq']) * 0.03
            max_freq = np.max(imported_data['freq']) * 0.03
            
            # Restrict the simulated data to the range of the imported data
            mask = (freq_thz >= min_freq) & (freq_thz <= max_freq)
            
            # Calculate chi-squared only for the valid range
            chi_squared = calculate_chi_squared(np.abs(tau_combined[mask]), interpolated_mag[mask])
            chi_squared_label = f"Chi-squared: {chi_squared:.4f}"
            ax1.text(0.95, 0.95, chi_squared_label, transform=ax1.transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right')

    ax1.set_xlabel('Frequency (THz)')
    ax1.set_ylabel('Transmission')
    ax1.set_title('Transmission of MAPbI3 and PbI2')
    ax1.legend()
    ax1.set_xlim(0, x_axis_range)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2.set_xlabel('Frequency (THz)')
    ax2.set_ylabel('Phase (degrees)')
    ax2.set_title('Phase of Combined System')
    ax2.legend()
    ax2.set_xlim(0, x_axis_range)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    canvas.draw()

# Add a button to trigger the auto fit
auto_fit_button = ttk.Button(button_frame, text="Auto Fit", command=least_squares_auto_fit)
auto_fit_button.pack(side=tk.LEFT, padx=5)

# Add after the other buttons
def toggle_simulation():
    global show_simulation
    show_simulation = not show_simulation
    update_plot()

# Add toggle button
# toggle_sim_button = ttk.Button(button_frame, text="Toggle Simulation", command=toggle_simulation)
# toggle_sim_button.pack(side=tk.LEFT, padx=5)

# Plot the initial dielectric function
update_plot()

# Start the GUI event loop
root.mainloop()

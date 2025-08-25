# Dielectric Function Calculator for MAPbI3 and PbI2

A Python-based GUI application for calculating and analyzing the dielectric function and transmission properties of methylammonium lead iodide (MAPbI3) perovskite thin films with lead iodide (PbI2) layers. This tool is designed for analyzing terahertz transmission spectroscopy data.

## Features

- **Interactive GUI**: Real-time parameter adjustment with sliders and input fields
- **Dual Material Modeling**: Simultaneous modeling of MAPbI3 and PbI2 layers
- **Data Import**: Support for `.3ftr` tab-delimited experimental data files
- **Automated Fitting**: Least-squares optimization for parameter extraction
- **Batch Processing**: Fit multiple files in a folder automatically
- **Real-time Visualization**: Interactive plots of transmission magnitude and phase
- **Chi-squared Analysis**: Quantitative comparison between simulation and experiment

## Physical Model

The application implements a dielectric function model based on:

### Lorentz Oscillator Model

The dielectric function is calculated using multiple Lorentz oscillators:

```
ε(ω) = ε_∞ ∏ᵢ (ω_LO,i² - ω² + iωγ_LO,i) / (ω_TO,i² - ω² + iωγ_TO,i)
```

Where:

- `ε_∞`: High-frequency dielectric constant
- `ω_TO,i`, `ω_LO,i`: Transverse and longitudinal optical phonon frequencies
- `γ_TO,i`, `γ_LO,i`: Damping constants for TO and LO modes

### Transmission Calculation

The transmission through thin films is calculated considering:

- Conductivity: `σ = iωε(ω)t` (where t is film thickness)
- Transmission: `τ = 1/(1 + μ₀c/(n_s+1) × σ)`²
- Phase offset due to thickness differences

## Installation

### Prerequisites

- Python 3.7+
- Required packages:
  ```bash
  pip install numpy matplotlib tkinter pandas scipy
  ```

### Running the Application

```bash
python DielectricFunctionCalc.py
```

## Usage

### Basic Operation

1. **Launch the application** - The GUI will open with default parameters for MAPbI3 and PbI2
2. **Adjust parameters** - Use sliders or input fields to modify material properties
3. **Import experimental data** - Click "Import Data" to load `.3ftr` files
4. **Optimize parameters** - Use "Auto Fit" for automated parameter extraction
5. **Batch processing** - Use "Fit Folder" to process multiple files

### Parameters

#### MAPbI3 Parameters

- `w_TO_1`, `w_TO_2`: TO phonon frequencies (cm⁻¹)
- `g_TO_1`, `g_TO_2`: TO damping constants (cm⁻¹)
- `w_LO_1`, `w_LO_2`: LO phonon frequencies (cm⁻¹)
- `g_LO_1`, `g_LO_2`: LO damping constants (cm⁻¹)
- `eps_inf`: High-frequency dielectric constant
- `Film Thickness`: MAPbI3 layer thickness (μm)

#### PbI2 Parameters

- `w_TO_PbI2`: TO phonon frequency (cm⁻¹)
- `g_TO_PbI2`: TO damping constant (cm⁻¹)
- `w_LO_PbI2`: LO phonon frequency (cm⁻¹)
- `g_LO_PbI2`: LO damping constant (cm⁻¹)
- `eps_inf_PbI2`: High-frequency dielectric constant
- `PbI2 Thickness`: PbI2 layer thickness (μm)

#### Additional Parameters

- `Thickness Difference`: Phase offset parameter (μm)
- `X-axis range`: Plot frequency range (THz)

### Data Format

The application expects `.3ftr` files with tab-delimited columns:

1. Frequency (THz)
2. Transmission magnitude
3. Phase (radians)

Data is automatically converted to appropriate units (cm⁻¹ for frequency, degrees for phase).

### Optimization Algorithm

The automated fitting uses a two-step least-squares optimization:

1. **Magnitude optimization**: Adjusts MAPbI3 and PbI2 thicknesses
2. **Phase optimization**: Adjusts thickness difference parameter

## Output

### Console Output

During optimization, the program outputs:

```
Optimization Results for [filename]:
MAPI Thickness: [value] (nm)
PbI2 Thickness: [value] (nm)
Thickness Difference: [value] (μm)
Total thickness: [value] (μm)
```

### Plots

- **Top plot**: Transmission magnitude vs frequency
- **Bottom plot**: Phase vs frequency
- Real-time chi-squared value display


## Technical Details

- **GUI Framework**: Tkinter
- **Plotting**: Matplotlib with TkAgg backend
- **Optimization**: SciPy least_squares with trust region reflective algorithm
- **Data Processing**: NumPy and Pandas
- **Interpolation**: SciPy interp1d for experimental data matching

## Contributing

This tool is designed for terahertz spectroscopy analysis of perovskite materials. Contributions for additional materials or analysis features are welcome.

## License

See LICENSE file for details.

## Citation

If you use this tool in your research, please cite the relevant papers on MAPbI3 dielectric properties and terahertz spectroscopy.

Sendner, M., Nayak, P. K., Egger, D. A., Beck, S., Müller, C., Epding, B., Kowalsky, W.,
Kronik, L., Snaith, H. J., Pucci, A., & Lovrinčić, R. (2016). Optical phonons in
methylammonium lead halide perovskites and implications for charge transport. Materials
Horizons, 3(6), 613–620. https://doi.org/10.1039/c6mh00275g

## Support

For questions or issues, please open an issue on the GitHub repository.

import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import io
import PyPDF2
import tabula
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
g = 32.2  # Acceleration due to gravity (ft/s^2)
E = 30e6  # Young's modulus for steel (psi)
rho_steel = 490  # Density of steel (lb/ft^3)

# API 11B units
api_11b_units = {
    "C-57-109-54": {"torque": 57000, "stroke": 54},
    "C-80-119-64": {"torque": 80000, "stroke": 64},
    "C-114-125-74": {"torque": 114000, "stroke": 74},
    "C-160-173-100": {"torque": 160000, "stroke": 100},
    "C-228-213-120": {"torque": 228000, "stroke": 120},
    "C-320-256-144": {"torque": 320000, "stroke": 144},
    "C-456-305-168": {"torque": 456000, "stroke": 168},
    "C-640-365-192": {"torque": 640000, "stroke": 192},
}

@st.cache_data
def calculate_fluid_properties(water_cut, oil_gravity, condensate_gravity=None, is_gas_well=False):
    water_density = 62.4  # lb/ft^3
    if is_gas_well:
        condensate_density = 141.5 / (131.5 + condensate_gravity) * 62.4  # lb/ft^3
        fluid_density = (water_cut/100 * water_density + (1 - water_cut/100) * condensate_density)
    else:
        oil_density = 141.5 / (131.5 + oil_gravity) * 62.4  # lb/ft^3
        fluid_density = (water_cut/100 * water_density + (1 - water_cut/100) * oil_density)
    return fluid_density

@st.cache_data
def calculate_rod_properties(rod_grade):
    yield_strengths = {"D": 90000, "K": 110000, "C": 130000, "UHS": 150000}
    return yield_strengths[rod_grade]

@st.cache_data
def optimize_rod_string(pump_depth, fluid_load, rod_grade, pump_diameter):
    yield_strength = calculate_rod_properties(rod_grade)
    safety_factor = 1.5
    allowable_stress = yield_strength / safety_factor
    
    rod_percentages = {
        1.06: [0, 28.5, 71.5, 0],
        1.25: [0, 30.6, 69.4, 0],
        1.5: [0, 33.8, 66.2, 0],
        1.75: [0, 37.5, 62.5, 0],
        2.0: [0, 41.7, 58.3, 0],
        2.25: [0, 46.5, 53.5, 0],
        2.5: [0, 50.8, 49.2, 0]
    }
    
    closest_pump_size = min(rod_percentages.keys(), key=lambda x: abs(x - pump_diameter))
    percentages = rod_percentages[closest_pump_size]
    
    rod_sizes = [1.0, 0.875, 0.75, 0.625]
    rod_areas = [np.pi * (d/2)**2 for d in rod_sizes]
    
    rod_string = []
    remaining_depth = pump_depth
    current_load = fluid_load
    
    for size, area, percentage in zip(rod_sizes, rod_areas, percentages):
        if percentage > 0:
            length = pump_depth * percentage / 100
            rod_string.append((size, length))
            remaining_depth -= length
            current_load += rho_steel * g * area * length
    
    max_stress = current_load / min(rod_areas)
    if max_stress > allowable_stress:
        st.warning(f"Rod string design exceeds allowable stress. Consider using a higher grade rod or reducing pump depth.")
    
    return rod_string[::-1]

@st.cache_data
def calculate_rod_stretch(load, rod_string, E):
    return sum(load * l / (E * np.pi * (d/2)**2) for d, l in rod_string)

@st.cache_data
def calculate_damping_factor(fluid_density, tubing_id, rod_diameter):
    annular_area = np.pi * ((tubing_id/2)**2 - (rod_diameter/2)**2)
    return 0.1 * fluid_density * annular_area

@st.cache_data
def gibbs_wave_equation_3d(y, t, L, rod_string, E, rho, c, stroke, speed, fluid_density, g, pump_area, damping_factor, deviation_angle):
    n = len(y) // 3
    u, v, w = y[:n], y[n:2*n], y[2*n:]
    
    omega = speed * 2 * np.pi / 60
    u_surface = 0.5 * stroke * (1 - np.cos(omega * t))
    v_surface = 0.5 * stroke * omega * np.sin(omega * t)
    
    fluid_load = fluid_density * g * L * pump_area
    
    du_dt = v
    dv_dt = np.zeros_like(v)
    dw_dt = np.zeros_like(w)
    
    current_depth = 0
    for i, (d, l) in enumerate(rod_string):
        A = np.pi * (d/2)**2
        section_start = int(current_depth / L * n)
        section_end = int((current_depth + l) / L * n)
        
        dv_dt[section_start:section_end] = (c**2 * np.gradient(np.gradient(u[section_start:section_end], l/n), l/n) - 
                                            g * np.cos(np.radians(deviation_angle)) - 
                                            damping_factor * v[section_start:section_end] / (rho * A))
        
        dw_dt[section_start:section_end] = (c**2 * np.gradient(np.gradient(w[section_start:section_end], l/n), l/n) - 
                                            g * np.sin(np.radians(deviation_angle)) - 
                                            damping_factor * w[section_start:section_end] / (rho * A))
        
        current_depth += l
    
    # Improved boundary conditions
    du_dt[0] = v_surface
    dv_dt[0] = c**2 * (u[1] - u_surface) / (L/n)**2 - g * np.cos(np.radians(deviation_angle))
    dw_dt[0] = 0
    du_dt[-1] = v[-1]
    dv_dt[-1] = c**2 * (u[-2] - u[-1]) / (L/n)**2 - fluid_load / (rho * A) - g * np.cos(np.radians(deviation_angle))
    dw_dt[-1] = 0
    
    return np.concatenate([du_dt, dv_dt, dw_dt])

@st.cache_data
def simulate_rod_pump(L, rod_string, E, rho, stroke, speed, fluid_density, g, pump_area, t_span, n_points, damping_factor, deviation_angle):
    c = np.sqrt(E / rho)
    y0 = np.zeros(3 * n_points)
    t = np.linspace(t_span[0], t_span[1], 1000)
    
    sol = odeint(gibbs_wave_equation_3d, y0, t, args=(L, rod_string, E, rho, c, stroke, speed, fluid_density, g, pump_area, damping_factor, deviation_angle))
    
    return t, sol

@st.cache_data
def calculate_polished_rod_load(surface_load, rod_weight, acceleration, fluid_load):
    return surface_load + rod_weight + acceleration * rod_weight / g - fluid_load

@st.cache_data
def calculate_gearbox_torque(polished_rod_load, crank_angle, crank_radius, pitman_length):
    torque_factor = (crank_radius * np.sin(crank_angle) + 
                     (crank_radius**2 * np.sin(crank_angle) * np.cos(crank_angle)) / 
                     np.sqrt(pitman_length**2 - crank_radius**2 * np.sin(crank_angle)**2))
    return polished_rod_load * torque_factor

@st.cache_data
def calculate_neutral_point(rod_string, fluid_density, pump_area, pump_depth):
    total_weight = 0
    buoyant_weight = 0
    for d, l in rod_string:
        area = np.pi * (d/2)**2
        weight = rho_steel * g * area * l
        buoyancy = fluid_density * g * area * l
        total_weight += weight
        buoyant_weight += buoyancy
    
    fluid_load = fluid_density * g * pump_depth * pump_area
    neutral_point = pump_depth * (1 - fluid_load / (total_weight - buoyant_weight))
    return neutral_point

@st.cache_data
def calculate_buckling_load(rod_string, tubing_id):
    critical_loads = []
    for d, l in rod_string:
        I = np.pi * d**4 / 64  # Moment of inertia
        r = (tubing_id - d) / 2  # Radial clearance
        Pcr = np.pi**2 * E * I / (4 * l**2) * (1 + r/l)  # Modified Euler buckling formula
        critical_loads.append(Pcr)
    return min(critical_loads)

@st.cache_data
def recommend_pump_depth(perforation_depth, fluid_level, safety_margin=100):
    return min(perforation_depth - safety_margin, fluid_level - safety_margin)

@st.cache_data
def recommend_sump_depth(pump_depth, tubing_id, liquid_rate, safety_factor=1.5):
    sump_volume = liquid_rate * 5.615 / 1440  # Convert bbl/day to ft^3/min
    sump_height = sump_volume / (np.pi * (tubing_id/24)**2)  # Convert tubing_id to ft
    return pump_depth + sump_height * safety_factor

@st.cache_data
def calculate_dog_leg_severity(md1, inc1, azi1, md2, inc2, azi2):
    """Calculate dog leg severity between two survey points."""
    dls = np.arccos(np.cos(np.radians(inc2 - inc1)) - 
                    np.sin(np.radians(inc1)) * np.sin(np.radians(inc2)) * 
                    (1 - np.cos(np.radians(azi2 - azi1)))) * (180 / np.pi) * 100 / (md2 - md1)
    return dls

@st.cache_data
def process_deviation_data(deviation_df):
    """Process deviation data and calculate additional parameters."""
    deviation_df['Inc_rad'] = np.radians(deviation_df['Angle'])
    deviation_df['Azi_rad'] = np.radians(deviation_df['Azimuth'])
    
    deviation_df['DLS'] = 0.0
    for i in range(1, len(deviation_df)):
        deviation_df.loc[i, 'DLS'] = calculate_dog_leg_severity(
            deviation_df.loc[i-1, 'SD'], deviation_df.loc[i-1, 'Angle'], deviation_df.loc[i-1, 'Azimuth'],
            deviation_df.loc[i, 'SD'], deviation_df.loc[i, 'Angle'], deviation_df.loc[i, 'Azimuth']
        )
    
    return deviation_df

@st.cache_data
def extract_deviation_data_from_pdf(uploaded_file):
    """Extract deviation data from the uploaded PDF file."""
    try:
        # First, try using tabula-py to extract tables
        tables = tabula.read_pdf(uploaded_file, pages='all', multiple_tables=True)
        if tables:
            df = pd.concat(tables, ignore_index=True)
        else:
            raise ValueError("No tables found in PDF")
    except Exception as e:
        st.warning(f" Falling back to PyPDF2 for text extraction.")
        
        # Fallback to PyPDF2 for text extraction
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        # Attempt to parse the extracted text into a DataFrame
        lines = text.split('\n')
        data = [line.split() for line in lines if line.strip() and line.split()[0].isdigit()]
        df = pd.DataFrame(data)

    # Ensure the DataFrame has the correct column names
    if len(df.columns) >= 10:
        df.columns = ['Sl No', 'SD', 'Angle', 'Azimuth', 'TVD', 'N-S', 'E-W', 'Net Drift', 'Net Dir', 'VS'] + list(df.columns[10:])
    else:
        st.error("The extracted data does not have the expected number of columns. Please check the PDF format.")
        return None

    # Convert columns to appropriate data types
    numeric_columns = ['SD', 'Angle', 'Azimuth', 'TVD', 'N-S', 'E-W', 'Net Drift', 'Net Dir', 'VS']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

@st.cache_data
def plot_wellpath(deviation_df):
    """Plot 3D well path."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(deviation_df['E-W'], deviation_df['N-S'], deviation_df['TVD'])
    ax.set_xlabel('East-West (m)')
    ax.set_ylabel('North-South (m)')
    ax.set_zlabel('TVD (m)')
    ax.invert_zaxis()
    plt.title('3D Well Path')
    return fig

def main():
    st.set_page_config(page_title="Advanced Gas Well Deliquification Rod Pump Designer", layout="wide")
    st.title("Advanced Gas Well Deliquification Rod Pump Designer")

    # Sidebar for input parameters
    with st.sidebar:
        st.header("Input Parameters")
        
        well_type = st.radio("Well Type", ["Gas Well", "Oil Well"])
        is_gas_well = well_type == "Gas Well"

        col1, col2 = st.columns(2)
        with col1:
            tubing_id = st.number_input("Tubing ID (inches)", min_value=1.0, max_value=4.0, value=2.375, step=0.125)
            gas_gravity = st.number_input("Gas Specific Gravity (air=1)", min_value=0.5, max_value=1.5, value=0.65, step=0.01)
            perforation_depth = st.number_input("Perforation Depth (ft)", min_value=1000.0, max_value=20000.0, value=8000.0, step=100.0)
        with col2:
            casing_id = st.number_input("Casing ID (inches)", min_value=4.0, max_value=9.0, value=5.5, step=0.125)
            gas_rate = st.number_input("Gas Production Rate (MSCF/day)", min_value=100.0, max_value=10000.0, value=500.0, step=50.0)
            fluid_level = st.number_input("Fluid Level (ft)", min_value=0.0, max_value=perforation_depth, value=3500.0, step=100.0)

        water_cut = st.slider("Water Cut (%)", min_value=0, max_value=100, value=50)
        
        if is_gas_well:
            condensate_gravity = st.number_input("Condensate Gravity (API)", min_value=10.0, max_value=100.0, value=60.0, step=1.0)
            liquid_rate = st.number_input("Condensate + Water Production Rate (bbl/day)", min_value=1.0, max_value=1000.0, value=50.0, step=1.0)
        else:
            oil_gravity = st.number_input("Oil Gravity (API)", min_value=10.0, max_value=70.0, value=40.0, step=1.0)
            liquid_rate = st.number_input("Liquid Production Rate (bbl/day)", min_value=1.0, max_value=1000.0, value=100.0, step=1.0)

        recommended_pump_depth = recommend_pump_depth(perforation_depth, fluid_level)
        pump_depth = st.number_input("Pump Depth (ft)", min_value=1000.0, max_value=10000.0, value=recommended_pump_depth, step=100.0)
        
        recommended_sump_depth = recommend_sump_depth(pump_depth, tubing_id, liquid_rate)
        sump_depth = st.number_input("Sump Depth (ft)", min_value=pump_depth, max_value=perforation_depth, value=recommended_sump_depth, step=100.0)

        pump_diameter = st.select_slider("Pump Diameter (inches)", options=[1.06, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5], value=1.5)
        surface_stroke_length = st.slider("Surface Stroke Length (inches)", min_value=20, max_value=120, value=36, step=1)
        pump_speed = st.slider("Pump Speed (strokes/min)", min_value=1, max_value=20, value=8, step=1)

        rod_grades = ["D", "K", "C", "UHS"]
        rod_grade = st.selectbox("Rod Grade", rod_grades, index=1)

        api_unit = st.selectbox("API 11B Unit", list(api_11b_units.keys()))
        crank_radius = st.number_input("Crank Radius (inches)", min_value=10.0, max_value=100.0, value=24.0, step=1.0)
        pitman_length = st.number_input("Pitman Length (inches)", min_value=50.0, max_value=300.0, value=120.0, step=1.0)

        is_deviated = st.checkbox("Is the well deviated?")
        
        uploaded_file = None
        deviation_angle = 0
        
        if is_deviated:
            st.subheader("Deviated Well Data")
            uploaded_file = st.file_uploader("Upload deviation data PDF", type="pdf")

        pump_fill = st.slider("Pump Fill (%)", min_value=0, max_value=100, value=75)

    # Main content area
    if st.button("Run Analysis", key="run_analysis"):
        # Perform calculations
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Calculating fluid properties...")
        progress_bar.progress(10)
        if is_gas_well:
            fluid_density = calculate_fluid_properties(water_cut, None, condensate_gravity, is_gas_well=True)
        else:
            fluid_density = calculate_fluid_properties(water_cut, oil_gravity)
        
        status_text.text("Calculating rod properties...")
        progress_bar.progress(20)
        yield_strength = calculate_rod_properties(rod_grade)
        
        status_text.text("Optimizing rod string...")
        progress_bar.progress(30)
        fluid_load = fluid_density * g * pump_depth * (np.pi * (pump_diameter/2/12)**2)
        rod_string = optimize_rod_string(pump_depth, fluid_load, rod_grade, pump_diameter)
        if rod_string is None:
            st.error("Failed to optimize rod string. Please check your input parameters.")
            return
        
        rod_weight = sum(rho_steel * np.pi * (d/2)**2 * l for d, l in rod_string)

        status_text.text("Simulating rod pump...")
        progress_bar.progress(50)
        L = pump_depth
        A = np.pi * (rod_string[0][0] / 2)**2
        rho = rho_steel
        stroke = surface_stroke_length / 12
        speed = pump_speed
        pump_area = np.pi * (pump_diameter / 2 / 12)**2
        t_span = (0, 60 / speed)
        n_points = 100
        damping_factor = calculate_damping_factor(fluid_density, tubing_id, rod_string[0][0])
        t, sol = simulate_rod_pump(L, rod_string, E, rho, stroke, speed, fluid_density, g, pump_area, t_span, n_points, damping_factor, deviation_angle)

        status_text.text("Calculating loads and stresses...")
        progress_bar.progress(70)
        surface_pos = sol[:, 0]
        surface_load = E * A * np.gradient(sol[:, 0], L)
        acceleration = np.gradient(np.gradient(surface_pos, t), t)
        polished_rod_load = calculate_polished_rod_load(surface_load, rod_weight, acceleration, fluid_load)
        fluid_load = fluid_density * g * L * pump_area
        downhole_pos = surface_pos + calculate_rod_stretch(polished_rod_load, rod_string, E)
        downhole_load = polished_rod_load - fluid_load - rod_weight

        # Adjust dynamometer cards based on pump fill
        downhole_load *= pump_fill / 100
        surface_load = downhole_load + rod_weight

        # Calculate API Max Fluid Load
        api_max_fluid_load = fluid_density * g * L * pump_area

        status_text.text("Generating results...")
        progress_bar.progress(90)

        # Results display
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dynamometer Cards")
            chart_data = pd.DataFrame({
                'Surface Position': surface_pos * 12,
                'Surface Load': surface_load,
                'Downhole Position': downhole_pos * 12,
                'Downhole Load': downhole_load
            })
            st.line_chart(chart_data)

        with col2:
            st.subheader("Key Results")
            max_stress = np.max(surface_load) / A
            buckling_load = calculate_buckling_load(rod_string, tubing_id)
            key_results = {
                "Maximum Polished Rod Load (lbs)": np.max(polished_rod_load),
                "Minimum Polished Rod Load (lbs)": np.min(polished_rod_load),
                "Peak-to-Peak Load (lbs)": np.max(polished_rod_load) - np.min(polished_rod_load),
                "Maximum Gearbox Torque (in-lbs)": np.max(calculate_gearbox_torque(polished_rod_load, np.linspace(0, 2*np.pi, len(t)), crank_radius/12, pitman_length/12)) * 12,
                "Maximum Stress (psi)": max_stress,
                "Yield Strength (psi)": yield_strength,
                "Safety Factor": yield_strength / max_stress,
                "API Max Fluid Load (lbs)": api_max_fluid_load,
                "Buckling Load (lbs)": buckling_load,
            }
            for key, value in key_results.items():
                st.metric(key, f"{value:.2f}")

        # Expandable sections for detailed analysis
        with st.expander("Rod String Design"):
            for i, (diameter, length) in enumerate(rod_string):
                st.write(f"Section {i+1}: {diameter:.3f} inch diameter, {length:.2f} ft length")
            neutral_point = calculate_neutral_point(rod_string, fluid_density, pump_area, pump_depth)
            st.write(f"Neutral Point: {neutral_point:.2f} ft")

        with st.expander("Pump Depth and Sump Analysis"):
            st.write(f"Recommended Pump Depth: {recommended_pump_depth:.2f} ft")
            st.write(f"Actual Pump Depth: {pump_depth:.2f} ft")
            st.write(f"Recommended Sump Depth: {recommended_sump_depth:.2f} ft")
            st.write(f"Actual Sump Depth: {sump_depth:.2f} ft")
            if pump_depth < recommended_pump_depth:
                st.warning("Consider lowering the pump to the recommended depth for optimal performance.")
            if sump_depth < recommended_sump_depth:
                st.warning("Consider increasing the sump depth to ensure adequate fluid storage.")

        if is_deviated and uploaded_file is not None:
            with st.expander("Deviated Well Analysis"):
                deviation_df = extract_deviation_data_from_pdf(uploaded_file)
                if deviation_df is not None:
                    deviation_df = process_deviation_data(deviation_df)
                    st.dataframe(deviation_df)
                    deviation_angle = deviation_df['Angle'].max()
                    wellpath_fig = plot_wellpath(deviation_df)
                    st.pyplot(wellpath_fig)
                    st.subheader("Dog Leg Severity Analysis")
                    max_dls = deviation_df['DLS'].max()
                    avg_dls = deviation_df['DLS'].mean()
                    st.write(f"Maximum Dog Leg Severity: {max_dls:.2f} °/100ft")
                    st.write(f"Average Dog Leg Severity: {avg_dls:.2f} °/100ft")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(deviation_df['SD'], deviation_df['DLS'])
                    ax.set_xlabel('Measured Depth (m)')
                    ax.set_ylabel('Dog Leg Severity (°/100ft)')
                    ax.set_title('Dog Leg Severity vs. Measured Depth')
                    ax.grid(True)
                    st.pyplot(fig)
                else:
                    st.error("Failed to extract data from the PDF. Please check the file format.")

        with st.expander("Gas Interference Analysis"):
            gas_interference_factor = gas_rate / (gas_rate + liquid_rate * 5.615)
            st.write(f"Gas Interference Factor: {gas_interference_factor:.2f}")
            if gas_interference_factor > 0.3:
                st.warning("High gas interference detected. Consider the following:")
                st.write("- Install or optimize gas separator")
                st.write("- Increase pump submergence")
                st.write("- Reduce pumping speed")
            else:
                st.info("Moderate to low gas interference detected. Monitor closely.")

        with st.expander("Rod String Stress Analysis"):
            for i, (diameter, length) in enumerate(rod_string):
                section_stress = max_stress * (diameter / rod_string[0][0])**2
                safety_factor = yield_strength / section_stress
                st.write(f"Section {i+1}: {diameter:.3f} inch diameter, {length:.2f} ft length")
                st.write(f"  Maximum stress: {section_stress:.2f} psi")
                st.write(f"  Safety factor: {safety_factor:.2f}")
                if safety_factor < 1.3:
                    st.error(f"Warning: Low safety factor in section {i+1}. Consider redesigning rod string.")
                elif safety_factor > 2.5:
                    st.info(f"Section {i+1} may be oversized. Consider optimizing for cost.")
                else:
                    st.success(f"Section {i+1} design is acceptable.")

        with st.expander("Pump Efficiency Analysis"):
            theoretical_production = pump_area * surface_stroke_length / 12 * pump_speed * 1440 / 5.615  # bbl/day
            pump_efficiency = liquid_rate / theoretical_production * 100
            st.write(f"Theoretical Production: {theoretical_production:.2f} bbl/day")
            st.write(f"Actual Production: {liquid_rate:.2f} bbl/day")
            st.write(f"Pump Efficiency: {pump_efficiency:.2f}%")
            if pump_efficiency < 70:
                st.warning("Low pump efficiency. Consider the following:")
                st.write("- Check for worn pump components")
                st.write("- Verify pump size and stroke length are appropriate for the well")
                st.write("- Investigate potential gas interference or fluid pound issues")

        with st.expander("Power Consumption Analysis"):
            hydraulic_hp = 7.36e-6 * liquid_rate * fluid_density * pump_depth
            friction_hp = 6.31e-7 * rod_weight * surface_stroke_length * pump_speed
            total_hp = hydraulic_hp + friction_hp
            st.write(f"Hydraulic Horsepower: {hydraulic_hp:.2f} hp")
            st.write(f"Friction Horsepower: {friction_hp:.2f} hp")
            st.write(f"Total Horsepower: {total_hp:.2f} hp")

        with st.expander("Economic Analysis"):
            col1, col2 = st.columns(2)
            with col1:
                if is_gas_well:
                    gas_price = st.number_input("Gas Price ($/MCF)", value=3.0, step=0.1)
                    condensate_price = st.number_input("Condensate Price ($/bbl)", value=50.0, step=1.0)
                else:
                    oil_price = st.number_input("Oil Price ($/bbl)", value=50.0, step=1.0)
            with col2:
                operating_cost = st.number_input("Operating Cost ($/day)", value=100.0, step=10.0)
            
            if is_gas_well:
                daily_revenue = gas_rate * gas_price / 1000 + liquid_rate * (1 - water_cut/100) * condensate_price
            else:
                daily_revenue = liquid_rate * (1 - water_cut/100) * oil_price
            
            daily_profit = daily_revenue - operating_cost
            
            st.metric("Daily Revenue", f"${daily_revenue:.2f}")
            st.metric("Daily Profit", f"${daily_profit:.2f}")
            st.metric("Monthly Profit", f"${daily_profit * 30:.2f}")

            if daily_profit < 0:
                st.error("Warning: The well is currently operating at a loss. Consider optimizing production or reducing costs.")
            elif daily_profit < 100:
                st.warning("The well is marginally profitable. Consider ways to increase production or reduce costs.")
            else:
                st.success("The well is operating profitably.")

        with st.expander("Optimization Suggestions"):
            suggestions = []
            if pump_efficiency < 70:
                suggestions.append("Consider adjusting pump speed or stroke length to improve pump efficiency.")
            if max_stress > 0.8 * yield_strength:
                suggestions.append("Evaluate the possibility of using a higher grade rod or increasing rod size to reduce stress.")
            if gas_interference_factor > 0.2:
                suggestions.append("Investigate options for better gas separation to reduce gas interference.")
            if daily_profit < 200:
                suggestions.append("Explore ways to increase production or reduce operating costs to improve profitability.")
            
            if suggestions:
                for suggestion in suggestions:
                    st.write(f"- {suggestion}")
            else:
                st.write("No immediate optimization suggestions. Continue monitoring performance.")

        with st.expander("Future Production Forecast"):
            col1, col2 = st.columns(2)
            with col1:
                decline_rate = st.number_input("Annual Decline Rate (%)", min_value=0.0, max_value=50.0, value=10.0, step=1.0)
            with col2:
                forecast_years = st.number_input("Forecast Years", min_value=1, max_value=20, value=5, step=1)
            
            years = np.arange(forecast_years + 1)
            forecasted_production = liquid_rate * (1 - decline_rate/100) ** years
            cumulative_production = np.cumsum(forecasted_production * 365)
            
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = ax1.twinx()
            ax1.plot(years, forecasted_production, 'b-', label='Daily Production')
            ax2.plot(years, cumulative_production, 'r--', label='Cumulative Production')
            ax1.set_xlabel('Years')
            ax1.set_ylabel('Daily Production (bbl/day)', color='b')
            ax2.set_ylabel('Cumulative Production (bbl)', color='r')
            ax1.tick_params(axis='y', labelcolor='b')
            ax2.tick_params(axis='y', labelcolor='r')
            plt.title('Production Forecast')
            fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
            st.pyplot(fig)
            
            st.metric("Estimated Cumulative Production", f"{cumulative_production[-1]:.0f} bbl", f"after {forecast_years} years")

        with st.expander("Conclusion"):
            st.write("Based on the analysis, here are the key takeaways:")
            st.write(f"1. The current rod pump design has a safety factor of {key_results['Safety Factor']:.2f} for yield strength.")
            st.write(f"2. The pump is operating at {pump_efficiency:.2f}% efficiency.")
            st.write(f"3. The well is currently generating a daily profit of ${daily_profit:.2f}.")
            st.write(f"4. Over the next {forecast_years} years, the well is expected to produce {cumulative_production[-1]:.0f} bbl cumulatively.")
            st.write("5. Regular monitoring and optimization of the rod pump system is recommended to maintain efficiency and profitability.")

        progress_bar.progress(100)
        status_text.text("Analysis complete!")

if __name__ == "__main__":
    main()
        

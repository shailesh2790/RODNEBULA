import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
import base64

st.set_page_config(page_title="Advanced Gas Well Deliquification Rod Pump Designer", layout="wide")

st.title("Advanced Gas Well Deliquification Rod Pump Designer")

# Sidebar for input parameters
st.sidebar.header("Input Parameters")

# Well Data
st.sidebar.subheader("Well Data")
well_depth = st.sidebar.number_input("Well Depth (m)", min_value=700.0, max_value=3500.0, value=2000.0)
tubing_id = st.sidebar.number_input("Tubing ID (inches)", min_value=1.0, max_value=4.0, value=2.375)
casing_id = st.sidebar.number_input("Casing ID (inches)", min_value=4.0, max_value=9.0, value=5.5)
gas_gravity = st.sidebar.number_input("Gas Specific Gravity (air=1)", min_value=0.5, max_value=1.5, value=0.65)
gas_rate = st.sidebar.number_input("Gas Production Rate (m3/day)", min_value=1000.0, max_value=100000.0, value=20000.0)

# Fluid Data
st.sidebar.subheader("Liquid Data")
water_cut = st.sidebar.number_input("Water Cut (%)", min_value=0.0, max_value=100.0, value=80.0)
condensate_gravity = st.sidebar.number_input("Condensate Gravity (API)", min_value=20.0, max_value=70.0, value=45.0)
liquid_rate = st.sidebar.number_input("Liquid Production Rate (m3/day)", min_value=0.1, max_value=50.0, value=5.0)

# Pump Data
st.sidebar.subheader("Pump Data")
pump_depth = st.sidebar.number_input("Pump Depth (m)", min_value=500.0, max_value=well_depth, value=1800.0)
pump_diameter = st.sidebar.selectbox("Pump Diameter (inches)", [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75])
surface_stroke_length = st.sidebar.number_input("Surface Stroke Length (inches)", min_value=20.0, max_value=200.0, value=48.0)
pump_speed = st.sidebar.number_input("Pump Speed (strokes/min)", min_value=1.0, max_value=30.0, value=12.0)

# Rod String Data
st.sidebar.subheader("Rod String Data")
rod_grades = ["D", "K", "C"]
rod_grade = st.sidebar.selectbox("Rod Grade", rod_grades)

# Surface Unit Data
st.sidebar.subheader("Surface Unit Data")
unit_type = st.sidebar.selectbox("Unit Type", ["Conventional", "Mark II", "Air Balanced"])
crank_radius = st.sidebar.number_input("Crank Radius (inches)", min_value=10.0, max_value=100.0, value=24.0)
pitman_length = st.sidebar.number_input("Pitman Length (inches)", min_value=50.0, max_value=300.0, value=120.0)

# Constants
g = 9.81  # Acceleration due to gravity (m/s^2)
E = 207e9  # Young's modulus for steel (Pa)
rho_steel = 7850  # Density of steel (kg/m^3)

# Calculations
def calculate_fluid_properties(water_cut, condensate_gravity):
    water_density = 1000  # kg/m^3
    condensate_density = 141.5 / (131.5 + condensate_gravity) * 1000  # kg/m^3
    fluid_density = (water_cut/100 * water_density + (1 - water_cut/100) * condensate_density)
    return fluid_density

def calculate_rod_properties(rod_grade):
    if rod_grade == "D":
        yield_strength = 620e6  # Pa
    elif rod_grade == "K":
        yield_strength = 760e6  # Pa
    else:  # grade C
        yield_strength = 895e6  # Pa
    return yield_strength

def design_rod_string(pump_depth, fluid_load, pump_diameter):
    sections = []
    if pump_depth <= 610:  # 2000 ft
        sections.append((0.75, pump_depth))
    elif pump_depth <= 1220:  # 4000 ft
        sections.append((0.75, pump_depth * 0.7))
        sections.append((0.875, pump_depth * 0.3))
    else:
        sections.append((0.75, pump_depth * 0.5))
        sections.append((0.875, pump_depth * 0.3))
        sections.append((1.0, pump_depth * 0.2))
    return sections

def calculate_rod_area(diameter):
    return np.pi * (diameter / 2)**2

def calculate_fluid_load(depth, diameter, fluid_density):
    area = np.pi * (diameter / 2)**2
    return fluid_density * g * depth * area

def calculate_rod_weight(depth, diameters, lengths):
    weight_per_meter = {
        0.5: 0.62,
        0.625: 0.99,
        0.75: 1.43,
        0.875: 1.95,
        1.0: 2.54,
        1.125: 3.22
    }
    total_weight = sum(weight_per_meter[d] * l * g for d, l in zip(diameters, lengths))
    return total_weight

def surface_position(t, stroke, speed):
    omega = speed * 2 * np.pi / 60
    return 0.5 * stroke * (1 - np.cos(omega * t))

def surface_velocity(t, stroke, speed):
    omega = speed * 2 * np.pi / 60
    return 0.5 * stroke * omega * np.sin(omega * t)

def wave_equation(y, t, L, A, E, rho, c, stroke, speed, fluid_density, g, pump_area, friction_factor, buoyancy_factor):
    n = len(y) // 2
    u, v = y[:n], y[n:]
    
    # Surface boundary condition
    u_surface = surface_position(t, stroke, speed)
    v_surface = surface_velocity(t, stroke, speed)
    
    # Pump boundary condition (with friction and buoyancy)
    fluid_load = fluid_density * g * L * pump_area
    friction_load = friction_factor * fluid_load
    buoyancy_load = buoyancy_factor * fluid_load
    net_load = fluid_load + friction_load - buoyancy_load
    u_pump = max(0, u[-1] - net_load / (E * A))
    
    # Wave equation
    du_dt = v
    dv_dt = c**2 * np.gradient(np.gradient(u, L), L) - g * (1 - buoyancy_factor)
    
    # Apply boundary conditions
    du_dt[0] = v_surface
    dv_dt[0] = c**2 * (u[1] - u_surface) / (L**2)
    du_dt[-1] = v[-1]
    dv_dt[-1] = c**2 * (u_pump - u[-2]) / (L**2)
    
    return np.concatenate([du_dt, dv_dt])

def simulate_rod_pump(L, A, E, rho, stroke, speed, fluid_density, g, pump_area, t_span, n_points, friction_factor, buoyancy_factor):
    c = np.sqrt(E / rho)  # Wave speed in the rod
    
    # Initial conditions
    u0 = np.zeros(n_points)
    v0 = np.zeros(n_points)
    y0 = np.concatenate([u0, v0])
    
    # Time array
    t = np.linspace(t_span[0], t_span[1], 1000)
    
    # Solve ODE
    sol = odeint(wave_equation, y0, t, args=(L, A, E, rho, c, stroke, speed, fluid_density, g, pump_area, friction_factor, buoyancy_factor))
    
    return t, sol

def calculate_polished_rod_load(surface_load, rod_weight):
    return surface_load + rod_weight

def calculate_counterbalance_weight(max_polished_rod_load, min_polished_rod_load):
    return (max_polished_rod_load + min_polished_rod_load) / 2

def calculate_torque_factor(crank_angle, crank_radius, pitman_length):
    return (crank_radius * np.sin(crank_angle) + 
            (crank_radius**2 * np.sin(crank_angle) * np.cos(crank_angle)) / 
            np.sqrt(pitman_length**2 - crank_radius**2 * np.sin(crank_angle)**2))

def calculate_gearbox_torque(polished_rod_load, torque_factor):
    return polished_rod_load * torque_factor

def calculate_neutral_point(rod_weight, fluid_load):
    return rod_weight / (rod_weight + fluid_load) * pump_depth

# Main calculations
fluid_density = calculate_fluid_properties(water_cut, condensate_gravity)
yield_strength = calculate_rod_properties(rod_grade)
rod_string = design_rod_string(pump_depth, calculate_fluid_load(pump_depth, pump_diameter, fluid_density), pump_diameter)
rod_weight = calculate_rod_weight(pump_depth, [r[0] for r in rod_string], [r[1] for r in rod_string])

# Rod string properties
L = pump_depth
A = calculate_rod_area(rod_string[0][0])  # Use the top rod section for simplification
rho = rho_steel

# Simulation parameters
stroke = surface_stroke_length / 39.37  # Convert to meters
speed = pump_speed
pump_area = np.pi * (pump_diameter / 2 / 39.37)**2  # Convert to square meters

t_span = (0, 60 / speed)  # Simulate one complete stroke
n_points = 100

# Friction and buoyancy factors
friction_factor = 0.1  # Adjust as needed
buoyancy_factor = 0.2  # Adjust as needed

# Run simulation
t, sol = simulate_rod_pump(L, A, E, rho, stroke, speed, fluid_density, g, pump_area, t_span, n_points, friction_factor, buoyancy_factor)

# Extract surface position and load
surface_pos = sol[:, 0]
surface_load = E * A * np.gradient(sol[:, 0], L)

# Calculate polished rod load
polished_rod_load = calculate_polished_rod_load(surface_load, rod_weight)

# Calculate counterbalance weight
max_polished_rod_load = np.max(polished_rod_load)
min_polished_rod_load = np.min(polished_rod_load)
counterbalance_weight = calculate_counterbalance_weight(max_polished_rod_load, min_polished_rod_load)

# Calculate gearbox torque
crank_angles = np.linspace(0, 2*np.pi, len(t))  # Match the number of time steps
torque_factors = calculate_torque_factor(crank_angles, crank_radius/39.37, pitman_length/39.37)  # Convert to meters
gearbox_torque = calculate_gearbox_torque(polished_rod_load, torque_factors)

# Calculate neutral point
fluid_load = calculate_fluid_load(pump_depth, pump_diameter/39.37, fluid_density)
neutral_point = calculate_neutral_point(rod_weight, fluid_load)

# Display results
st.header("Simulation Results")

# Plot dynacard
st.subheader("Surface Dynamometer Card")
fig, ax = plt.subplots()
ax.plot(surface_pos * 39.37, polished_rod_load / 4.448)  # Convert to inches and lbs
ax.set_xlabel("Position (inches)")
ax.set_ylabel("Polished Rod Load (lbs)")
ax.set_title("Surface Dynamometer Card")
st.pyplot(fig)

# Plot position and load vs time
st.subheader("Position and Load vs Time")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(t, surface_pos * 39.37)  # Convert to inches
ax1.set_ylabel("Position (inches)")
ax1.set_title("Surface Position vs Time")

ax2.plot(t, polished_rod_load / 4.448)  # Convert to lbs
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Polished Rod Load (lbs)")
ax2.set_title("Polished Rod Load vs Time")

st.pyplot(fig)

# Plot gearbox torque
st.subheader("Gearbox Torque")
fig, ax = plt.subplots()
ax.plot(crank_angles, gearbox_torque / 1.356)  # Convert to ft-lbs
ax.set_xlabel("Crank Angle (radians)")
ax.set_ylabel("Gearbox Torque (ft-lbs)")
ax.set_title("Gearbox Torque vs Crank Angle")
st.pyplot(fig)

# Display key results
st.subheader("Key Results")
st.write(f"Maximum Polished Rod Load: {max_polished_rod_load/4.448:.2f} lbs")
st.write(f"Minimum Polished Rod Load: {min_polished_rod_load/4.448:.2f} lbs")
st.write(f"Peak-to-Peak Load: {(max_polished_rod_load - min_polished_rod_load)/4.448:.2f} lbs")
st.write(f"Counterbalance Weight: {counterbalance_weight/4.448:.2f} lbs")
st.write(f"Neutral Point: {neutral_point:.2f} m")

# Stress calculations
top_rod_area = np.pi * (rod_string[0][0]/2/39.37)**2  # Convert to square meters
max_stress = np.max(surface_load) / top_rod_area / 6894.76  # Convert to psi
st.write(f"Maximum Stress: {max_stress:.2f} psi")
st.write(f"Yield Strength: {yield_strength/6894.76:.2f} psi")

if max_stress > yield_strength/6894.76:
    st.error("Warning: Maximum stress exceeds yield strength!")
else:
    st.success("Rod string design is within yield strength limits.")

# Rod string design
st.subheader("Rod String Design")
for i, (diameter, length) in enumerate(rod_string):
    st.write(f"Section {i+1}: {diameter} inch diameter, {length:.2f} m length")



# Function to create PDF
def create_pdf(fig1, fig2, fig3, key_results, rod_string):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Save figures as images
    img_buffer1 = BytesIO()
    fig1.savefig(img_buffer1, format='png')
    img_buffer1.seek(0)

    img_buffer2 = BytesIO()
    fig2.savefig(img_buffer2, format='png')
    img_buffer2.seek(0)

    img_buffer3 = BytesIO()
    fig3.savefig(img_buffer3, format='png')
    img_buffer3.seek(0)

    # Add title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Gas Well Deliquification Rod Pump Design Results")

    # Add figures
    c.drawImage(ImageReader(img_buffer1), 50, height - 300, width=500, height=200)
    c.drawImage(ImageReader(img_buffer2), 50, height - 550, width=500, height=200)
    c.showPage()  # New page
    c.drawImage(ImageReader(img_buffer3), 50, height - 300, width=500, height=200)

    # Add key results
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 350, "Key Results")
    c.setFont("Helvetica", 12)
    y = height - 380
    for key, value in key_results.items():
        c.drawString(50, y, f"{key}: {value}")
        y -= 20

    # Add rod string design
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y - 30, "Rod String Design")
    c.setFont("Helvetica", 12)
    y -= 60
    for i, (diameter, length) in enumerate(rod_string):
        c.drawString(50, y, f"Section {i+1}: {diameter} inch diameter, {length:.2f} m length")
        y -= 20

    c.save()
    buffer.seek(0)
    return buffer

# Display results
st.header("Simulation Results")



# Create a dictionary of key results
key_results = {
    "Maximum Polished Rod Load": f"{max_polished_rod_load/4.448:.2f} lbs",
    "Minimum Polished Rod Load": f"{min_polished_rod_load/4.448:.2f} lbs",
    "Peak-to-Peak Load": f"{(max_polished_rod_load - min_polished_rod_load)/4.448:.2f} lbs",
    "Counterbalance Weight": f"{counterbalance_weight/4.448:.2f} lbs",
    "Neutral Point": f"{neutral_point:.2f} m",
    "Maximum Stress": f"{max_stress:.2f} psi",
    "Yield Strength": f"{yield_strength/6894.76:.2f} psi"
}

# Add download button for PDF
if st.button("Generate PDF Report"):
    pdf = create_pdf(fig, fig, fig, key_results, rod_string)
    b64 = base64.b64encode(pdf.getvalue()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="rod_pump_design_results.pdf">Download PDF Report</a>'
    st.markdown(href, unsafe_allow_html=True)

st.sidebar.info("This is an advanced prototype for gas well deliquification using rod pumps. For a full design, consult with shailesh")
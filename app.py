# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ga_module import(
    generate_coordinates,
    run_ga,
    plot_routes,
    evalVRP,
)

# page 1 configurations

st.set_page_config(
    page_title="VRP Genetic Algorithm UI", # setting the title of the tab
    layout="wide",  # keeping the size of the column wider
    initial_sidebar_state="expanded",  # using this because our sidebar contains controls for number of locations , vehicles
)

# Title Description
st.title("Vehicle Routing Problem (VRP) - Genetic Algorithm") # title on the web page

# written for the guidance of user  how to use the website
with st.expander("ℹ️ How to Use This App"):
    st.write("""
        - Select number of locations and vehicles.
        - Adjust Genetic Algorithm (GA)  parameters.
        - Generate new locations or upload your own.
        - Run Genetic Algorithm (GA) to find optimized routes.
        - Download routes or data as CSV.
    """)

# building the sidebar controls

with st.sidebar:
    # allowing user to choose the number of locations and number of vehicles
    st.header("1. Problem Settings")

    # choosing the number of locations
    num_locations = st.slider(
        "Number of Locations",
        min_value=5,
        max_value=50,
        value=19,
        step=1,
        # if user got stuck what is this click on the question mark near this option it will get the following message
        help="Total number of delivery locations (excluding the depot).",
    )

    # choosing the number of vehicles
    num_vehicles = st.slider(
        "Number of Vehicles",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        # if user is not getting what is this option then click on the question mark near this option he will get the following message
        help="How many vehicles to dispatch.",
    )


    # putting one horizontal line in between
    st.markdown("---")

    # now choosing the parameters of the genetic algorithm
    st.header("2. Genetic Algorithm Parameters")

    pop_size = st.number_input(
        "Population Size", min_value=10, max_value=500, value=200, step=10
    )

    cxpb = st.slider("Crossover Probability (cxpb)", 0.0, 1.0, 0.7, 0.05)
    mutpb = st.slider("Mutation Probability (mutpb)", 0.0, 1.0, 0.2, 0.05)
    tournsize = st.number_input(
        "Tournament Size", min_value=2, max_value=10, value=3, step=1
    )

    ngen = st.number_input(
        "Generations (ngen)", min_value=1, max_value=200, value=30, step=1
    )

    st.markdown("---")
    st.header("3. Generate / Run")

    uploaded_file = st.file_uploader("Upload Locations CSV",type=["csv"])
    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        st.session_state.locations = list(zip(uploaded_df["X"], uploaded_df["Y"]))
        st.session_state.depot=(100.0, 100.0)
        st.success("📤 Uploaded locations loaded successfully!")

    new_locs = st.button("Generate New Locations")
    run_button = st.button("Run GA")

# ── 4. Session state: hold locations + depot ───────────────────────────────────
if "locations" not in st.session_state or new_locs:
    # make the new coordinates
    st.session_state.locations = generate_coordinates(num_locations)

    # setting the depot value
    st.session_state.depot = (100.0, 100.0)

# storing the locations values saved into the locations variable
locations = st.session_state.locations

# storing the coordinates of the depot into the depot variable
depot = st.session_state.depot

# setting the short title location coordinates
st.subheader("Location Coordinates")

# creating the dataframe of coordinates using pd.dataframe
coords_df = pd.DataFrame(locations, columns=["X", "Y"])
coords_df.index.name = "Index"  # giving the index name
st.dataframe(coords_df.style.format({"X": "{:.2f}", "Y": "{:.2f}"}), height=250)  # formatting the coordiantes of the locations upto 2 decimal places only

# when the run GA button is clicked
if run_button:
    with st.spinner("Running Genetic Algorithm..."):
        progress_bar = st.progress(0)

        def update_progress(gen, ngen):
            progress_bar.progress(min((gen+1)/ngen, 1.0))

        # calling the function sending the parameters and storing the results of the functions into the variable names
        best_individual, logbook = run_ga(
            locations=locations,
            depot=depot,
            num_vehicles=num_vehicles,
            pop_size=pop_size,
            cxpb=cxpb,
            mutpb=mutpb,
            tournsize=tournsize,
            ngen=ngen,
            random_seed=42,
            progress_callback=update_progress
        )

        progress_bar.empty()

    st.success("GA finished!")  # telling that the algorithm successfully executed.

    # putting the values in the variable which will be used in the plotting
    gens = logbook.select("gen")
    min_fits = logbook.select("min")
    avg_fits = logbook.select("avg")

    # plotting the minimum and average fitness curves over generations
    fig1, ax1 = plt.subplots()
    ax1.plot(gens, min_fits, label="Min Fitness")
    ax1.plot(gens, avg_fits, label="Avg Fitness", linestyle="--")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.set_title("📉 Genetic Algorithm Fitness Over Generations")
    ax1.legend()
    st.pyplot(fig1, use_container_width=True) # using the container width=True to make the plot wider and broader

    # optimized route plots
    route_fig = plot_routes(
        individual=best_individual,
        locations=locations,
        depot=depot,
        num_vehicles=num_vehicles,
        title="Optimized Vehicle Routes",
    )
    st.pyplot(route_fig, use_container_width=True)

    total_dist, variance = evalVRP(best_individual, locations, depot, num_vehicles)  # storing the values of total_distance and variance which were calculated using the
    #  evalVRP function which is declared in the ga_module.py

    st.subheader("📊 Route Metrics")
    st.metric("Total Distance travelled by all the trucks",f"{total_dist:.2f}")
    st.metric("Variance between distances travelled by each truck ",f"{variance:.2f}")
    st.metric("Avg Distance/Vehicle",f"{total_dist/num_vehicles:.2f}")

    # creating the dataframe which will tell us which location number is taken at which step
    route_order = pd.DataFrame(
        {
            "Step": list(range(len(best_individual))),
            "Location Index": best_individual,
        }
    )

    csv_bytes = route_order.to_csv(index=False).encode("utf-8")  # converts the dataframe into csv file using the string formatting
    st.download_button( # add the download button
        label="Download Best Route (CSV)",
        data=csv_bytes,
        file_name="best_route.csv",
        mime="text/csv",  # tells the browser that this is a csv file
    )

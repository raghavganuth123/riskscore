import streamlit as st
import pandas as pd
from datetime import datetime
import os
from geopy.distance import geodesic
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
TRANSACTIONS_FILE = 'transactions.csv'
BLOCKLIST_FILE = 'blocklist.txt'
FRAUD_THRESHOLD = 90 # Score above which user is added to blocklist
SUSPICIOUS_THRESHOLD = 50 # Score above which user is marked suspicious

# -------------------------
# 1. Transaction data handling
# -------------------------
# @st.cache_data # Caching might be too aggressive if files change often outside the app
def load_transactions():
    """Loads transactions from the CSV file into a pandas DataFrame."""
    if os.path.exists(TRANSACTIONS_FILE):
        try:
            df = pd.read_csv(TRANSACTIONS_FILE)
            # Ensure timestamp column is parsed correctly
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Ensure latitude/longitude are numeric
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            df.dropna(subset=['latitude', 'longitude'], inplace=True) # Drop rows where conversion failed
            return df
        except Exception as e:
            st.error(f"Error loading transactions file ({TRANSACTIONS_FILE}): {e}")
            # Return empty dataframe if loading fails
            return pd.DataFrame(columns=['sender', 'receiver', 'amount', 'timestamp', 'latitude', 'longitude'])
    return pd.DataFrame(columns=['sender', 'receiver', 'amount', 'timestamp', 'latitude', 'longitude'])

def save_transactions(df):
    """Saves the DataFrame to the transactions CSV file."""
    try:
        df.to_csv(TRANSACTIONS_FILE, index=False)
    except Exception as e:
        st.error(f"Error saving transactions file ({TRANSACTIONS_FILE}): {e}")

# -------------------------
# 1b. Blocklist handling
# -------------------------
# @st.cache_data # Caching might be too aggressive if files change often outside the app
def load_blocklist():
    """Loads blocked user IDs from the blocklist file into a set."""
    if not os.path.exists(BLOCKLIST_FILE):
        return set()
    try:
        with open(BLOCKLIST_FILE, 'r') as f:
            # Read lines, strip whitespace, ignore empty lines
            blocked_users = {line.strip() for line in f if line.strip()}
        return blocked_users
    except Exception as e:
        st.error(f"Error loading blocklist file ({BLOCKLIST_FILE}): {e}")
        return set() # Return empty set on error

def save_blocklist(blocked_users_set):
    """Saves the set of blocked user IDs to the blocklist file."""
    try:
        with open(BLOCKLIST_FILE, 'w') as f:
            for user_id in sorted(list(blocked_users_set)): # Save sorted list
                f.write(f"{user_id}\n")
    except Exception as e:
        st.error(f"Error saving blocklist file ({BLOCKLIST_FILE}): {e}")


# -------------------------
# 2. Rule-based Risk Scoring
# (Function remains the same as original)
# -------------------------
def compute_rule_based_score(df):
    """Computes risk scores based on predefined rules."""
    if df.empty:
        return {}

    risk_scores = {}
    last_locations = {}
    last_timestamps = {}

    # Pre-calculate daily averages efficiently
    df_copy = df.copy() # Work on a copy to avoid modifying the original df in session state directly
    df_copy['date'] = df_copy['timestamp'].dt.date
    daily_averages = df_copy.groupby(['sender', 'date'])['amount'].mean().to_dict()
    # Default average if no history (can be tuned)
    global_avg = df_copy['amount'].mean() if not df_copy.empty else 200

    # Ensure dataframe is sorted for time diff and location check logic
    df_sorted = df_copy.sort_values(by=['sender', 'timestamp']).reset_index()

    for idx, row in df_sorted.iterrows():
        sender = row['sender']
        score = 0

        # --- Rule 1: Amount exceeds dynamic threshold ---
        daily_key = (sender, row['date'])
        # Use sender's daily average, fallback to global average if needed
        avg_daily = daily_averages.get(daily_key, global_avg)
        threshold = avg_daily * 2.5 # Threshold is 2.5 times the average
        if row['amount'] > threshold:
            score += 3

        # --- Rule 2: Rapid successive transactions ---
        if sender in last_timestamps:
            time_diff = (row['timestamp'] - last_timestamps[sender]).total_seconds() / 60 # Difference in minutes
            if time_diff < 5: # Less than 5 minutes apart
                score += 2
        last_timestamps[sender] = row['timestamp'] # Update last timestamp for sender

        # --- Rule 3: Geolocation jump ---
        curr_loc = (row['latitude'], row['longitude'])
        if sender in last_locations:
            try:
                # Ensure coordinates are valid floats before calculating distance
                if isinstance(curr_loc[0], (int, float)) and isinstance(curr_loc[1], (int, float)) and \
                   isinstance(last_locations[sender][0], (int, float)) and isinstance(last_locations[sender][1], (int, float)):
                    dist = geodesic(curr_loc, last_locations[sender]).km
                    if dist > 500: # Significant jump (e.g., > 500 km)
                        score += 4
                else:
                     # Silently skip distance check if coords invalid (already cleaned on load)
                     pass
            except ValueError as e:
                 # Handle potential issues with geodesic calculation
                 # st.warning(f"Could not calculate distance for user {sender} at index {idx}. Error: {e}") # Optional warning
                 pass
        # Update last location only if current location coordinates are valid
        if isinstance(curr_loc[0], (int, float)) and isinstance(curr_loc[1], (int, float)):
            last_locations[sender] = curr_loc # Update last location for sender


        # Accumulate score for the user across all their transactions
        risk_scores[sender] = risk_scores.get(sender, 0) + score

    return risk_scores

# -------------------------
# 3. ML-based Anomaly Detection
# (Function remains the same as original)
# -------------------------
def compute_ml_scores(df):
    """Computes anomaly scores using Isolation Forest based on transaction amount."""
    if df.empty or 'amount' not in df.columns or len(df) < 2: # Need at least 2 samples for scaling/IF
        return {}

    # Ensure 'amount' is numeric and handle potential NaNs introduced earlier
    features = df[['amount']].copy()
    features['amount'] = pd.to_numeric(features['amount'], errors='coerce')
    features.dropna(inplace=True) # Drop rows where amount is not a number

    if features.empty or len(features) < 2: # Check again after dropna
        return {}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Adjust contamination based on expected anomaly rate, or keep it small
    iso = IsolationForest(contamination=0.05, random_state=42) # Lowered contamination

    # Add prediction back to original index positions
    df_valid_indices = features.index # Indices where amount was valid
    predictions = iso.fit_predict(X_scaled) # Predicts -1 for anomalies, 1 for inliers

    # Create a temporary series with predictions aligned to the original DataFrame index
    ml_anomaly_series = pd.Series(predictions, index=df_valid_indices)

    # Create a copy to avoid modifying the input df directly
    df_with_ml = df.copy()
    # Add the 'ml_anomaly' column, filling with a default (e.g., 1 for inlier) where prediction wasn't possible
    df_with_ml['ml_anomaly'] = ml_anomaly_series.reindex(df.index, fill_value=1)


    ml_scores = {}
    # Calculate ML contribution to score per sender
    # Assign score points for each transaction flagged as an anomaly (-1)
    anomaly_scores = df_with_ml[df_with_ml['ml_anomaly'] == -1].groupby('sender').size() * 5 # 5 points per anomaly
    ml_scores = anomaly_scores.to_dict()

    # Ensure all users in the original df have an entry, even if 0
    for user in df['sender'].unique():
        if user not in ml_scores:
            ml_scores[user] = 0

    return ml_scores


# ---------------------------------------------------
# Streamlit App Logic
# ---------------------------------------------------

st.set_page_config(layout="wide")

# Initialize session state
if 'transactions_df' not in st.session_state:
    st.session_state.transactions_df = load_transactions()

if 'blocklist' not in st.session_state:
    st.session_state.blocklist = load_blocklist()

# --- Sidebar Navigation ---
st.sidebar.title("🛡️ Risk Scoring System")
options = ["📊 View Scores & Update Blocklist", "➕ Add New Transaction", "🚫 View Blocklist", "📄 View Raw Transactions"]
choice = st.sidebar.radio("Select an option:", options)
st.sidebar.markdown("---")
st.sidebar.info(f"Transactions File: `{TRANSACTIONS_FILE}`\n\nBlocklist File: `{BLOCKLIST_FILE}`")


# --- Main Page Content ---

if choice == "➕ Add New Transaction":
    st.header("➕ Add New Transaction")
    st.markdown("Enter the details for the new transaction below.")

    # Use a form for better input grouping
    with st.form("transaction_form", clear_on_submit=True):
        sender = st.text_input("Sender:", key="sender_input")
        receiver = st.text_input("Receiver:", key="receiver_input")
        amount = st.number_input("Amount:", min_value=0.01, format="%.2f", key="amount_input")
        ip_lat = st.number_input("Latitude:", format="%.6f", key="lat_input", help="Enter the latitude of the transaction origin.")
        ip_long = st.number_input("Longitude:", format="%.6f", key="long_input", help="Enter the longitude of the transaction origin.")

        submitted = st.form_submit_button("Add Transaction")

        if submitted:
            # --- Input Validation ---
            if not sender or not receiver or not amount or ip_lat is None or ip_long is None:
                 st.error("❌ Please fill in all transaction details.")
            elif amount <= 0:
                 st.error("❌ Amount must be positive.")
            else:
                # --- Blocklist Check ---
                blocklist = st.session_state.blocklist # Use blocklist from session state

                # Check SENDER first
                if sender in blocklist:
                    st.error(f"❌ Transaction REJECTED: Sender '{sender}' is on the blocklist.")
                    st.warning(f"🚨 ALERT: Blocked sender '{sender}' attempted transaction from:")
                    st.code(f"Latitude: {ip_lat}, Longitude: {ip_long}")
                    # Generate Map URLs
                    Maps_url = f"https://www.google.com/maps?q={ip_lat},{ip_long}"
                    openstreetmap_url = f"https://www.openstreetmap.org/?mlat={ip_lat}&mlon={ip_long}#map=15/{ip_lat}/{ip_long}"
                    st.markdown("🗺️ **View on map:**")
                    st.markdown(f"- [Google Maps]({Maps_url})")
                    st.markdown(f"- [OpenStreetMap]({openstreetmap_url})")

                # Check RECEIVER
                elif receiver in blocklist:
                    st.error(f"❌ Transaction REJECTED: Receiver '{receiver}' is on the blocklist.")

                # --- If checks passed, proceed ---
                else:
                    timestamp = datetime.now()
                    new_transaction = pd.DataFrame([[sender, receiver, amount, timestamp, ip_lat, ip_long]],
                                                   columns=['sender', 'receiver', 'amount', 'timestamp', 'latitude', 'longitude'])

                    # Use pd.concat to add the new row
                    updated_df = pd.concat([st.session_state.transactions_df, new_transaction], ignore_index=True)

                    # Ensure correct types before sorting and saving
                    updated_df['timestamp'] = pd.to_datetime(updated_df['timestamp'])
                    updated_df['latitude'] = pd.to_numeric(updated_df['latitude'], errors='coerce')
                    updated_df['longitude'] = pd.to_numeric(updated_df['longitude'], errors='coerce')
                    updated_df.dropna(subset=['latitude', 'longitude'], inplace=True) # Re-apply dropna just in case

                    # Sort and reset index
                    updated_df.sort_values(by=["sender", "timestamp"], inplace=True)
                    updated_df.reset_index(drop=True, inplace=True)

                    # Save to file and update session state
                    save_transactions(updated_df)
                    st.session_state.transactions_df = updated_df # Update the state
                    st.success(f"✅ Transaction from '{sender}' to '{receiver}' added successfully.")
                    st.balloons() # Fun feedback!

elif choice == "📊 View Scores & Update Blocklist":
    st.header("📊 User Risk Scores")
    st.markdown("Calculates risk scores based on rules and machine learning. Users exceeding the fraud threshold will be added to the blocklist.")

    # Button to trigger evaluation
    if st.button("Calculate Scores & Update Blocklist Now"):
        df_eval = st.session_state.transactions_df.copy() # Work on a copy

        if df_eval.empty:
            st.warning("⚠️ No transactions found. Please add transactions first.")
        else:
            with st.spinner("Calculating scores..."):
                # Ensure sorting for rule-based scoring
                df_eval_sorted = df_eval.sort_values(by=['sender', 'timestamp']).copy()

                rule_scores = compute_rule_based_score(df_eval_sorted)
                ml_scores = compute_ml_scores(df_eval_sorted) # Pass the same sorted df copy

                # --- Combine Scores and Display ---
                final_scores = {}
                all_users = set(rule_scores.keys()).union(ml_scores.keys())
                for user in all_users:
                    final_scores[user] = rule_scores.get(user, 0) + ml_scores.get(user, 0)

                if not final_scores:
                    st.info("No scores generated (perhaps not enough data).")
                else:
                    # Sort users by final score
                    sorted_risk = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)

                    st.subheader("🧠 Final User Risk Scores (High to Low):")
                    users_to_block = set()
                    score_data = []
                    for user, score in sorted_risk:
                        if score >= FRAUD_THRESHOLD:
                            label = "⚠️ Fraud"
                            status_icon = "🚨"
                            users_to_block.add(user) # Mark user for blocklist
                        elif score >= SUSPICIOUS_THRESHOLD:
                            label = "🔍 Suspicious"
                            status_icon = "⚠️"
                        else:
                            label = "✅ Likely Legitimate"
                            status_icon = "✅"
                        score_data.append({"User": user, "Score": score, "Status": f"{status_icon} {label}"})

                    st.dataframe(pd.DataFrame(score_data), use_container_width=True)

                    # --- Update Blocklist ---
                    if users_to_block:
                        st.markdown("---")
                        st.subheader("🚫 Blocklist Update")
                        st.write(f"Users meeting fraud threshold ({FRAUD_THRESHOLD}+): **{', '.join(users_to_block) or 'None'}**")

                        current_blocklist = st.session_state.blocklist # From state
                        newly_blocked = users_to_block - current_blocklist # Find users not already blocked

                        if newly_blocked:
                            st.warning(f"Adding new users to blocklist: **{', '.join(newly_blocked)}**")
                            updated_blocklist = current_blocklist.union(users_to_block)
                            save_blocklist(updated_blocklist) # Save to file
                            st.session_state.blocklist = updated_blocklist # Update state
                            st.success("Blocklist updated successfully.")
                        else:
                            st.info("All high-risk users are already on the blocklist. No updates needed.")
                    else:
                        st.info("\nNo users met the threshold for blocking in this evaluation.")

elif choice == "🚫 View Blocklist":
    st.header("🚫 Current Blocklist")
    current_blocklist = st.session_state.blocklist
    if current_blocklist:
        st.markdown("The following users are currently blocked:")
        # Display as a list
        for user in sorted(list(current_blocklist)):
            st.markdown(f"- `{user}`")
        # Provide a way to download the blocklist
        st.download_button(
            label="Download Blocklist (.txt)",
            data="\n".join(sorted(list(current_blocklist))),
            file_name="blocklist.txt",
            mime="text/plain"
        )
    else:
        st.info("✅ The blocklist is currently empty.")

elif choice == "📄 View Raw Transactions":
    st.header("📄 Raw Transaction Data")
    st.markdown(f"Displaying data from `{TRANSACTIONS_FILE}`.")
    if not st.session_state.transactions_df.empty:
        # Display the dataframe with some formatting options
        st.dataframe(
            st.session_state.transactions_df.sort_values(by='timestamp', ascending=False).style.format({
                'amount': '{:.2f}',
                'latitude': '{:.6f}',
                'longitude': '{:.6f}',
                'timestamp': '{:%Y-%m-%d %H:%M:%S}'
            }),
            use_container_width=True
        )
        # Provide a way to download the data
        csv = st.session_state.transactions_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Transactions (.csv)",
            data=csv,
            file_name='transactions_export.csv',
            mime='text/csv',
        )
    else:
        st.warning("⚠️ No transactions loaded or available.")

# --- Footer or additional info ---
st.sidebar.markdown("---")
st.sidebar.write(f"Last Refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
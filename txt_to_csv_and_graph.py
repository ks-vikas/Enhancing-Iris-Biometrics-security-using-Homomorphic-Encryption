import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev


# Define input and output file names
input_file = 'output.txt' #text file
output_file = 'result.csv' #csv file

##########################################################################################
'''
This is to convert text file into csv file
'''
# Initialize serial number
serial_no = 1

# Open the input file and read lines
with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    # Create a CSV writer object
    csv_writer = csv.writer(outfile)
    
    # Write the header row
    csv_writer.writerow(['serial no.', 'image1', 'image2', 'hamming distance'])
    
    # Process each line in the input file
    for line in infile:
        # Strip any leading/trailing whitespace and split the line into components
        parts = line.strip().split(',')
        
        # Write the data to the CSV file with the serial number
        csv_writer.writerow([serial_no] + parts)
        
        # Increment the serial number
        serial_no += 1

#################################################################################################
'''
This is to plot the graph of far, frr, cerr with threshold.
'''
# Load the data
def load_data(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # hamming_distance = round(float(row['hamming distance']), 6)
            # data.append((row['image1'], row['image2'], hamming_distance))
            data.append((row['image1'], row['image2'], float(row['hamming distance'])))
    return data

# Calculate FAR and FRR for a given threshold
'''
FAR = number of false positives ÷ (number of false positives + number of true negatives) x 100.
FRR = number of false negatives ÷ (number of false negatives + number of true positives) x 100
'''
def calculate_far_frr(data, threshold):
    false_acceptances = 0
    false_rejections = 0
    total_imposter = 0 #false positive or true negative
    total_genuine = 0
    

    for img1, img2, distance in data:
        if img1.split('_')[0] == img2.split('_')[0]:
            total_genuine += 1
            if distance > threshold:
                false_rejections += 1
        else:
            total_imposter += 1
            if distance <= threshold:
                false_acceptances += 1

    far = false_acceptances / total_imposter if total_imposter > 0 else 0
    frr = false_rejections / total_genuine if total_genuine > 0 else 0
    return far, frr

# Calculate Crossover Error Rate (CERR)
def calculate_cerr(data):
    thresholds = np.linspace(0, 1, 5000)
    min_diff = float('inf')
    cerr_threshold = 0
    far_at_cerr = 0
    frr_at_cerr = 0
    
    
    far_values = [] #to draw graph
    frr_values = [] #to draw graph  

    for threshold in thresholds:
        far, frr = calculate_far_frr(data, threshold)
        
        far_values.append(far) #to draw graph
        frr_values.append(frr) #to draw graph     
        
        if abs(far - frr) < min_diff:
            min_diff = abs(far - frr)
            cerr_threshold = threshold
            far_at_cerr = far
            frr_at_cerr = frr

    return cerr_threshold, far_at_cerr, frr_at_cerr, thresholds, far_values, frr_values


input_file = output_file  # Replace with your input file name
data = load_data(input_file)

# Calculate FAR, FRR, and CERR
cerr_threshold, far, frr, thresholds, far_values, frr_values = calculate_cerr(data)

print(f"Crossover Error Rate (CERR): {cerr_threshold}")
print(f"FAR at CERR: {far}")
print(f"FRR at CERR: {frr}")

# Plotting FAR and FRR straight lines. comment out for curved graph.
plt.figure(figsize=(10, 6))
plt.plot(thresholds, far_values, label='FAR', color='red')
plt.plot(thresholds, frr_values, label='FRR', color='blue')
plt.axvline(cerr_threshold, color='green', linestyle='--', label=f'CERR (threshold={cerr_threshold:.3f})')
# plt.axhline(far, color='gray', linestyle='--')
# plt.axhline(frr, color='gray', linestyle='--')

plt.xlabel('Threshold')
plt.ylabel('Rate')
plt.title('FAR and FRR vs. Threshold')
plt.legend()
plt.grid(True)
plt.show()

# # uncomment for curved graph
# # Use spline interpolation for smooth curves
# far_spline = splrep(thresholds, far_values, s=0)
# frr_spline = splrep(thresholds, frr_values, s=0)

# thresholds_smooth = np.linspace(0, 1, 10000)
# far_smooth = splev(thresholds_smooth, far_spline)
# frr_smooth = splev(thresholds_smooth, frr_spline)

# # Plotting FAR and FRR with smooth curves
# plt.figure(figsize=(10, 6))
# plt.plot(thresholds_smooth, far_smooth, label='FAR', color='red')
# plt.plot(thresholds_smooth, frr_smooth, label='FRR', color='blue')
# plt.axvline(cerr_threshold, color='green', linestyle='--', label=f'CERR (threshold={cerr_threshold:.3f})')
# plt.axhline(far, color='gray', linestyle='--')
# plt.axhline(frr, color='gray', linestyle='--')

# plt.xlabel('Threshold')
# plt.ylabel('Rate')
# plt.title('FAR and FRR vs. Threshold')
# plt.legend()
# plt.grid(True)
# plt.show()

'''
- load weight, sample vector, perfect output - from safetensors
- row by row, extract top 2% outliers, replace them with zeroes
- for every bucket - sort and replace with positional encoding + sign
- do mmul, compare cos sim score


'''

import numpy as np
from safetensors import safe_open
import pdb

tensors = {}
with safe_open("../q4data2-00001-of-00001.safetensors", framework="np") as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

control = tensors['control']
core2 = tensors['core']
v = tensors['v']

print(control)
print(v @ core2.T)


def cossim(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0  # Return 0 if either vector has zero magnitude to avoid division by zero
    cosine_sim = dot_product / (magnitude_vec1 * magnitude_vec2)
    return cosine_sim

def rearrange8(arr):
    # Generate the new order of indices
    n_rows = arr.shape[0]
    groups = (n_rows + 7) // 8  # This rounds up to ensure all rows are covered
    indices = np.array([i * 8 + j for j in range(8) for i in range(groups) if i * 8 + j < n_rows])

    # Reorder the array
    return arr[indices].copy()

def rearrangeOutliers(arr):
    # Assuming 'arr' is your original array with shape (rows, 3) and dtype np.float64
    rows = arr.shape[0]

    # Initialize a new array with shape (rows, 4) and dtype np.float32
    new_arr = np.zeros((rows, 4), dtype=np.float32)

    # Copy the data from the original array to the new array
    new_arr[:, :3] = arr
    return new_arr

print("cos sim:", cossim(control, v @ core2.T))

def convert(core2):
    def extract_outliers(core, perc):
        # Flatten the matrix and calculate absolute values
        flat_core = core.flatten()
        abs_values = np.abs(flat_core)

        # Get the number of elements to consider as the top 2%
        top_2_percent_count = int(len(flat_core) * perc)

        # Get indices of the top 2% values sorted by absolute value in descending order
        indices_of_top_values = np.argsort(-abs_values)[:top_2_percent_count]
        # Get the top 2% values
        top_values = flat_core[indices_of_top_values]

        # Convert flat indices to 2D indices
        top_indices_2d = np.unravel_index(indices_of_top_values, core.shape)

        # Create a table of the top values and their indices
        outliers_table = np.column_stack((top_values, top_indices_2d[0], top_indices_2d[1]))

        # Zero out the top 2% values in the original matrix
        for idx in zip(*top_indices_2d):
            core[idx] = 0

        # The `outliers_table` now contains the extracted values and their original indices
        # Each row in `outliers_table` has the format [value, row_index, col_index]

        # Displaying the first few rows of the outliers table
        print("outliers", outliers_table[:5])

        # Check if the values are zeroed out in the original matrix by printing some part of the matrix
        print(core[top_indices_2d[0][0:5], top_indices_2d[1][0:5]])
        return outliers_table, core

    #old_core = core.copy()

    for perc in [0.02]:#[0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04]:
        #core = core2.copy()
        outliers, core = extract_outliers(core2.copy(), perc)

        outCore = np.zeros(core.shape)
        for value, row_idx, col_idx in outliers:
            outCore[int(row_idx), int(col_idx)] = value

#        print(f"{perc*100:.2g}%", cossim(control, v @ outCore.T), cossim(control, v @ core.T))

        '''bucketize'''


        # Process each row
        sorted_buckets = []
        for row in core:
            # Reshape the row into buckets of size 8
            reshaped_row = row.reshape(-1, 8)
            
            # Sort each bucket by absolute values in descending order
            sorted_row = []
            bucket_id = 0
            for bucket in reshaped_row:
                # Get indices and values sorted by absolute values in descending order
                indices_sorted = np.argsort(-np.abs(bucket))
                sorted_bucket = [(bucket[i], i + bucket_id*8) for i in indices_sorted]
                bucket_id += 1
                
                # Append sorted tuples to the sorted_row list
                sorted_row.append(sorted_bucket)
            
            # Append the processed row to the sorted_buckets list
            sorted_buckets.append(sorted_row)

        # Now `sorted_buckets` contains the sorted buckets for each row
        # Each element in `sorted_buckets` is a list of buckets, where each bucket is a list of tuples (value, original_index)

        # Example output of the first bucket of the first row
        print(sorted_buckets[0][0])
        print(sorted_buckets[0][1])
        print(sorted_buckets[0][2])



        '''transpose'''


        # Assuming sorted_buckets is already defined as described in previous steps
        num_rows = core.shape[0]
        num_cols = core.shape[1] // 8  # Columns in the output are as many as the buckets
        num_elements_per_bucket = 8  # Each bucket has 8 elements

        # Initialize an output list to hold the transformed data rows with value and original position
        output_rows = []

        # Since each bucket has 8 elements, and you have num_cols buckets per original row
        # Create num_elements_per_bucket new rows for each original row
        for row_index, row in enumerate(sorted_buckets):
            # Temporary storage for the new rows created from this original row
            new_rows = [[] for _ in range(num_elements_per_bucket)]

            # Iterate over each bucket and each element in the bucket
            for bucket in row:
                for element_index, (value, original_pos) in enumerate(bucket):
                    # Append (value, original_col_index) where original_col_index is recalculated from the bucket
                    original_col_index = original_pos
                    new_rows[element_index].append((value, original_col_index))

            # Append these new rows to the output_rows
            output_rows.extend(new_rows)

        # Now, output_rows contains all the new rows formatted as [(value, original_index), (value, original_index), ...]
        # If you need this in a more structured array format or need to separate values and indices, further processing can be done

        # For demonstration, let's print the first few elements of the first few new rows
        for i in range(min(5, len(output_rows))):
            print("New row", i, ":", output_rows[i][:5])  # Print first few elements of each new row



        # Calculate statistics for each row
        stats = []

        for row in output_rows:
            # Extract the values from each tuple (ignoring the original positions for this calculation)
            values = [item[0] for item in row]

            # Calculate the absolute values
            abs_values = np.abs(values)

            # Calculate mean, min, and max of absolute values
            avg_abs = np.mean(abs_values)
            min_abs = np.min(abs_values)
            max_abs = np.max(abs_values)

            # Store the statistics
            stats.append((avg_abs, min_abs, max_abs))

        # Print statistics for the first few rows for demonstration
        for i in range(min(5, len(stats))):
            print(f"Row {i+1} - Avg: {stats[i][0]:.4f}, Min: {stats[i][1]:.4f}, Max: {stats[i][2]:.4f}")


        ''' mul for testing '''
        '''
        # Assuming V is your input vector of length matching the number of rows in 'core'
        V =v

        # Initialize output vector
        output_vector = np.zeros(core.shape[1])  # Length equal to the number of columns in the original matrix

        # Iterate through each row in V
        for inputRowID, scalar in enumerate(V):
            # Iterate over each bucket within the row (0 to 7)
            for n in range(8):
                # Get the specific row from output_rows
                row = output_rows[inputRowID * 8 + n]
                
                # Process each element in the row
                for value, original_index in row:
                    #if value == 0:
                    #    print('jest zero', original_index)
                    # Multiply the input scalar with the value and add to the corresponding index in the output vector
                    output_vector[original_index] += scalar * value

        print(output_vector)

        '''
        # Assuming V is your input vector of length matching the number of rows in 'core'
        V =v

        # Initialize output vector
        output_vector2 = np.zeros(core.shape[1])  # Length equal to the number of columns in the original matrix

        # Iterate through each row in V
        for inputRowID, scalar in enumerate(V):
            # Iterate over each bucket within the row (0 to 7)
            for n in range(8):
                # Get the specific row from output_rows
                row = output_rows[inputRowID * 8 + n]
                avg_abs_value = stats[inputRowID * 8 + n][0]  # Accessing the average absolute value

                # Process each element in the row
                for value, original_index in row:
                    contribution = scalar * np.sign(value) * avg_abs_value
                    # Multiply the input scalar with the value and add to the corresponding index in the output vector
                    output_vector2[original_index] += contribution
#                pdb.set_trace()

        print(output_vector2)

#        print(f"{perc*100:.2g}%", cossim(v @ core2, (v @ outCore) + (output_vector2)))
        
        '''
        extract probes
        prepare for serialization
        '''
        diagonal_vector = np.diag(core)
        probes = v * diagonal_vector


        bucket_stats = np.array(stats, dtype=np.float32)[:,:2]
        bucket_stats[:, 1] = bucket_stats[:,0] # Swift expects a structure of [(Float32=avg_abs, Float32=avg_abs)]

        out_tensors = {
            "probes": diagonal_vector,
            "bucket.stats": bucket_stats.view(), # hacking around to get [(Float16, Float16, Float16, Float16)] in swift that gets cast later back to Float32
            "buckets": rearrange8(np.array(output_rows)),
            "outliers": rearrangeOutliers(np.array(outliers))
        }
        out_tensors["buckets"][:,:,0] = np.sign(out_tensors["buckets"][:,:,0])
        avg_abs_values = [stat[0] for stat in stats]
        # Create a new array by duplicating each average absolute value and formatting as required
        out_tensors["stats"] = rearrange8(np.array([[x, x] for x in avg_abs_values], dtype=np.float32))


        # convert output_rows into a half-byte representation
        # Process each bucket
        processed_buckets = []

        for row in output_rows:
            processed_row = []
            converted_items = [(8 if value < 0 else 0, int(index) % 8) for value, index in row]
            
            # Step 2: Bit-pack each item
            bit_packed_items = [(item[0] + item[1]) for item in converted_items]

            # Step 3: Merge every two half-bytes into one byte
            bucket_bytes = []
            for i in range(0, len(bit_packed_items), 2):
                # Ensure there's a pair to process, otherwise, just take the single half-byte left
                if i + 1 < len(bit_packed_items):
                    byte = (bit_packed_items[i] << 4) + bit_packed_items[i + 1]
                else:
                    byte = (bit_packed_items[i] << 4)  # Handle odd number of items by shifting the last one
                bucket_bytes.append(byte)
                
            processed_buckets.append(bucket_bytes)

    #    print(processed_buckets[0][0])  # Print the first processed bucket of the first row

        '''
            merge words, and bit-convert into float16, because that's what the Swift implementation
            is ready for
        '''

        processed_bucket_words = []

        for row in processed_buckets:
            processed_row_words = []
            bucket_words = []
            for i in range(0, len(row), 2):
                # Make sure there is a pair to process
                assert i + 1 < len(row)
                word = (row[i] << 8) + row[i + 1]
                bucket_words.append(word)

            processed_bucket_words.append(bucket_words)

        # Now convert this list of words into a NumPy array of float16
        # Flatten the list of words to make it suitable for a NumPy array conversion
        flat_words = [word for row in processed_bucket_words for word in row]

        # Convert to a NumPy array
        np_words = np.array(flat_words, dtype=np.uint16)  # First as uint16 to preserve the bit structure

        # Cast to float16
        np_float16_words = np_words.view(np.float16)

#        pdb.set_trace()
        np_float16_words = np_float16_words.reshape((len(processed_buckets), -1))

        out_tensors["buckets"] = np_float16_words

#        pdb.set_trace()

        return out_tensors

    '''

    other
    '''


    '''
    bucketMul implementation for testing

    abs_probes = np.abs(probes)
    sorted_indices = np.argsort(-abs_probes)
    sorted_probes = probes[sorted_indices]

    effort = 0.20
    for effort in [0.05, 0.10, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40]:
        cutoff = abs(sorted_probes[int(4096*effort)])

        output_vector3 = np.zeros(core.shape[1])  # Length equal to the number of columns in the original matrix

        count_all = 0
        count_done = 0
        # Iterate through each row in V
        for inputRowID, scalar in enumerate(V):
            # Iterate over each bucket within the row (0 to 7)
            for n in range(8):
                # Get the specific row from output_rows
                row = output_rows[inputRowID * 8 + n]
                avg_abs_value = stats[inputRowID * 8 + n][0]  # Accessing the average absolute value
                count_all += 1
                if avg_abs_value * abs(scalar) < cutoff:
                    continue
                count_done += 1
                # Process each element in the row
                for value, original_index in row:
                    contribution = scalar * np.sign(value) * avg_abs_value
                    # Multiply the input scalar with the value and add to the corresponding index in the output vector
                    output_vector3[original_index] += contribution

        print(count_done, count_all, count_done/count_all)
        print(f"{perc*100:.2g}% {effort*100:.2g}%", cossim(v @ core2, (v @ outCore) + (output_vector3)))
    '''

#pdb.set_trace()


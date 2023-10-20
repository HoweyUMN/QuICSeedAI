import pandas as pd
import numpy as np

# Test
class preprocess:
    """Compiles data preprocessing steps into a cohesive, modular set of functions for RT-Quic datasets
    
    Author: Kyle Howey
    Version: October 2023"""

    def __init__(self, time_increment):
        self.dataframes = {} # dictionary of dataframes 
        self.time_increment = time_increment # time between measurement for each sample
        self.combined_dataframe = np.null
        self.filenames = []
        self.timepoitns_h = []
        self.samples = {}

    def import_data(self, paths, filenames = []):
        """Takes a list of CSV filenames and paths and imports them into a pandas dataframe with the keys as 
        the given name or paths if no names are provided. Performs error checking and puts data into a uniform
        format.
        
        
        Returns dictionary of imported dataframes."""

        # Make sure user input is in a usable format
        if len(filenames) == 0:
            filenames = paths
        elif len(filenames) != len(paths):
            raise Exception("Filename list must be same length as path or be left unspecified") 

        self.filenames = filenames
        # Import the data from the CSVs
        data = {}
        for i,filename in enumerate(filenames):
            data[filename] = pd.read_csv(paths[i], header = None)

        data = self._correct_filenames(data)

        # Ensuring data is uniform (ie data begins in the same position)
        for filename, df in data.items():
            if(not np.all(np.array(df.iloc[:4,0].tolist()) == np.array(data[filenames[0]].iloc[:4,0].tolist()))):
                raise Exception("Data in file {} is inconsistent with other CSV files".format(filename))
            
        self.data = self._preprocess_data(filenames, data)
        return self.data
    
    def _correct_filenames(self, data):
        """Private function, adds filename to replicate ID for later reference"""

        # Append filename to replicate ID in each file
        def append_filename(x, suffix):
            if pd.isnull(x):
                return x
            else:
                return x + '|' + suffix

        for filename,df in data.items():
            # Print replicate IDs before
            print("Before: ", end="")
            print(df.iloc[2,:10].tolist())
            # Append filename to replicate ID
            df.iloc[2,:] = df.iloc[2,:].apply(lambda x: append_filename(x, suffix=filename.split('.')[0]))
            # Print replicate IDs after
            print("After: ", end="")
            print(df.iloc[2,:10].tolist())
        
        return data
    
    def _preprocess_data(self, filenames, data):
        """Private function, puts imported data into a uniform format for training models"""

        data = self._enforce_uniform_start(filenames, data)
        data = self._enforce_uniform_end(filenames, data)

        return data

    def _enforce_uniform_start(self, filenames, data):
        """Private function, enforces all dataframes have the same start time at 0"""

        # Extract minimum time for each file
        initial_times = []
        for filename,df in data.items():
            print(filename + ": ")
            print(df.iloc[:5,0].tolist())
            print("initial_time=%s\n" % (str(df.iloc[4,0])))
            initial_times.append(float(df.iloc[4,0]))

        ## Find files that don't start at t=0
        files_to_correct = []
        for i, initial_time in enumerate(initial_times):
            if initial_time != 0:
                files_to_correct.append(i)

        # Simple function to make next step less complex
        def stringify(t):
            if int(t) == float(t):
                return str(int(t))
            else:
                return str(t)

        # Pad files to start at t=0
        for i in files_to_correct:
            filename = filenames[i]
            print("Padding with empty strings to time 0 in %s" % (filename,))
            initial_time = initial_times[i]
            df = data[filename]
            # Compute times to fill in
            times = [stringify(i) for i in np.arange(0,initial_time, self.time_increment)]
            # Generate block of new rows to fill in
            data_to_insert = np.full([len(times), df.shape[1]-1], '', dtype='<U16')
            data_to_insert[:,0] = times
            # Insert block of new rows
            df = pd.concat([df.iloc[:4,:], pd.DataFrame(data_to_insert), df.iloc[4:,:]]).reset_index(drop=True)
            data[filename] = df

        return data
    
    def _enforce_uniform_end(self, filenames, data):
        """Private function, ensures all dataframes have the same end time, adding padding to correct files without this"""

        # Find the length of each file
        lengths = []
        for filename in filenames:
            lengths.append(len(data[filename]))

        min_length = min(lengths)
        # VALIDATION: Left column equal up to minimum length
        for filename,df in data.items():
            if(not np.all(np.array(df.iloc[:min_length,0].tolist()) == np.array(data[filenames[0]].iloc[:min_length,0].tolist()))):
                raise Exception("Data in file {} is inconsistent with other CSV files".format(filename))
        
        # Extract max time for each file
        final_times = []
        for i in range(len(filenames)):
            filename = filenames[i]
            df = data[filename]

            print(filename + ": \n...", end="")
            print(df.iloc[len(df)-4:len(df),0].tolist())
            print("final_time=%s\n" % (str(df.iloc[len(df)-1,0])))
            final_times.append(float(df.iloc[len(df)-1,0]))

        # Determine latest final time
        latest_time = max(final_times)

        # Enforce each file goes to final time
        files_to_correct = []
        for i, final_time in enumerate(final_times):
            if final_time != latest_time:
                files_to_correct.append(i)

        def stringify(t):
            if int(t) == float(t):
                return str(int(t))
            else:
                return str(t)

        for i in files_to_correct:
            filename = filenames[i]
            print("Padding with empty strings to %s hours in %s" % (str(latest_time), filename))
            final_time = final_times[i]
            df = data[filename]
            # Compute times to fill in
            times = [stringify(i) for i in np.arange(final_time + self.time_increment, latest_time + self.time_increment, self.time_increment)]
            # Generate block of new rows to fill in
            data_to_insert = np.full([len(times), df.shape[1]-1], '', dtype='<U16')
            data_to_insert[:,0] = times
            # Insert block of new rows
            df = pd.concat([df.iloc[:len(df),:], pd.DataFrame(data_to_insert)]).reset_index(drop=True)
            data[filename] = df

        ## Double Checking Work
        for filename,df in data.items():
            if(not np.all(np.array(df.iloc[:,0].tolist()) == np.array(data[filenames[0]].iloc[:,0].tolist()))):
                raise Exception("Data in file {} is inconsistent with other CSV files".format(filename))

        for filename,df in data.items():
            print(filename + ": \t", end='')
            print(df.shape)

        return data

    def combine_dataframes(self, filenames):
        """Combines previously saved dataframes into one massive one
        which is stored in an attribute
        
        Returns all dataframes combined into one"""
        ### Combine Data
        dfs_to_combine = []
        for filename,df in self.dataframes.items():
            if filename == filenames[0]:
                dfs_to_combine.append(df)
            else:
                dfs_to_combine.append(df.iloc[:, 1:])  # Remove first column from subsequent dfs when combining

        self.combined_dataframe = pd.concat(dfs_to_combine, axis=1, ignore_index=True).reset_index(drop=True)

        print("Shape of combined data: ", end="")
        print(self.combined_dataframe.shape)

        return self.combined_dataframe
    
    def data2numpy(self):
        """Converts all stored dataframes into a single numpy
        array and stores metadata in separate lists in attributes
        for reference during training.
        
        Returns list of: [numpy array of samples, numpy array of labels]"""

        if(self.combined_dataframe == np.null):
            self.combine_dataframes(self.filenames)
        # Collect timepoints from first column
        self.timepoints_h = self.combined_dataframe.iloc[4:, 0].astype(float).tolist()

        # Pull data instances from each column
        X = []
        y = []
        sample = []
        sample_id = []
        well_name = []
        col_idx = []
        for j in range(1, self.combined_dataframe.shape[1]):
            # Parse out column
            column = self.combined_dataframe.iloc[:, j].tolist()
            # Extract label
            if column[0] == 'Pos':
                y.append(1)
            elif column[0] == 'Neg':
                y.append(0)
            else:
                raise ValueError("Label was not 'Pos' or 'Neg'")
            # Extract sample note
            sample.append(column[1])
            # Extract sample identifier
            sample_id.append(column[2])
            # Extract well name
            well_name.append(column[3])
            # Extract RT-QuIC curve
            curve = column[4:]
            curve = list(pd.to_numeric(curve, errors='coerce'))
            self.X.append(curve)
            # Extract column index
            col_idx.append(j)

        # Convert data to numpy
        X = np.array(X)
        y = np.array(y)

        self._check_array_integrity(X, y, sample, sample_id, well_name, col_idx)

        return [X, y]
    
    def _check_array_integrity(self, X, y, sample, sample_id, well_name, col_idx):
        """Private function to ensure values are valid before
        assignment to attributes
        
        If all integrity checks pass, saves values to attributes"""

        if(len(X) != len(y) or len(y) != len(sample)):
            raise ValueError("Arrays identifying samples are not equal lengths!")
        if(len(y) != len(sample_id)):
            raise ValueError("Sample identifiers and sample lables are different length arrays!")
        if(len(y) != len(well_name)):
            raise ValueError("Sample labels and well name list are different length arrays!")
        if(len(y) != len(col_idx)):
            raise ValueError("Sample labels and column identifiers are different length arrays!")
        
        self._add_data_to_attributes(self, X, y, sample, sample_id, well_name, col_idx)
        
    def _add_data_to_attributes(self, X, y, sample, sample_id, well_name, col_idx):
        # Create dictionary of samples
        samples = {}

        for i in range(len(y)):
            key = sample_id[i]
            if key not in samples:
                samples[key] = {}
                samples[key]['X'] = []
                samples[key]['y_list'] = []
                samples[key]['sample_list'] = []
                samples[key]['well_name_list'] = []
                samples[key]['col_idx_list'] = []
            
            samples[key]['X'].append(X[i].copy())
            samples[key]['y_list'].append(y[i].copy())
            samples[key]['sample_list'].append(sample[i])
            samples[key]['well_name_list'].append(well_name[i])
            samples[key]['col_idx_list'].append(col_idx[i])

        # Checking all data is uniform
        keys_to_delete = set()
        for key in samples.keys():
            # Add count
            samples[key]['well_count'] = len(samples[key]['y_list'])

                    # Check all labels are the same
            if sum(samples[key]['ylist'])!=0 and sum(samples[key]['ylist'])!=len(samples[key]['ylist']):
                print(samples[key]['ylist'])
                keys_to_delete.add(key)
            else:
                samples[key]['y'] = samples[key]['ylist'][0]
            
            # Convert grouped sample data to array
            samples[key]['X'] = np.array(samples[key]['X'])

            # Check if all samples are the same
            sample_list = np.array(sample_list)
            if not np.all(sample_list == sample_list[0]):
                print(sample_list)
                keys_to_delete.add(key)
            else:
                samples[key]['sample'] = sample_list[0]

        # Delete samples that failed integrity check
        for key in keys_to_delete:
            samples.pop(key)

        print("Deleted %i sample(s) due to label inconsistency" % (len(keys_to_delete),))

        self.samples = samples

        
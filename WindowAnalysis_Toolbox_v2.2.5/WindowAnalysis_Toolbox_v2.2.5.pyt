import os
import arcpy
from arcpy import env
from arcpy.sa import *
import pandas as pd
from pathlib import Path
import numpy as np


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Toolbox"
        self.alias = "toolbox"

        # List of tool classes associated with this toolbox
        self.tools = [Tool1, Tool2, Tool4, Tool5, Tool6, Tool7, Tool8, Tool9, Tool10, Tool11, Tool12]


class Tool1(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Extract Exposures"
        self.description = "The Extract Exposures tool reads the exposure for shapefile points across multiple rasters. Exposure values will be added to the original input .shp attribute table." 
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        params = [
            arcpy.Parameter(displayName="Environmental Rasters Location (.gdb)",
                            name="environ",
                            datatype="DEWorkspace",
                            parameterType="Required",
                            direction="Input"),
            
            arcpy.Parameter(displayName="Exposure Points Folder",
                            name="caseFolder",
                            datatype="DEFolder",
                            parameterType="Required",
                            direction="Input"),
                                        
            arcpy.Parameter(displayName="Raster Name string segment that contains the year",
                            name="seg_num",
                            datatype="GPString",
                            parameterType="Required",
                            direction="Input"),   

            arcpy.Parameter(displayName="Raster Name string segment that PRECEEDS year",
                            name="pre_year",
                            datatype="GPString",
                            parameterType="Required",
                            direction="Input"),    

            arcpy.Parameter(displayName="Raster Name string segment that FOLLOWS year",
                            name="post_year",
                            datatype="GPString",
                            parameterType="Required",
                            direction="Input"),                   
        ]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        environ = parameters[0].valueAsText
        caseFolder = parameters[1].valueAsText
        seg_num = int(parameters[2].valueAsText)  # Convert seg_num to integer
        pre_year = parameters[3].valueAsText
        post_year = parameters[4].valueAsText

        # Set the current workspace (your gdb of environmental rasters)
        arcpy.env.workspace = environ  

        # Read all raster layers in the current workspace
        rasters = arcpy.ListRasters("*", "all")

        # Filter rasters based on pre-year and post-year segments (ignoring 'IsNull')
        filtered_rasters = [r for r in rasters if pre_year in r and post_year in r]

        # Keep track of processed years to avoid duplication
        processed_years = set()

        # List to hold raster name and field name pairs
        inRasterList = []

        # Iterate over each filtered raster and extract the year to create field names
        for raster in filtered_rasters:
            try:
                # Split the raster name by underscores and extract the year segment
                year = raster.split("_")[seg_num].strip()  # Ensure we only get the year and clean it
            except IndexError:
                # Handle the case where the seg_num index is out of bounds
                arcpy.AddError(f"Error: Raster name '{raster}' does not have enough segments to extract the year at index {seg_num}")
                continue

            # Skip if this year has already been processed
            if year in processed_years:
                arcpy.AddMessage(f"Skipping duplicate year: {year}")
                continue

            # Add the year to the set of processed years
            processed_years.add(year)

            # Construct the full raster name and the field name using the extracted year
            rasterName = pre_year + year + post_year
            fieldName = "pb_" + year  # The field name will be 'pb_YYYY' based on the year

            # Add the raster name and field name to the list for extraction
            inRasterList.append([rasterName, fieldName])
            arcpy.AddMessage(f"Processed raster: {rasterName} with field name: {fieldName}")

        # Process each shapefile in the case folder
        for file in os.listdir(caseFolder):
            if file.endswith(".shp"):
                # Create the full path to the shapefile
                CasesSet = os.path.join(caseFolder, file)
                arcpy.AddMessage(f"Processing shapefile: {CasesSet}")
                
                # Perform the extraction of raster values to points
                try:
                    arcpy.sa.ExtractMultiValuesToPoints(CasesSet, inRasterList, "NONE")
                    arcpy.AddMessage(f"Extraction completed for shapefile: {CasesSet}")
                except Exception as e:
                    arcpy.AddError(f"Error extracting values for shapefile '{CasesSet}': {str(e)}")

        return

    def postExecute(self, parameters):
        """This method takes place after outputs are processed and
        added to the display."""
        return


import os
import arcpy
import pandas as pd


class Tool2(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Clean Exposure Data and Merge with Population Density"
        self.description = ("This tool adds an exposure value column for each case/control, extracts the correct year of exposure data for each row in the data table, "
                            "and merges population density data based on whether the exposure data is controls or mortality. "
                            "If multiple rows exist for the same study_ID and year, it averages the exposure and population density values. "
                            "The tool also maintains the 'index_year' column.")
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions."""
        params = [
            arcpy.Parameter(displayName="Parent Folder Containing Subfolders",
                            name="parent_folder",
                            datatype="DEFolder",
                            parameterType="Required",
                            direction="Input"),

            arcpy.Parameter(displayName="Control Population Density Shapefile",
                            name="popd_controls",
                            datatype="DEShapeFile",
                            parameterType="Required",
                            direction="Input"),

            arcpy.Parameter(displayName="Mortality Population Density Shapefile",
                            name="popd_mortality",
                            datatype="DEShapeFile",
                            parameterType="Required",
                            direction="Input"),

            arcpy.Parameter(displayName="First Year of Environmental Data",
                            name="startyr",
                            datatype="GPString",
                            parameterType="Required",
                            direction="Input"),

            arcpy.Parameter(displayName="Last Year of Environmental Data",
                            name="endyr",
                            datatype="GPString",
                            parameterType="Required",
                            direction="Input"),

            arcpy.Parameter(displayName="Output folder for merged CSV data",
                            name="outpath_merged",
                            datatype="DEFolder",
                            parameterType="Required",
                            direction="Output"),
        ]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal validation is performed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool parameter."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        parent_folder = parameters[0].valueAsText
        popd_controls = parameters[1].valueAsText
        popd_mortality = parameters[2].valueAsText
        startyr = int(parameters[3].valueAsText)
        endyr = int(parameters[4].valueAsText)
        outpath_merged = parameters[5].valueAsText

        # Check if the output folder exists, if not, create it
        if not os.path.exists(outpath_merged):
            os.makedirs(outpath_merged)
        else:
            arcpy.AddMessage(f"Output folder already exists: {outpath_merged}")

        # Define the folder names (B folders)
        b_folders = ['Union_M1', 'Union_M2', 'Intersect_M1', 'Intersect_M2']

        # Iterate over each B folder (Union_M1, Union_M2, etc.)
        for b_folder in b_folders:
            b_folder_path = os.path.join(parent_folder, b_folder)

            # Check if the B folder exists
            if not os.path.exists(b_folder_path):
                arcpy.AddError(f"Folder not found: {b_folder_path}")
                continue

            # Iterate over the subfolders (C folders) inside each B folder
            for subfolder_name in os.listdir(b_folder_path):
                c_folder_path = os.path.join(b_folder_path, subfolder_name)

                # Check if the subfolder is a valid directory
                if not os.path.isdir(c_folder_path):
                    continue  # Skip non-folder items like .DS_Store

                # Find all shapefiles in this subfolder
                for file in os.listdir(c_folder_path):
                    if file.endswith(".shp"):  # Only process .shp files
                        shp_file_path = os.path.join(c_folder_path, file)
                        arcpy.AddMessage(f"Processing shapefile: {shp_file_path}")

                        # Determine if the shapefile is for controls or mortality
                        if "controls" in file.lower():
                            popd_shp = popd_controls
                        elif "mortality" in file.lower():
                            popd_shp = popd_mortality
                        else:
                            arcpy.AddMessage(f"Skipping shapefile (unknown type): {file}")
                            continue

                        # Check if "expsr_val" already exists
                        field_names = [field.name for field in arcpy.ListFields(shp_file_path)]
                        if "expsr_val" in field_names:
                            arcpy.AddMessage(f"'expsr_val' column already exists in {file}. Skipping exposure value updates.")
                        else:
                            # Add the field and process if "expsr_val" does not exist
                            arcpy.AddField_management(shp_file_path, "expsr_val", "FLOAT")

                            # Define the fields to be used in the cursor
                            fields = ['study_ID', 'year', 'expsr_val']

                            # Append all the year-based exposure fields for the range specified
                            for year in range(startyr, endyr + 1):
                                fields.append('pb_' + str(year))  # Exposure fields based on the year

                            # Process the shapefile and update the exposure values
                            with arcpy.da.UpdateCursor(shp_file_path, fields) as cursor:
                                for row in cursor:
                                    year = row[1]  # Extract the year from the shapefile row
                                    if year >= startyr:
                                        exposure_field = 'pb_' + str(year)  # Exposure field corresponding to the year
                                        exposure_field_index = fields.index(exposure_field)
                                        exposure = row[exposure_field_index]  # Get the exposure value from the correct year
                                        row[2] = exposure  # Update the "expsr_val" field with the correct exposure value
                                        cursor.updateRow(row)
                                    elif year < startyr:
                                        row[2] = -9999  # If year is outside the scope, set exposure value to -9999 (N/A)
                                        cursor.updateRow(row)

                        # Merge with the population density shapefile
                        arcpy.AddMessage(f"Merging exposure data with population density data from: {popd_shp}")

                        # Create dataframes from the shapefiles
                        df_exposure = pd.DataFrame(arcpy.da.FeatureClassToNumPyArray(shp_file_path, ['study_ID', 'year', 'index_year', 'expsr_val']))
                        df_popd = pd.DataFrame(arcpy.da.FeatureClassToNumPyArray(popd_shp, ['study_ID', 'year', 'pd_7x7']))

                        # Replace missing exposure values (-9999) with 0 for clarity... meaning NO exposure data is treated as 0
                        df_exposure = df_exposure.replace(-9999, 0)

                        # Average exposure data for same study_ID and year
                        df_exposure_grouped = df_exposure.groupby(['study_ID', 'year', 'index_year'])['expsr_val'].mean().reset_index()

                        # Average population density data for same study_ID and year
                        df_popd_grouped = df_popd.groupby(['study_ID', 'year'])['pd_7x7'].mean().reset_index()

                        # Merge exposure data with population density
                        df_merged = pd.merge(df_exposure_grouped, df_popd_grouped, on=['study_ID', 'year'], how='outer')

                        # Create the output CSV path
                        output_csv_name = f"PopCleaned_{subfolder_name}_{b_folder}.csv"
                        output_csv_path = os.path.join(outpath_merged, output_csv_name)

                        # Save the merged data to CSV
                        df_merged.to_csv(output_csv_path, index=False)
                        arcpy.AddMessage(f"Finished processing and saved merged data to: {output_csv_path}")

        return

    def postExecute(self, parameters):
        """This method takes place after outputs are processed and added to the display."""
        return

import arcpy
import pandas as pd

class Tool11(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Add Exposure Data Medias"
        self.description = "This tool adds two versions of exposure data (M1 and M2), summing the exposure values for matching study_ID and year, and retaining the population density exposures from the first CSV. Tool can used for any pairs of csvs where you only want to sum the exposure readings but maintain a single (not summed) population density"
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        params = [
            arcpy.Parameter(displayName="First exposure CSV",
                            name="exposure_csv_1",
                            datatype="DEFile",
                            parameterType="Required",
                            direction="Input"),

            arcpy.Parameter(displayName="Second exposure CSV",
                            name="exposure_csv_2",
                            datatype="DEFile",
                            parameterType="Required",
                            direction="Input"),
                            
            arcpy.Parameter(displayName="Output CSV",
                            name="output_csv",
                            datatype="DEFile",
                            parameterType="Required",
                            direction="Output"),
        ]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        import pandas as pd

        exposure_csv_1 = parameters[0].valueAsText
        exposure_csv_2 = parameters[1].valueAsText
        output_csv = parameters[2].valueAsText

        # Read the input CSV files into pandas DataFrames
        df1 = pd.read_csv(exposure_csv_1)
        df2 = pd.read_csv(exposure_csv_2)

        # Ensure index_year is preserved during merge
        df1 = df1[['study_ID', 'year', 'index_year', 'expsr_val', 'pd_7x7']]
        df2 = df2[['study_ID', 'year', 'index_year', 'expsr_val']]

        # Merge the DataFrames on study_ID and year, summing exposure values
        merged_df = pd.merge(df1, df2, on=['study_ID', 'year', 'index_year'], suffixes=('_1', '_2'))
        
        # Calculate the summed exposure values
        merged_df['expsr_val'] = merged_df['expsr_val_1'] + merged_df['expsr_val_2']
        
        # Ensure index_year values are integers
        merged_df['index_year'] = merged_df['index_year'].astype(int)
        merged_df['expsr_val'] = merged_df['expsr_val'].astype(int)

        # Select relevant columns to output
        output_df = merged_df[['study_ID', 'year', 'index_year', 'expsr_val', 'pd_7x7']]

        # Write the output DataFrame to a CSV file
        output_df.to_csv(output_csv, index=False)

        arcpy.AddMessage(f"Output CSV created at {output_csv}")


import os
import arcpy
import pandas as pd

class Tool4(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Summary Exposure Calculation"
        self.description = ("This tool calculates Median, Sum, Max, and Min summary statistics, processes files based on their version "
                            "(Union/Intersect, Controls/Mortality, M1/M2/M1M2), and outputs them into structured folders.")
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        params = [
            arcpy.Parameter(displayName="Parent Folder Containing All CSV Files",
                            name="parent_folder",
                            datatype="DEFolder",
                            parameterType="Required",
                            direction="Input"),

            arcpy.Parameter(displayName="Parent Output Folder",
                            name="parent_output_folder",
                            datatype="DEFolder",
                            parameterType="Required",
                            direction="Output"),

            arcpy.Parameter(displayName="Number of Timeframe Years",
                            name="timeframes_num",
                            datatype="GPLong",
                            parameterType="Required",
                            direction="Input"),
        ]
        return params

    def execute(self, parameters, messages):
        """The source code of the tool."""
        parent_folder = parameters[0].valueAsText
        parent_output_folder = parameters[1].valueAsText
        timeframes_num = int(parameters[2].valueAsText)

        # Define summary types
        summary_types = ["median", "accum", "max", "min"]

        # Loop through all files in the parent folder
        for root, dirs, files in os.walk(parent_folder):
            for file in files:
                if file.endswith(".csv"):
                    # Determine the file type by checking file name
                    file_path = os.path.join(root, file)
                    file_name = file.lower()

                    # Determine the version type (Union/Intersect, M1/M2, Controls/Mortality)
                    if "union" in file_name:
                        version_type = "Union"
                    elif "intersect" in file_name:
                        version_type = "Intersect"
                    else:
                        arcpy.AddMessage(f"Skipping unknown file version: {file}")
                        continue

                    if "m1m2" in file_name:
                        subversion = "M1M2"
                    elif "m1" in file_name:
                        subversion = "M1"
                    elif "m2" in file_name:
                        subversion = "M2"
                    else:
                        arcpy.AddMessage(f"Skipping unknown subversion: {file}")
                        continue

                    if "controls" in file_name:
                        control_type = "Controls"
                    elif "mortality" in file_name:
                        control_type = "Mortality"
                    else:
                        arcpy.AddMessage(f"Skipping unknown control type: {file}")
                        continue

                    # Create output folder path based on version type and control type
                    output_folder_path = os.path.join(parent_output_folder, f"{version_type}_{subversion}", control_type)
                    if not os.path.exists(output_folder_path):
                        os.makedirs(output_folder_path)

                    # Read the data
                    df_merged = pd.read_csv(file_path)

                    # Group data by study_ID and index_year
                    diagnosis_grouped = df_merged.groupby('study_ID')['index_year'].median().reset_index()

                    # Calculate and save each summary statistic separately
                    for calc_type in summary_types:
                        # Define output file path
                        outpath = os.path.join(output_folder_path, f"{file.replace('.csv', '')}_{calc_type}.csv")

                        # Summary calculation type
                        if calc_type == "median":
                            arcpy.AddMessage("Currently summarizing: Median")
                            diagnosis_grouped_stat = self.calculate_stat(diagnosis_grouped, df_merged, timeframes_num, calc_type, "median")
                            relevant_columns = [col for col in diagnosis_grouped_stat.columns if 'median_expsr' in col or 'accum_pd' in col]
                        elif calc_type == "accum":
                            arcpy.AddMessage("Currently summarizing: Accum")
                            diagnosis_grouped_stat = self.calculate_stat(diagnosis_grouped, df_merged, timeframes_num, calc_type, "sum")
                            relevant_columns = [col for col in diagnosis_grouped_stat.columns if 'accum_expsr' in col or 'accum_pd' in col]
                        elif calc_type == "max":
                            arcpy.AddMessage("Currently summarizing: Max")
                            diagnosis_grouped_stat = self.calculate_stat(diagnosis_grouped, df_merged, timeframes_num, calc_type, "max")
                            relevant_columns = [col for col in diagnosis_grouped_stat.columns if 'max_expsr' in col or 'accum_pd' in col]
                        elif calc_type == "min":
                            arcpy.AddMessage("Currently summarizing: Min")
                            diagnosis_grouped_stat = self.calculate_stat(diagnosis_grouped, df_merged, timeframes_num, calc_type, "min")
                            relevant_columns = [col for col in diagnosis_grouped_stat.columns if 'min_expsr' in col or 'accum_pd' in col]

                        # Include 'study_ID' and 'index_year' in the columns to save
                        relevant_columns = ['study_ID', 'index_year'] + relevant_columns

                        # Save each summary statistic to its own CSV
                        diagnosis_grouped_stat[relevant_columns].to_csv(outpath, index=False, header=True)
                        arcpy.AddMessage(f"{calc_type.capitalize()}: {outpath} CSV created.")

    def calculate_stat(self, diagnosis_grouped, df_merged, timeframes_num, calc_type, stat_method):
        """Helper method to calculate statistics based on the type (median, accum, max, min)"""
        def find_stat(num_years, patient_all_expsr_df, patient_ID, diagnosis_yr, stat_method):
            earliest_yr = diagnosis_yr - num_years
            patient_expsr_subframe = patient_all_expsr_df[patient_all_expsr_df['study_ID'] == patient_ID]
            filteredbyyear = patient_expsr_subframe[patient_expsr_subframe['year'] >= earliest_yr]
            
            if stat_method == "median":
                return filteredbyyear['expsr_val'].median()
            elif stat_method == "sum":
                return filteredbyyear['expsr_val'].sum()
            elif stat_method == "max":
                return filteredbyyear['expsr_val'].max()
            elif stat_method == "min":
                return filteredbyyear['expsr_val'].min()

        # Apply the statistics calculation for exposure
        for i in range(timeframes_num + 1):
            new_col_name = f'{calc_type}_expsr_{i}_yrs'
            diagnosis_grouped[new_col_name] = diagnosis_grouped.apply(
                lambda x: find_stat(i, df_merged, x['study_ID'], x['index_year'], stat_method), axis=1)

        # Apply the accumulated population density (always sum) calculation
        for i in range(timeframes_num + 1):
            new_col_name_pd = f'accum_pd_7x7_{i}_yrs'
            diagnosis_grouped[new_col_name_pd] = diagnosis_grouped.apply(
                lambda x: self.find_yrs_prior_accum_pd(i, df_merged, x['study_ID'], x['index_year']), axis=1)

        return diagnosis_grouped

    def find_yrs_prior_accum_pd(self, num_years, patient_all_expsr_df, patient_ID, diagnosis_yr):
        """Helper method to calculate accumulated population density for timeframes"""
        earliest_yr = diagnosis_yr - num_years
        patient_expsr_subframe = patient_all_expsr_df[patient_all_expsr_df['study_ID'] == patient_ID]
        filteredbyyear = patient_expsr_subframe[patient_expsr_subframe['year'] >= earliest_yr]
        pd_expsr = filteredbyyear['pd_7x7'].sum()
        return pd_expsr


class Tool5(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Window Analysis and Export"
        self.description = "This tool takes case and control summary exposure csvs and runs a rolling window analysis on them. Creates both log and non-log transformed export files. Output should be a folder as there will be many csv files."
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        params = [
            arcpy.Parameter(displayName="Folder that contains case and control summary folders",
                            name="aaa_folder",
                            datatype="DEFolder",
                            parameterType="Required",
                            direction="Input"),

            arcpy.Parameter(displayName="Number of timeframe years",
                            name="timeframes_num2",
                            datatype="GPLong",
                            parameterType="Required",
                            direction="Input"),

            arcpy.Parameter(displayName="Attribute Data File (containing controls and cases)",
                            name="attrib_file",
                            datatype="DEFile",
                            parameterType="Required",
                            direction="Input"),

            arcpy.Parameter(displayName="Output folder for Window Analysis files",
                            name="output_folder",
                            datatype="DEFolder",
                            parameterType="Required",
                            direction="Input"),

        ]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed. This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""

        aaa_folder = parameters[0].valueAsText
        timeframes_num2 = parameters[1].valueAsText
        attrib_file = parameters[2].valueAsText
        output_folder = parameters[3].valueAsText

        import os
        import pandas as pd
        import numpy as np
        import glob

        control_folder = os.path.join(aaa_folder, "Controls")
        mortality_folder = os.path.join(aaa_folder, "Mortality")

        calc_types = ["median", "accum", "max", "min"]

        abbreviation = {
            "median": "median",
            "accum": "accum",
            "max": "max",
            "min": "min"
        }

        for calc_type in calc_types:
            arcpy.AddMessage(f"Processing {calc_type} files...")
            control_files = glob.glob(os.path.join(control_folder, f"*{calc_type}*.csv"))
            mortality_files = glob.glob(os.path.join(mortality_folder, f"*{calc_type}*.csv"))

            calc_abb = abbreviation.get(calc_type, calc_type)

            arcpy.AddMessage(f"Expected control files: {control_files}")
            arcpy.AddMessage(f"Expected mortality files: {mortality_files}")

            existing_control_files = [file_name for file_name in control_files if os.path.exists(
                os.path.join(control_folder, file_name.replace("\\", "/")))]
            existing_mortality_files = [file_name for file_name in mortality_files if os.path.exists(
                os.path.join(mortality_folder, file_name.replace("\\", "/")))]

            if existing_control_files:
                control_file2 = str(existing_control_files[0])
            else:
                arcpy.AddMessage(f"No {calc_type} control files found or there is an error.")

            if existing_mortality_files:
                mortality_file2 = str(existing_mortality_files[0])
            else:
                arcpy.AddMessage(f"No {calc_type} mortality files found or there is an error.")

            if os.path.join(control_folder, control_file2) in control_files and os.path.join(mortality_folder,
                                                                                              mortality_file2) in mortality_files:
                arcpy.AddMessage(f"There's a match! {control_file2} and {mortality_file2}")
                control_path = os.path.join(control_folder, control_file2)
                arcpy.AddMessage(f"Importing {control_path} into dataframe...")
                mortality_path = os.path.join(mortality_folder, mortality_file2)
                arcpy.AddMessage(f"Importing {mortality_path} into dataframe...")

                df_control = pd.read_csv(control_path)
                patient_median_exprs_df = pd.read_csv(mortality_path)

                # The rest of your existing code for window analysis goes here

                # region Function: sliding window
                def sliding_window(iterable, size, overlap=0):
                    """
                        >>> list(sliding_window([1, 2, 3, 4], size=2))
                        [(1, 2), (3, 4)]
                        >>> list(sliding_window([1, 2, 3], size=2, overlap=1))
                        [(1, 2), (2, 3)]
                        >>> list(sliding_window([1, 2, 3, 4, 5], size=3, overlap=1))
                        [(1, 2, 3), (3, 4, 5)]
                        >>> list(sliding_window([1, 2, 3, 4], size=3, overlap=1))
                        [(1, 2, 3), (3, 4)]
                        >>> list(sliding_window([1, 2, 3, 4], size=10, overlap=8))
                        [(1, 2, 3, 4)]
                    """
                    start = 0
                    end = size
                    step = size - overlap
                    if step <= 0:
                        ValueError("overlap must be smaller then size")
                    length = len(iterable)
                    windows95 = []
                    while end < length:
                        output = iterable.iloc[start:end]
                        windows95.append(output)
                        start += step
                        end += step
                    return windows95
                # endregion

                # region function: rejoin attribute data
                """
                -------------------------------------------------------------
                Function: Rejoin CDCP and Case data with age and sex data
                -------------------------------------------------------------
                """

                def add_attribute_data(attribute_data_file, join_to_data_df):

                    # Read csv files as pandas df
                    df_attributes = pd.read_csv(attribute_data_file)

                    joined_df = pd.merge(join_to_data_df, df_attributes, on="study_ID", how="inner")
                    joined_df.insert(0, 'ID', joined_df.index + 1)
                    joined_df = joined_df.drop(columns=['study_ID'])
                    final_window_df = joined_df[['ID', 'disease', new_chem_col_name, new_pop_col_name, 'AGE', 'SEX']]
                    return final_window_df
                # endregion

                # Loop: For each year, for each case pop dens window, match appropriate control pop dens
                for i in range(int(timeframes_num2) + 1):
                    new_df_name = f'{calc_abb}_pb_Accum_pd_{i}_yrs'
                    new_chem_col_name = f'{calc_abb}_expsr_{i}_yrs'
                    new_pop_col_name = f'accum_pd_7x7_{i}_yrs'
                    new_df = patient_median_exprs_df.filter(['study_ID', new_chem_col_name, new_pop_col_name],
                                                             axis=1)
                    sorted_df = new_df.sort_values(by=[f'accum_pd_7x7_{i}_yrs'], ascending=True)
                    sorted_df['rank'] = sorted_df[f'accum_pd_7x7_{i}_yrs'].rank(axis=0, method='first')
                    df_window = sliding_window(sorted_df, 50, overlap=40)

                    window_count = 50

                    for d in df_window:
                        low_pop = d.iat[0, 2]
                        high_pop = d.iat[49, 2]
                        filtered_control_df = df_control.loc[df_control[f'accum_pd_7x7_{i}_yrs'] >= low_pop].loc[df_control[f'accum_pd_7x7_{i}_yrs'] <= high_pop]
                        matched_controls_df = filtered_control_df[['study_ID', new_chem_col_name, new_pop_col_name]]
                        d = d.drop(columns=['rank'])
                        matched_df = pd.concat([d, matched_controls_df]) 
                        final_wind_df = add_attribute_data(str(attrib_file), matched_df)

                        arcpy.AddMessage(f"Exporting {calc_type} results...")

                        # Create a folder for each calc_type within the specified parent output folder
                        calc_type_folder = os.path.join(output_folder, f"{calc_type}_Output")
                        os.makedirs(calc_type_folder, exist_ok=True)
                        output_file_name = f'{new_df_name}' + f'_Window{window_count}'
                        output_file_path = os.path.join(calc_type_folder, output_file_name + ".csv")
                        final_wind_df.to_csv(output_file_path, index=False, header=True)

                        arcpy.AddMessage(f"Exporting log results for {calc_type}...")

                        # log transformation...
                        df_orig = final_wind_df
                        df_log = df_orig.iloc[:, [2]].applymap(lambda x: (x * 1000000))
                        df_log.columns = 'mil_' + df_log.columns
                        df_orig.drop(df_orig.iloc[:, [2]], axis=1, inplace=True)
                        df_orig.insert(2, df_log.columns[0], df_log.iloc[:, [0]])
                        df_log = df_orig.iloc[:, [2]].applymap(lambda x: np.log10(x + 1))
                        df_log.columns = 'log_' + df_log.columns
                        df_orig.drop(df_orig.iloc[:, [2]], axis=1, inplace=True)
                        df_orig.insert(2, df_log.columns[0], df_log.iloc[:, [0]])

                        log_folder_path = os.path.join(output_folder, "Log")
                        os.makedirs(log_folder_path, exist_ok=True)
                        calc_type_log_folder = os.path.join(log_folder_path, f"{calc_type}_Log_Output")
                        os.makedirs(calc_type_log_folder, exist_ok=True)

                        log_output_file_path = os.path.join(calc_type_log_folder, output_file_name + "_LogM.csv")
                        df_orig.to_csv(log_output_file_path, index=False, header=True)

                        window_count += 10
        return



class Tool6(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "R Logistic Regression"
        self.description = "Logistic Regression Tool with Confidence Intervals"
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        # Set up input parameters for the tool
        params = [
            arcpy.Parameter(displayName="Folder containing all window analysis files (Level A folder: 'Window_R_Input')",
                            name="input_WA_folder",
                            datatype="DEFolder",
                            parameterType="Required",
                            direction="Input"),
            arcpy.Parameter(displayName="Output folder for all logistic regression files",
                            name="output_folder",
                            datatype="DEFolder",
                            parameterType="Required",
                            direction="Input"),
            arcpy.Parameter(displayName="Include Population Density in Logistic Regression (Optional)",
                            name="include_population_density",
                            datatype="GPBoolean",
                            parameterType="Optional",
                            direction="Input")
        ]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal validation is performed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool parameter."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        import pandas as pd
        import numpy as np
        import statsmodels.api as sm
        import os

        # Retrieve the input parameters
        input_WA_folder = parameters[0].valueAsText
        output_folder = parameters[1].valueAsText
        include_population_density = parameters[2].value

        folder_variations = ["Accum", "Median", "Max", "Min"]

        for level_b_folder in os.listdir(input_WA_folder):
            level_b_path = os.path.join(input_WA_folder, level_b_folder)

            if os.path.isdir(level_b_path):
                arcpy.AddMessage(f"Working on {level_b_folder}")
                log_folder = os.path.join(level_b_path, "Log")

                if os.path.exists(log_folder):
                    output_level_b_folder = os.path.join(output_folder, level_b_folder)
                    os.makedirs(output_level_b_folder, exist_ok=True)

                    for variation in folder_variations:
                        arcpy.AddMessage(f"Working on variation: {variation}")
                        output_folder_variation = os.path.join(output_level_b_folder, variation + "_LogROutput")
                        os.makedirs(output_folder_variation, exist_ok=True)

                        input_folder = os.path.join(log_folder, variation.lower() + "_Log_Output")
                        try:
                            os.chdir(input_folder)
                        except FileNotFoundError:
                            arcpy.AddError(f"Error: The input folder '{input_folder}' does not exist.")
                            continue

                        # Look for CSV and Excel files
                        file_list = [file for file in os.listdir() if file.endswith("LogM.csv")]
                        excel_files = [file for file in os.listdir() if file.endswith("LogM.xlsx")]

                        # Convert Excel files to CSV
                        for excel_file in excel_files:
                            try:
                                df_excel = pd.read_excel(excel_file, engine="openpyxl")
                                csv_name = os.path.splitext(excel_file)[0] + ".csv"
                                df_excel.to_csv(csv_name, index=False)
                                file_list.append(csv_name)
                                arcpy.AddMessage(f"Converted {excel_file} to {csv_name}")
                            except Exception as e:
                                arcpy.AddError(f"Error converting Excel file '{excel_file}': {str(e)}")

                        if not file_list:
                            arcpy.AddError(f"No CSV files ending with 'LogM.csv' found in '{input_folder}'.")
                            continue

                        for file_name in file_list:
                            try:
                                temp_data = pd.read_csv(file_name)
                                temp_data_new = temp_data.iloc[:, 1:]
                                temp_data_new.columns = ["disease"] + list(temp_data_new.columns[1:])

                                if "disease" not in temp_data_new.columns:
                                    arcpy.AddError(f"Error: The column 'disease' is missing from '{file_name}'.")
                                    continue

                                # Exclude population density if not included
                                if include_population_density:
                                    columns_to_include = temp_data_new.columns
                                else:
                                    columns_to_include = [col for col in temp_data_new.columns if not col.startswith("accum_pd")]

                                temp_data_filtered = temp_data_new[columns_to_include]
                                y = temp_data_filtered['disease']
                                X = temp_data_filtered.iloc[:, 1:]
                                X = sm.add_constant(X)

                                temp_regression = sm.Logit(y, X).fit(disp=False)

                                # Build results DataFrame with coefficients, odds ratios, and confidence intervals
                                logit_results = pd.DataFrame({
                                    'Variable': X.columns,
                                    'Coefficient': temp_regression.params,
                                    'Std. Error': temp_regression.bse,
                                    'z value': temp_regression.params / temp_regression.bse,
                                    'P>|z|': temp_regression.pvalues,
                                    'Odds Ratio': np.exp(temp_regression.params),
                                    'CI_oddsratio_low_95': np.exp(temp_regression.params - 1.96 * temp_regression.bse),
                                    'CI_oddsratio_up_95': np.exp(temp_regression.params + 1.96 * temp_regression.bse),
                                    'CI_oddsratio_low_99': np.exp(temp_regression.params - 2.58 * temp_regression.bse),
                                    'CI_oddsratio_up_99': np.exp(temp_regression.params + 2.58 * temp_regression.bse)
                                })

                                # Save results to CSV
                                output_file_path = os.path.join(output_folder_variation, os.path.splitext(file_name)[0] + "_results.csv")
                                logit_results.to_csv(output_file_path, index=False)
                                arcpy.AddMessage(f"Results saved to: {output_file_path}")

                            except Exception as e:
                                arcpy.AddError(f"Error processing file '{file_name}': {str(e)}")
                                continue

    def postExecute(self, parameters):
        """This method takes place after outputs are processed and added to the display."""
        return


    

class Tool7(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Subject Location Preparation"
        self.description = "This tool adds a row of data for each location of each subject for each year"
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        params = [
            arcpy.Parameter(displayName="CSV file containing untouched, unduplicated subject locations",
                            name="input_file",
                            datatype="DEFile",
                            parameterType="Required",
                            direction="Input"),
            
            arcpy.Parameter(displayName="Output CSV file name containing duplicated subject locations",
                            name="output_file",
                            datatype="DEFile",
                            parameterType="Required",
                            direction="Input")
        ]
        return params


    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        input_file = parameters[0].valueAsText
        output_file = parameters[1].valueAsText
        
        import pandas as pd
        import os
        import csv

        # Read CSV file
        ALS_locs = pd.read_csv(input_file)

        # Data wrangling
        ALS_locs_wrang = ALS_locs.apply(lambda row: pd.DataFrame({
            'ID': row['ID'],
            'study_ID': row['study_ID'],
            'ALS_status': row['ALS_status'],
            'SEX': row['SEX'],
            'AGE': row['AGE'],
            'source_type': row['source_type'],
            'source': row['source'],
            'seq': row['seq'],
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'yr_addr_end': row['yr_addr_end'],
            'yr_addr_start': row['yr_addr_start'],
            'index_year': row['index_year'],
            'num_yrs_ad': row['num_yrs_ad'],
            'index_yr_minus15': row['index_minus15'],
            'year': list(range(row['yr_addr_start'], row['yr_addr_end'] + 1))
        }), axis=1)

        # Concatenate the list of DataFrames into a single DataFrame
        ALS_locs_wrang = pd.concat(ALS_locs_wrang.values, ignore_index=True)

        # Write to CSV
        ALS_locs_wrang.to_csv(output_file, index=False)
    
        return


class Tool8(object):
    def __init__(self):
        """Create test CSV to monitor progress"""
        self.label = "Create test CSV"
        self.description = "Thistool creates a CSV output that allows you to see if your window analysis is going smoothly."
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions."""
        # Define input parameters
        params = [
            arcpy.Parameter(displayName="Input Union (N2Z) Media 1 Mortality file",
                            name="N2Z_M1_Mort_file",
                            datatype="DEFile",
                            parameterType="Required",
                            direction="Input"),
            arcpy.Parameter(displayName="Input Union (N2Z) Media 1 Control file",
                            name="N2Z_M1_Cont_file",
                            datatype="DEFile",
                            parameterType="Required",
                            direction="Input"),
            arcpy.Parameter(displayName="Input Union (N2Z) Media 2 Mortality file",
                            name="N2Z_M2_Mort_file",
                            datatype="DEFile",
                            parameterType="Required",
                            direction="Input"),
            arcpy.Parameter(displayName="Input Union (N2Z) Media 2 Control file",
                            name="N2Z_M2_Cont_file",
                            datatype="DEFile",
                            parameterType="Required",
                            direction="Input"),            
            arcpy.Parameter(displayName="Input Intersect (Orig) M1M2 Mortality file",
                            name="Orig_M1M2_Mort_file",
                            datatype="DEFile",
                            parameterType="Required",
                            direction="Input"),
            arcpy.Parameter(displayName="Input Intersect (Orig) M1M2 Control file",
                            name="Orig_M1M2_Cont_file",
                            datatype="DEFile",
                            parameterType="Required",
                            direction="Input"),
            arcpy.Parameter(displayName="Input Union (N2Z) M1M2 Mortality file",
                            name="N2Z_M1M2_Mort_file",
                            datatype="DEFile",
                            parameterType="Required",
                            direction="Input"),
            arcpy.Parameter(displayName="Input Union (N2Z) M1M2 Control file",
                            name="N2Z_M1M2_Cont_file",
                            datatype="DEFile",
                            parameterType="Required",
                            direction="Input"),

            arcpy.Parameter(displayName="Output CSV file Location",
                            name="output_loc",
                            datatype="DEFile",
                            parameterType="Required",
                            direction="Output")
        ]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed. This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        # Get input parameters
        N2Z_M1_Mort_file = parameters[0].valueAsText
        N2Z_M1_Cont_file = parameters[1].valueAsText
        N2Z_M2_Mort_file = parameters[2].valueAsText
        N2Z_M2_Cont_file = parameters[3].valueAsText
        Orig_M1M2_Mort_file = parameters[4].valueAsText
        Orig_M1M2_Cont_file = parameters[5].valueAsText
        N2Z_M1M2_Mort_file = parameters[6].valueAsText
        N2Z_M1M2_Cont_file = parameters[7].valueAsText
        output_loc = parameters[8].valueAsText
        

        import pandas as pd
        from openpyxl import Workbook

        # Load Mortality and Control Data (N2Z M1)
        mortality_data = pd.read_csv(N2Z_M1_Mort_file)
        control_data = pd.read_csv(N2Z_M1_Cont_file)

        # Specify the columns you want to select from each dataset
        mortality_selected_columns = ['study_ID', 'year', 'expsr_val']
        control_selected_columns = ['study_ID', 'year', 'expsr_val']

        # Rename the selected columns to make them unique
        mortality_data_selected = mortality_data[mortality_selected_columns].rename(columns={'expsr_val': 'M1'})
        control_data_selected = control_data[control_selected_columns].rename(columns={'expsr_val': 'M1'})

        # Add a 'CaseOrControl' column to indicate the source
        mortality_data_selected['CaseOrControl'] = 1
        control_data_selected['CaseOrControl'] = 0


        # Combine mortality and control data
        combined_data = pd.concat([mortality_data_selected, control_data_selected])

        combined_data = combined_data[['study_ID', 'CaseOrControl', 'year', 'M1']]

        additional_data_list=[]
        # Replace 'additional_data_1.csv', 'additional_data_2.csv', etc., with your file paths (N2Z M2)
        additional_data_files = [N2Z_M2_Mort_file, N2Z_M2_Cont_file]
        for file_path in additional_data_files:
            additional_data = pd.read_csv(file_path)
            additional_data = additional_data[['study_ID', 'year', 'expsr_val']]
            additional_data = additional_data.rename(columns={'expsr_val': 'M2'})
            additional_data_list.append(additional_data)


        additional_data_combined = pd.concat(additional_data_list)

        # Merge the additional columns into the existing DataFrames using 'Study_ID' and 'Year' as keys
        combined_data = pd.merge(combined_data, additional_data_combined, on=['study_ID', 'year'], how='left')
        # Add more merges for additional files as needed

        additional_data_list_2=[]
        # Replace 'additional_data_1.csv', 'additional_data_2.csv', etc., with your file paths (Orig M1M2)
        additional_data_files = [Orig_M1M2_Mort_file, Orig_M1M2_Cont_file]
        for file_path in additional_data_files:
            additional_data = pd.read_csv(file_path)
            additional_data = additional_data[['study_ID', 'year', 'expsr_val']]
            additional_data = additional_data.rename(columns={'expsr_val': 'Intersect(M1+M2)'})
            additional_data_list_2.append(additional_data)


        additional_data_combined_2 = pd.concat(additional_data_list_2)

        combined_data = pd.merge(combined_data, additional_data_combined_2, on=['study_ID', 'year'], how='left')


        additional_data_list_3=[]
        # Replace 'additional_data_1.csv', 'additional_data_2.csv', etc., with your file paths (N2Z M1M2)
        additional_data_files = [N2Z_M1M2_Mort_file, N2Z_M1M2_Cont_file]
        for file_path in additional_data_files:
            additional_data = pd.read_csv(file_path)
            additional_data = additional_data[['study_ID', 'year', 'expsr_val']]
            additional_data = additional_data.rename(columns={'expsr_val': 'Union(M1+M2)'})
            additional_data_list_3.append(additional_data)


        additional_data_combined_3 = pd.concat(additional_data_list_3)

        combined_data = pd.merge(combined_data, additional_data_combined_3, on=['study_ID', 'year'], how='left')
        # Calculate the ratio and add it as a new column
        combined_data['Ratio (Int/Union)'] = combined_data['Intersect(M1+M2)'] / combined_data['Union(M1+M2)']

        # Sort the combined data by the 'Year' column in descending order (most recent first)
        combined_data.sort_values(by='year', ascending=False, inplace=True)

        # Create an Excel Writer Object 
        # Select location you want to export to
        writer = pd.ExcelWriter(output_loc, engine='openpyxl')
        writer.book = Workbook()

        # Iterate through years
        for year in combined_data['year'].unique():
            # Filter data for the current year
            year_data = combined_data[combined_data['year'] == year]

            # Create a new sheet for the current year
            year_data.to_excel(writer, sheet_name=str(year), index=False)

        # Save and close the Excel file
        writer.save()


        # Set messages
        arcpy.AddMessage(f"Tool executed successfully. Output at {output_loc}.")

        return

class Tool9(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Combine R Results"
        self.description = "This tool takes a folder of logistic regression outputs from the Window analysis and combines them into one CSV file. This file can them be formatted for result visualizations."
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        params = [
            arcpy.Parameter(displayName="Folder containing all R log regression outputs. (parent folder for Intersect_M1M2, Union_M1, Union_M2, and Union_M1M2 folders that contain R result csvs.)",
                            name="input_folder",
                            datatype="DEFolder",
                            parameterType="Required",
                            direction="Input"),
            
            arcpy.Parameter(displayName="Overall Output folder for Combined CSVs (subfolders will be created by the tool.)",
                            name="output_folder",
                            datatype="DEFolder",
                            parameterType="Required",
                            direction="Input")
        ]
        return params


    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return
    

    def execute(self, parameters, messages):
        """The source code of the tool."""
        input_folder = parameters[0].valueAsText
        output_folder = parameters[1].valueAsText
        
        import pandas as pd
        import os

        # List of variations
        variations = ["Accum", "Median", "Max", "Min"]

        for IntOrUni_folder in os.listdir(input_folder):
            input_IntOrUni_Folder = os.path.join(input_folder, IntOrUni_folder)
            if os.path.isdir(input_IntOrUni_Folder):
                output_IntOrUni_Folder = os.path.join(output_folder, IntOrUni_folder)
                if not os.path.exists(output_IntOrUni_Folder):
                    os.makedirs(output_IntOrUni_Folder)

                # Iterate over each variation
                for variation in variations:
                    # Get the folder path for the current variation
                    variation_folder = os.path.join(input_IntOrUni_Folder, variation + "_LogROutput")
                    
                    # Check if the variation folder exists
                    if os.path.exists(variation_folder):
                        # Initialize list to store DataFrame for each file
                        combined_data = []

                        # Iterate over CSV files in the current variation folder
                        for filename in os.listdir(variation_folder):
                            if filename.endswith('.csv'):
                                csv_path = os.path.join(variation_folder, filename)
                                # Read the CSV file skipping the secon row that contains the constant
                                df = pd.read_csv(csv_path, skiprows=[1])
                                # Add a new column with index row names
                                df['Index_Row'] = df.index
                                # Move the Index_Row column to the front
                                df = df[['Index_Row'] + [col for col in df.columns if col != 'Index_Row']]
                                # Add a new column with the file name
                                df['File_Name'] = filename
                                if 'Index_Row' in df.columns:
                                    df.drop(columns=['Index_Row'], inplace=True)

                                # Append the DataFrame to the list
                                combined_data.append(df)

                        if combined_data:
                            # Combine all DataFrames into a single DataFrame
                            combined_df = pd.concat(combined_data, ignore_index=True)

                            # Save the combined DataFrame to a new CSV file
                            output_file = os.path.join(output_IntOrUni_Folder, f"{variation}_combined.csv")
                            combined_df.to_csv(output_file, index=False)
                            arcpy.AddMessage(f"Combined CSV files for {variation} successfully. Output can be found at {output_file}.")
                        else:
                            arcpy.AddMessage(f"No CSV files were processed for {variation}.")
        
        return
    
class Tool10(object):
    def __init__(self):
        """Reformat R Results"""
        self.label = "Reformat R Results"
        self.description = "This tool reformats R results so that they can be visualized in Microsoft Excel."
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        params = [
            # Input folder parameter
            arcpy.Parameter(
                displayName="Input Folder",
                name="input_folder",
                datatype="DEFolder",
                parameterType="Required",
                direction="Input"
            ),
            
            arcpy.Parameter(
                displayName="Analysis Type (IM1M2, UM1, UM2, or UM1M2)",
                name="analysis_type",
                datatype="GPString",
                parameterType="Required",
                direction="Input"
            )
        ]
        
        # Set value list for the 'analysis_type' parameter
        params[1].filter.type = "ValueList"
        params[1].filter.list = ["IM1M2", "UM1", "UM2", "UM1M2"]
        return params

    def execute(self, parameters, messages):
        """Execute the tool"""
        import os
        import glob
        import pandas as pd

        # Get parameter values
        input_folder = parameters[0].valueAsText
        analysis_type = parameters[1].valueAsText

        # Define paths for the new folders inside the input folder
        odds_ratio_output_folder = os.path.join(input_folder, "Odds_Ratios_forVisualization")
        significance_output_folder = os.path.join(input_folder, "Significance_forVisualization")

        # Create the folders if they do not already exist
        if not os.path.exists(odds_ratio_output_folder):
            os.makedirs(odds_ratio_output_folder)

        if not os.path.exists(significance_output_folder):
            os.makedirs(significance_output_folder)

        # Function to extract window and year information from filename
        def extract_info(filename):
            parts = filename.split('_')
            year = int(parts[4])
            window = int(parts[6].replace("Window", ""))  # Extracting window number from the filename
            return year, window

        # Iterate over each file in the input folder
        for file_path in glob.glob(os.path.join(input_folder, '*.csv')):
            # Extract the file name without directory part
            file_name = os.path.basename(file_path)

            # Extracting the type of data from the file name
            data_type = file_name.split('_')[0]

            # Read the data from the CSV file
            data = pd.read_csv(file_path)

            # Filter rows where 'Variable' column starts with "log_mil"
            log_mil_data = data[data['Variable'].astype(str).str.startswith("log_mil")]

            # Columns to remove
            columns_to_remove = ['Coefficient', 'Std. Error', 'z value', 'P>|z|', 'CI_oddsratio_low_99', 'CI_oddsratio_up_99']

            # Remove the specified columns
            log_mil_data = log_mil_data.drop(columns=columns_to_remove)

            # Extract year and window information
            log_mil_data[['Year', 'Window']] = log_mil_data['File_Name'].apply(lambda x: pd.Series(extract_info(x)))

            # Sorting the data
            log_mil_data_sorted = log_mil_data.sort_values(by=['Year', 'Window'], ascending=[False, True])

            # Creating the odds ratio DataFrame
            odds_ratio_df = log_mil_data_sorted.pivot(index='Window', columns='Year', values='Odds Ratio')
            odds_ratio_df = odds_ratio_df.sort_index(axis=0, ascending=False).sort_index(axis=1, ascending=False)

            # Construct the output file name for odds ratio data
            odds_ratio_output_file_name = f"{analysis_type}_{data_type}_odds_ratio_data.xlsx"
            odds_ratio_output_file_path = os.path.join(odds_ratio_output_folder, odds_ratio_output_file_name)

            # Save the odds ratio DataFrame to Excel
            odds_ratio_df.to_excel(odds_ratio_output_file_path, index=True)

            # Check for significance using 95% confidence intervals
            for index, row in log_mil_data_sorted.iterrows():
                is_significant = (row['CI_oddsratio_low_95'] > 1) or (row['CI_oddsratio_up_95'] < 1)
                log_mil_data_sorted.at[index, 'Significance'] = 1 if is_significant else 0

            # Reshape the DataFrame with 'Year' as index and 'Window' as columns for Significance
            signif_df = log_mil_data_sorted.pivot(index='Window', columns='Year', values='Significance').fillna(0)
            signif_df = signif_df.sort_index(axis=0, ascending=False).sort_index(axis=1, ascending=False)

            # Construct the output file name for significance data
            significance_output_file_name = f"{analysis_type}_{data_type}_significance_data.xlsx"
            significance_output_file_path = os.path.join(significance_output_folder, significance_output_file_name)

            # Save the significance DataFrame to Excel
            signif_df.to_excel(significance_output_file_path, index=True)

            # Reshape the DataFrame with 'Year' as index and 'Window' as columns for CI_oddsratio_low_95
            ci_low_df = log_mil_data_sorted.pivot(index='Window', columns='Year', values='CI_oddsratio_low_95').fillna(0)
            ci_low_df = ci_low_df.sort_index(axis=0, ascending=False).sort_index(axis=1, ascending=False)

            # Construct the output file name for CI_oddsratio_low_95 data
            ci_low_output_file_name = f"{analysis_type}_{data_type}_ci_low_data.xlsx"
            ci_low_output_file_path = os.path.join(significance_output_folder, ci_low_output_file_name)

            # Save the CI_oddsratio_low DataFrame to Excel
            ci_low_df.to_excel(ci_low_output_file_path, index=True)

            # Reshape the DataFrame with 'Year' as index and 'Window' as columns for CI_oddsratio_up_95
            ci_up_df = log_mil_data_sorted.pivot(index='Window', columns='Year', values='CI_oddsratio_up_95').fillna(0)
            ci_up_df = ci_up_df.sort_index(axis=0, ascending=False).sort_index(axis=1, ascending=False)

            # Construct the output file name for CI_oddsratio_up_95 data
            ci_up_output_file_name = f"{analysis_type}_{data_type}_ci_up_data.xlsx"
            ci_up_output_file_path = os.path.join(significance_output_folder, ci_up_output_file_name)

            # Save the CI_oddsratio_up DataFrame to Excel
            ci_up_df.to_excel(ci_up_output_file_path, index=True)

        return




class Tool12(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Visualizations for OR and Significance Tables"
        self.description = "This tool formats odds ratio (OR) and significance tables, applies conditional formatting, and saves formatted Excel files in the Formatted_Outputs folder. For full details, see the Procedure for Creating Visualizations document in the shared drive."
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        params = [
            arcpy.Parameter(
                displayName="Parent Folder Containing All R Reformatted data",
                name="outer_parent_dir",
                datatype="DEFolder",
                parameterType="Required",
                direction="Input"
            )
        ]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify parameter values before validation."""
        return

    def updateMessages(self, parameters):
        """Modify tool messages after validation."""
        return

    def execute(self, parameters, messages):
        import os
        import arcpy
        from openpyxl import load_workbook
        from openpyxl.styles import Font, Alignment, PatternFill
        from openpyxl.utils import get_column_letter
        from openpyxl.formatting.rule import ColorScaleRule

        outer_parent_dir = parameters[0].valueAsText  

        if not os.path.exists(outer_parent_dir):
            arcpy.AddError(f"Directory does not exist: {outer_parent_dir}")
            raise ValueError("Invalid directory path.")

        # Font and formatting settings
        default_font = Font(name="Times New Roman", size=8)
        bold_black_font = Font(name="Times New Roman", size=8, bold=True, color="000000")  # Bold Black
        gray_font = Font(name="Times New Roman", size=8, color="4D4D4D")  # Dark Gray
        light_red_fill = PatternFill(start_color="FFCCCB", end_color="FFCCCB", fill_type="solid")  # Light Red Fill
        dark_red_font = Font(name="Times New Roman", size=8, color="FF0000")  # Dark Red Text
        row_height = 10
        column_width = 3.86

        # Loop through all subdirectories in the outer parent directory
        for subfolder in os.listdir(outer_parent_dir):
            subfolder_path = os.path.join(outer_parent_dir, subfolder)

            if not os.path.isdir(subfolder_path):
                continue
            
            # Define subfolders inside each dataset folder (e.g., N2Z_M1M2)
            or_folder = os.path.join(subfolder_path, "Odds_Ratios_forVisualization")
            significance_folder = os.path.join(subfolder_path, "Significance_forVisualization")
            output_folder = os.path.join(subfolder_path, "Formatted_Outputs")

            os.makedirs(output_folder, exist_ok=True)

            or_files = {
                f.split("_odds_ratio_data.xlsx")[0]: os.path.join(or_folder, f)
                for f in os.listdir(or_folder) if f.endswith("_odds_ratio_data.xlsx")
            } if os.path.exists(or_folder) else {}

            significance_files = {
                f.split("_significance_data.xlsx")[0]: os.path.join(significance_folder, f)
                for f in os.listdir(significance_folder) if f.endswith("_significance_data.xlsx")
            } if os.path.exists(significance_folder) else {}

            for key in or_files.keys() & significance_files.keys():
                or_file_path = or_files[key]
                significance_file_path = significance_files[key]
                output_file_path = os.path.join(output_folder, f"{key}_FINAL.xlsx")

                arcpy.AddMessage(f"Processing in {subfolder}: {key}")
                arcpy.AddMessage(f"  - OR File: {or_file_path}")
                arcpy.AddMessage(f"  - Significance File: {significance_file_path}")
                arcpy.AddMessage(f"  - Exporting to: {output_file_path}")

                or_wb = load_workbook(or_file_path)
                or_sheet = or_wb.active
                significance_wb = load_workbook(significance_file_path)
                significance_sheet = significance_wb.active

                for row in or_sheet.iter_rows():
                    for cell in row:
                        cell.font = default_font
                        cell.alignment = Alignment(horizontal='center', vertical='center')

                for i, row in enumerate(or_sheet.iter_rows(), start=1):
                    or_sheet.row_dimensions[i].height = row_height

                for col in range(1, or_sheet.max_column + 1):
                    or_sheet.column_dimensions[get_column_letter(col)].width = column_width

                min_color = "0070c0"
                mid_color = "ffffff"
                max_color = "ff0000"

                for row in or_sheet.iter_rows(min_row=2, min_col=2, max_row=or_sheet.max_row, max_col=or_sheet.max_column):
                    for cell in row:
                        if isinstance(cell.value, (int, float)):
                            cell.value = round(cell.value, 2)

                color_scale_rule = ColorScaleRule(
                    start_type="min", start_color=min_color,
                    mid_type="num", mid_value=1, mid_color=mid_color,
                    end_type="max", end_color=max_color,
                )

                end_col = get_column_letter(or_sheet.max_column)
                end_row = or_sheet.max_row
                or_sheet.conditional_formatting.add(f"B2:{end_col}{end_row}", color_scale_rule)

                for row_idx, (or_row, significance_row) in enumerate(
                    zip(or_sheet.iter_rows(min_row=2, min_col=2, max_row=or_sheet.max_row, max_col=or_sheet.max_column),
                        significance_sheet.iter_rows(min_row=2, min_col=2, max_row=significance_sheet.max_row, max_col=significance_sheet.max_column)),
                    start=2
                ):
                    for col_idx, (or_cell, significance_cell) in enumerate(zip(or_row, significance_row), start=2):
                        if significance_cell.value == 1:
                            or_cell.font = bold_black_font
                        elif significance_cell.value == 0:
                            or_cell.font = gray_font

                start_col_significance = or_sheet.max_column + 3

                for row_idx, significance_row in enumerate(
                    significance_sheet.iter_rows(min_row=1, min_col=1, max_row=significance_sheet.max_row, max_col=significance_sheet.max_column),
                    start=1
                ):
                    for col_idx, significance_cell in enumerate(significance_row, start=start_col_significance):
                        target_cell = or_sheet.cell(row=row_idx, column=col_idx)
                        target_cell.value = significance_cell.value
                        target_cell.font = default_font
                        target_cell.alignment = Alignment(horizontal='center', vertical='center')

                for row_idx, significance_row in enumerate(
                    or_sheet.iter_rows(
                        min_row=2, min_col=start_col_significance,
                        max_row=significance_sheet.max_row,
                        max_col=start_col_significance + significance_sheet.max_column - 1,
                    ),
                    start=2
                ):
                    for cell in significance_row:
                        if cell.value == 1:
                            cell.fill = light_red_fill
                            cell.font = dark_red_font
                        elif cell.value == 0:
                            cell.font = default_font

                for i in range(1, or_sheet.max_row + 1):
                    or_sheet.row_dimensions[i].height = row_height

                for col in range(start_col_significance, start_col_significance + significance_sheet.max_column):
                    or_sheet.column_dimensions[get_column_letter(col)].width = column_width

                or_wb.save(output_file_path)
                arcpy.AddMessage(f"*** Saved in {subfolder}: {output_file_path}")

        arcpy.AddMessage("***** Batch processing complete for all subdirectories *****")



        return

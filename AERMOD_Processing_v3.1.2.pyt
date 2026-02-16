"""
===============================================================================
AERMOD PROCESSING PIPELINE - ORGANIZED FOR REUSABLE NATIONWIDE DATA
===============================================================================

DESCRIPTION:
    Streamlined automated pipeline for processing EPA AERMOD data from raw CSV files
    through final raster outputs. Uses point grid with IDW interpolation.
    
    NEW ORGANIZATION:
    - Steps 1-2: Output to NATIONWIDE processing location (reusable)
    - Steps 3-8: Output to STUDY AREA location (state/region specific)

PRE-PROCESSING REQUIREMENTS (DO ONCE OUTSIDE THIS TOOL, Nation-wide):
    1. Convert EPA polygon grid to points (Feature To Point at centroid)
    2. Ensure points have a UniqueID field matching CSV uID values
    3. Save as: "EPA_Grid_Points.shp" (or .gdb feature class)
    4. If no CS is assigned, define project to GCS NAD 1983 (2011) Meters
    
    COORDINATE SYSTEM NOTES:
    - Step 2 (join): Works with ANY coordinate system (attribute join only)
    - Step 3 (clip): If used, clip boundary defines output coordinate system
    - Step 4 (null to zero): Works with any coordinate system
    - Step 5 (IDW): REQUIRES projected coordinate system (meters!!/feet)
    
    RECOMMENDED APPROACH:
    - If using clip boundary: Point grid can be projected in any coordinate system
    - If NOT using clip boundary: Point grid MUST be pre-projected
    
TOOLBOX SIMPLIFIED WORKFLOW:
    Tool 1: Aggregate Media          - Sum chemical concentration values (EPA Chemical CSVs) by unique cell ID 
    Tool 2: Simple Join (OPTIMIZED)  - Join point grid with aggregated CSV data (KEEPS ALL POINTS)
    
    >>> Steps 1-2 output to NATIONWIDE location (reusable for all study areas)
    
    Tool 3: Clip to Boundary         - Clip to boundary & reproject (optional)
    Tool 4: Set Null to Zero         - Replace NULL concentration values with 0
    Tool 5: IDW Interpolation        - Interpolate points to smooth raster surface
    Tool 6: Scale rasters by 1 billion
    Tool 7: Convert to integer data type
    Tool 8: Add M1 and M2 rasters    - UNION
    
    >>> Steps 3-8 output to STUDY AREA location (state/region specific)

OUTPUT STRUCTURE:
    
    NATIONWIDE_OUTPUT_WORKSPACE/
    └── Chemical_[num]/
        └── Years_[start]-[end]/
            ├── Step1_Aggregated_Media/
            └── Step2_Joined_Points/        ← REUSE THIS FOR ALL STUDY AREAS
    
    STUDY_AREA_OUTPUT_WORKSPACE/
    └── [Study_Area_Name]/
        └── Chemical_[num]/
            └── Years_[start]-[end]/
                ├── Step3_Clipped_Data/
                ├── Step4_Null_To_Zero/
                ├── Step5_IDW_Rasters/
                ├── Step6_Scaled_Rasters/
                ├── Step7_Integer_Rasters/
                ├── Step8_Combined_Rasters/
                └── processing_log.txt

===============================================================================
"""

# Standard imports
import arcpy
import os
import pandas as pd
import numpy as np
import glob
from glob import iglob
import shutil
import datetime
import time

# Tool configuration
arcpy.env.overwriteOutput = True

class Toolbox(object):
    def __init__(self):
        """AERMOD Processing Pipeline Toolbox"""
        self.label = "AERMOD Processing Pipeline - Organized"
        self.alias = "aermod_pipeline_organized"
        self.tools = [OrganizedAERMODPipeline]

class OrganizedAERMODPipeline(object):
    def __init__(self):
        self.label = "Organized AERMOD Pipeline (Nationwide + Study Area Outputs)"
        self.description = "AERMOD processing with separate nationwide and study area output locations"
        self.canRunInBackground = False

    def getParameterInfo(self):
        params = []

        # Tool selection parameter
        params.append(arcpy.Parameter(
            displayName="Tools to run, 8 total (comma-separated, e.g., '1,2,3' or 'all' for complete pipeline)",
            name="tools_to_run",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        ))
        params[0].value = "all"

        # Core processing parameters
        params.append(arcpy.Parameter(
            displayName="Chemical Attribute Data year folders directory location",
            name="Year_dir",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        ))

        params.append(arcpy.Parameter(
            displayName="Chemical Number",
            name="chem_num",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        ))

        params.append(arcpy.Parameter(
            displayName="Year to process (optional - leave blank for all years)",
            name="year_input",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        ))

        # NEW: Separate output workspaces
        params.append(arcpy.Parameter(
            displayName="Nationwide Output Workspace (for Steps 1-2, reusable data)",
            name="nationwide_workspace",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        ))

        params.append(arcpy.Parameter(
            displayName="Study Area Output Workspace (for Steps 3-8, region-specific data)",
            name="study_area_workspace",
            datatype="DEFolder",
            parameterType="Optional",
            direction="Input"
        ))

        params.append(arcpy.Parameter(
            displayName="Study Area Name (e.g., 'Ohio', 'California', 'Great_Lakes')",
            name="study_area_name",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        ))

        params.append(arcpy.Parameter(
            displayName="First Year of Data",
            name="startyr",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        ))

        params.append(arcpy.Parameter(
            displayName="Last Year of Data",
            name="endyr",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        ))

        # Point grid parameter
        params.append(arcpy.Parameter(
            displayName="Empty Point Grid (with UniqueID field matching CSV uID values)",
            name="point_grid",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Input"
        ))

        params.append(arcpy.Parameter(
            displayName="Media type to process",
            name="media",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        ))
        params[10].filter.type = "ValueList"
        params[10].filter.list = ["M1", "M2", "Both"]
        params[10].value = "Both"

        params.append(arcpy.Parameter(
            displayName="Continue if years are skipped?",
            name="continue_on_skip",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        ))
        params[11].value = True

        # Testing and file management parameters
        params.append(arcpy.Parameter(
            displayName="Clean up previous run before starting",
            name="cleanup_previous",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        ))
        params[12].value = False

        params.append(arcpy.Parameter(
            displayName="Clip Boundary (optional - defines output coordinate system if provided)",
            name="clip_boundary",
            datatype="DEFeatureClass",
            parameterType="Optional",
            direction="Input"
        ))

        params.append(arcpy.Parameter(
            displayName="Output Cell Size for Rasters (in output coordinate system units)",
            name="cell_size",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        ))
        params[14].value = "100"  # Default cell size

        params.append(arcpy.Parameter(
            displayName="Reference Raster for Cell Size (optional - overrides cell size parameter)",
            name="reference_raster",
            datatype="DERasterDataset",
            parameterType="Optional",
            direction="Input"
        ))

        # Input directory for continuing from specific step
        params.append(arcpy.Parameter(
            displayName="Input directory for continuing pipeline (optional - specify existing step output)",
            name="input_directory",
            datatype="DEFolder",
            parameterType="Optional",
            direction="Input"
        ))

        # IDW Parameters
        params.append(arcpy.Parameter(
            displayName="IDW Power Parameter (default=2, standard inverse distance weighting)",
            name="idw_power",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input"
        ))
        params[17].value = 2.0  # Default IDW power

        params.append(arcpy.Parameter(
            displayName="IDW Search Radius - Number of Points (default=4)",
            name="idw_num_points",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"
        ))
        params[18].value = 4  # Default number of points for variable search radius

        return params

    def isLicensed(self):
        # Check for Spatial Analyst extension
        try:
            if arcpy.CheckExtension("Spatial") == "Available":
                return True
            else:
                return False
        except:
            return False

    def updateParameters(self, parameters):
        # Auto-populate study area workspace if running steps 3-8
        tools_param = parameters[0].valueAsText
        if tools_param:
            try:
                if tools_param.lower() != "all":
                    tools_list = [int(x.strip()) for x in tools_param.split(',')]
                    needs_study_area = any(t >= 3 for t in tools_list)
                    
                    if needs_study_area and not parameters[5].altered:
                        # Study area workspace is needed but not set
                        parameters[5].setWarningMessage("Study Area Workspace required for Steps 3-8")
            except:
                pass
        return

    def updateMessages(self, parameters):
        # Check for Spatial Analyst license
        if arcpy.CheckExtension("Spatial") != "Available":
            parameters[0].setErrorMessage("Spatial Analyst extension is required for IDW interpolation")
        
        # Validate study area workspace for steps 3-8
        tools_param = parameters[0].valueAsText
        if tools_param:
            try:
                if tools_param.lower() == "all":
                    tools_list = list(range(1, 9))
                else:
                    tools_list = [int(x.strip()) for x in tools_param.split(',')]
                
                needs_study_area = any(t >= 3 for t in tools_list)
                
                if needs_study_area:
                    if not parameters[5].valueAsText:
                        parameters[5].setErrorMessage("Study Area Workspace required when running Steps 3-8")
                    if not parameters[6].valueAsText:
                        parameters[6].setWarningMessage("Study Area Name recommended for organization")
            except:
                pass
        
        return

    def execute(self, parameters, messages):
        # Check out Spatial Analyst extension
        if arcpy.CheckExtension("Spatial") == "Available":
            arcpy.CheckOutExtension("Spatial")
        else:
            arcpy.AddError("Spatial Analyst extension is not available")
            return

        try:
            # Extract parameters
            tools_to_run = parameters[0].valueAsText
            year_dir = parameters[1].valueAsText
            chem_num = parameters[2].valueAsText
            year_input = parameters[3].valueAsText
            nationwide_workspace = parameters[4].valueAsText
            study_area_workspace = parameters[5].valueAsText
            study_area_name = parameters[6].valueAsText if parameters[6].valueAsText else "Default_Study_Area"
            startyr = parameters[7].valueAsText
            endyr = parameters[8].valueAsText
            point_grid = parameters[9].valueAsText
            media = parameters[10].valueAsText
            continue_on_skip = parameters[11].value
            cleanup_previous = parameters[12].value
            clip_boundary = parameters[13].valueAsText
            cell_size = parameters[14].valueAsText
            reference_raster = parameters[15].valueAsText
            input_directory = parameters[16].valueAsText
            idw_power = parameters[17].value if parameters[17].value else 2.0
            idw_num_points = parameters[18].value if parameters[18].value else 4

            # Parse tools to run
            if tools_to_run.lower() == "all":
                tools_list = [1, 2, 3, 4, 5, 6, 7, 8]
            else:
                try:
                    tools_list = [int(x.strip()) for x in tools_to_run.split(',')]
                    tools_list = [t for t in tools_list if 1 <= t <= 8]
                    if not tools_list:
                        raise ValueError("No valid tools specified")
                except Exception as e:
                    arcpy.AddError(f"Invalid tools specification: {tools_to_run}. Use numbers 1-8 separated by commas, or 'all'")
                    return

            # Validate study area workspace for steps 3-8
            needs_study_area = any(t >= 3 for t in tools_list)
            if needs_study_area and not study_area_workspace:
                arcpy.AddError("Study Area Workspace is required when running Steps 3-8")
                return

            # Validate point grid
            if not arcpy.Exists(point_grid):
                arcpy.AddError(f"Point grid not found: {point_grid}")
                return
            
            # Verify it's a point feature class
            desc = arcpy.Describe(point_grid)
            if desc.shapeType != "Point":
                arcpy.AddError(f"Point grid must be a point feature class, not {desc.shapeType}")
                return
            
            arcpy.AddMessage(f"Using point grid: {point_grid}")
            arcpy.AddMessage(f"Point grid coordinate system: {desc.spatialReference.name}")
            
            # Validate coordinate system requirements
            self._validate_coordinate_system_requirements(point_grid, clip_boundary, tools_list)

            # Create managed workspace structures
            workspace_structure = self._create_organized_workspace(
                nationwide_workspace, 
                study_area_workspace, 
                study_area_name,
                chem_num, 
                startyr, 
                endyr,
                cleanup_previous,
                tools_list
            )

            # Initialize data flow variables
            data_flow = {
                'step1_output': None,
                'step2_output': None,
                'step3_output': None,
                'step4_output': None,
                'step5_output': None,
                'step6_output': None,
                'step7_output': None,
                'step8_output': None
            }

            arcpy.AddMessage(f"\n=== Starting Organized AERMOD Pipeline ===")
            arcpy.AddMessage(f"Running Tools: {tools_list}")
            arcpy.AddMessage(f"Nationwide workspace: {workspace_structure.get('nationwide_base', 'N/A')}")
            if needs_study_area:
                arcpy.AddMessage(f"Study area workspace: {workspace_structure.get('study_area_base', 'N/A')}")
                arcpy.AddMessage(f"Study area name: {study_area_name}")
            arcpy.AddMessage(f"IDW Parameters: Power={idw_power}, Number of Points={idw_num_points}")
            arcpy.AddMessage(f"Scaling: 1 billion (1,000,000,000) for 3 decimal places of precision")
            
            # Pre-populate data_flow from input directory if provided
            if input_directory and os.path.exists(input_directory):
                arcpy.AddMessage(f"\n=== Scanning input directory for existing outputs ===")
                self._scan_input_directory(input_directory, data_flow, chem_num, media)
                arcpy.AddMessage("=== Input directory scan complete ===\n")
            
            # ========================================================================
            # EXECUTE REQUESTED TOOLS
            # ========================================================================
            
            # Step 1: Aggregate Media (NATIONWIDE)
            if 1 in tools_list:
                arcpy.AddMessage("\n=== RUNNING TOOL 1: Aggregate Media (NATIONWIDE) ===")
                data_flow['step1_output'] = self._run_step1_aggregate_media(
                    year_dir, chem_num, year_input, workspace_structure, continue_on_skip
                )
                arcpy.AddMessage(f"Tool 1 Complete. Output: {data_flow['step1_output']}")

            # Step 2: Simple Join (NATIONWIDE)
            if 2 in tools_list:
                arcpy.AddMessage("\n=== RUNNING TOOL 2: Join Point Grid with CSV Data (NATIONWIDE - REUSABLE) ===")
                if not data_flow['step1_output']:
                    arcpy.AddError("Tool 2 requires Tool 1 output. Either run Tool 1 first or specify input directory with Step 1 output.")
                    return
                data_flow['step2_output'] = self._run_step2_simple_join_optimized(
                    data_flow['step1_output'], point_grid, chem_num, media, workspace_structure, startyr, endyr
                )
                arcpy.AddMessage(f"Tool 2 Complete. Output: {data_flow['step2_output']}")
                arcpy.AddMessage(f"\n*** IMPORTANT: Step 2 output can be reused for ANY study area ***")
                arcpy.AddMessage(f"*** Save this path for future use: {workspace_structure.get('step2')} ***\n")

            # Step 3: Clip to boundary (STUDY AREA)
            if 3 in tools_list:
                arcpy.AddMessage("\n=== RUNNING TOOL 3: Clip Data (STUDY AREA SPECIFIC) ===")
                if not data_flow['step2_output']:
                    arcpy.AddError("Tool 3 requires Tool 2 output. Either run Tools 1-2 first or specify input directory with Step 2 output.")
                    return
                if clip_boundary:
                    data_flow['step3_output'] = self._run_step3_clip_data(
                        data_flow['step2_output'], workspace_structure, clip_boundary, chem_num
                    )
                    arcpy.AddMessage(f"Tool 3 Complete. Output: {data_flow['step3_output']}")
                else:
                    arcpy.AddMessage("Tool 3: No clip boundary provided, using joined outputs directly...")
                    data_flow['step3_output'] = data_flow['step2_output']

            # Step 4: Set Null Concentrations to Zero (STUDY AREA)
            if 4 in tools_list:
                arcpy.AddMessage("\n=== RUNNING TOOL 4: Set Null Concentrations to Zero (STUDY AREA) ===")
                if not data_flow['step3_output']:
                    arcpy.AddError("Tool 4 requires Tool 3 output. Either run Tools 1-3 first or specify input directory with Step 3 output.")
                    return
                data_flow['step4_output'] = self._run_step4_null_to_zero_points(
                    data_flow['step3_output'], workspace_structure, chem_num
                )
                arcpy.AddMessage(f"Tool 4 Complete. Output: {data_flow['step4_output']}")

            # Step 5: IDW Interpolation (STUDY AREA)
            if 5 in tools_list:
                arcpy.AddMessage("\n=== RUNNING TOOL 5: IDW Interpolation (STUDY AREA) ===")
                if not data_flow['step4_output']:
                    arcpy.AddError("Tool 5 requires Tool 4 output. Either run Tools 1-4 first or specify input directory with Step 4 output.")
                    return
                data_flow['step5_output'] = self._run_step5_idw_interpolation(
                    data_flow['step4_output'], workspace_structure, chem_num, cell_size, reference_raster, 
                    clip_boundary, idw_power, idw_num_points
                )
                arcpy.AddMessage(f"Tool 5 Complete. Output: {data_flow['step5_output']}")

            # Step 6: Scale rasters by 1 billion (STUDY AREA)
            if 6 in tools_list:
                arcpy.AddMessage("\n=== RUNNING TOOL 6: Scale Rasters by 1 Billion (STUDY AREA) ===")
                if not data_flow['step5_output']:
                    arcpy.AddError("Tool 6 requires Tool 5 output. Either run Tools 1-5 first or specify input directory with Step 5 output.")
                    return
                data_flow['step6_output'] = self._run_step6_scale_rasters(
                    data_flow['step5_output'], workspace_structure, chem_num
                )
                arcpy.AddMessage(f"Tool 6 Complete. Output: {data_flow['step6_output']}")

            # Step 7: Convert to integer data type (STUDY AREA)
            if 7 in tools_list:
                arcpy.AddMessage("\n=== RUNNING TOOL 7: Convert to Integer (STUDY AREA) ===")
                if not data_flow['step6_output']:
                    arcpy.AddError("Tool 7 requires Tool 6 output. Either run Tools 1-6 first or specify input directory with Step 6 output.")
                    return
                data_flow['step7_output'] = self._run_step7_convert_to_int(
                    data_flow['step6_output'], workspace_structure, chem_num
                )
                arcpy.AddMessage(f"Tool 7 Complete. Output: {data_flow['step7_output']}")

            # Step 8: Add M1 and M2 rasters (STUDY AREA)
            if 8 in tools_list:
                arcpy.AddMessage("\n=== RUNNING TOOL 8: Combine M1 and M2 Rasters (STUDY AREA) ===")
                if not data_flow['step7_output']:
                    arcpy.AddError("Tool 8 requires Tool 7 output. Either run Tools 1-7 first or specify input directory with Step 7 output.")
                    return
                if media == "Both" and 'M1' in data_flow['step7_output'] and 'M2' in data_flow['step7_output']:
                    data_flow['step8_output'] = self._run_step8_add_media_rasters(
                        data_flow['step7_output'], workspace_structure, chem_num, startyr, endyr
                    )
                    arcpy.AddMessage(f"Tool 8 Complete. Output: {data_flow['step8_output']}")
                else:
                    arcpy.AddMessage("Tool 8: Cannot combine - requires both M1 and M2 data")
            
            arcpy.AddMessage("\n=== Pipeline execution completed successfully! ===")
            
            # Print output summary
            arcpy.AddMessage("\n=== OUTPUT SUMMARY ===")
            if 1 in tools_list or 2 in tools_list:
                arcpy.AddMessage(f"Nationwide outputs: {workspace_structure.get('nationwide_base', 'N/A')}")
            if any(t >= 3 for t in tools_list):
                arcpy.AddMessage(f"Study area outputs: {workspace_structure.get('study_area_base', 'N/A')}")
            
            # Print reusability message
            if 2 in tools_list:
                arcpy.AddMessage("\n=== REUSABILITY REMINDER ===")
                arcpy.AddMessage("Step 2 outputs (nationwide joined points) can be reused!")
                arcpy.AddMessage(f"Location: {workspace_structure.get('step2')}")
                arcpy.AddMessage("To process a different study area:")
                arcpy.AddMessage("  1. Run tools: 3,4,5,6,7,8")
                arcpy.AddMessage("  2. Set 'Input directory' to the Step2 path above")
                arcpy.AddMessage("  3. Provide new clip boundary and study area name")
                arcpy.AddMessage("  4. Set new Study Area Workspace")
            
            # Print summary
            completed_tools = [str(t) for t in tools_list]
            arcpy.AddMessage(f"\nCompleted tools: {', '.join(completed_tools)}")
            arcpy.AddMessage(f"\nNote: To convert final raster values back to μg/m³, divide by 1,000,000,000")
            
        except Exception as e:
            arcpy.AddError(f"Pipeline failed: {str(e)}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            raise
        finally:
            # Check in Spatial Analyst extension
            arcpy.CheckInExtension("Spatial")

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _validate_coordinate_system_requirements(self, point_grid, clip_boundary, tools_list):
        """Validate coordinate system setup before processing"""
        
        point_desc = arcpy.Describe(point_grid)
        point_sr = point_desc.spatialReference
        
        # If running Step 5 (IDW interpolation)
        if 5 in tools_list:
            # Check if clip boundary is provided
            if clip_boundary and arcpy.Exists(clip_boundary):
                clip_desc = arcpy.Describe(clip_boundary)
                clip_sr = clip_desc.spatialReference
                
                # Verify clip boundary is projected
                if clip_sr.type != "Projected":
                    arcpy.AddError(f"Clip boundary must be in a PROJECTED coordinate system for IDW interpolation.")
                    arcpy.AddError(f"Current clip boundary coordinate system: {clip_sr.name} (Type: {clip_sr.type})")
                    raise ValueError("Clip boundary must use projected coordinate system")
                else:
                    arcpy.AddMessage(f"✓ Clip boundary is projected: {clip_sr.name}")
                    arcpy.AddMessage(f"  Output rasters will use this coordinate system")
            else:
                # No clip boundary - point grid must already be projected
                if point_sr.type != "Projected":
                    arcpy.AddError(f"Point grid must be in a PROJECTED coordinate system for IDW interpolation when no clip boundary is provided.")
                    arcpy.AddError(f"Current point grid coordinate system: {point_sr.name} (Type: {point_sr.type})")
                    arcpy.AddError(f"Either:")
                    arcpy.AddError(f"  1. Provide a clip boundary in a projected coordinate system, OR")
                    arcpy.AddError(f"  2. Pre-project your point grid to a projected coordinate system")
                    raise ValueError("Point grid must use projected coordinate system when no clip boundary provided")
                else:
                    arcpy.AddMessage(f"✓ Point grid is projected: {point_sr.name}")
                    arcpy.AddMessage(f"  Output rasters will use this coordinate system")

    def _create_organized_workspace(self, nationwide_workspace, study_area_workspace, study_area_name,
                                   chem_num, startyr, endyr, cleanup_previous, tools_list):
        """Create organized workspace structure with separate nationwide and study area locations"""
        
        year_range = f"{startyr}-{endyr}"
        structure = {}
        
        # NATIONWIDE WORKSPACE (Steps 1-2)
        if 1 in tools_list or 2 in tools_list:
            nationwide_base = os.path.join(
                nationwide_workspace,
                f"Chemical_{chem_num}",
                f"Years_{year_range}"
            )
            
            # Cleanup if requested
            if cleanup_previous and os.path.exists(nationwide_base):
                shutil.rmtree(nationwide_base)
                arcpy.AddMessage(f"Cleaned up previous nationwide run for Chemical {chem_num}")
            
            os.makedirs(nationwide_base, exist_ok=True)
            
            structure['nationwide_base'] = nationwide_base
            structure['step1'] = os.path.join(nationwide_base, "Step1_Aggregated_Media")
            structure['step2'] = os.path.join(nationwide_base, "Step2_Joined_Points")
            
            # Create README for nationwide data
            readme_path = os.path.join(nationwide_base, "README_REUSE_THIS.txt")
            with open(readme_path, 'w') as f:
                f.write(f"NATIONWIDE BASE DATA - REUSABLE FOR ALL STUDY AREAS\n")
                f.write(f"=" * 60 + "\n\n")
                f.write(f"Chemical: {chem_num}\n")
                f.write(f"Years: {year_range}\n")
                f.write(f"Created: {datetime.datetime.now()}\n\n")
                f.write(f"IMPORTANT: Step2_Joined_Points can be reused for unlimited study areas!\n\n")
                f.write(f"To use for a new study area:\n")
                f.write(f"1. Run tools: 3,4,5,6,7,8\n")
                f.write(f"2. Set 'Input directory' to: {os.path.join(nationwide_base, 'Step2_Joined_Points')}\n")
                f.write(f"3. Provide clip boundary for your study area\n")
                f.write(f"4. Provide Study Area Workspace and Name\n\n")
                f.write(f"Contents:\n")
                f.write(f"- Step1_Aggregated_Media: CSV files aggregated by unique cell ID\n")
                f.write(f"- Step2_Joined_Points: Point grid with pollution data joined (ALL POINTS RETAINED)\n")
        
        # STUDY AREA WORKSPACE (Steps 3-8)
        if any(t >= 3 for t in tools_list):
            if not study_area_workspace:
                arcpy.AddWarning("Study area workspace not provided for steps 3-8")
            else:
                study_area_base = os.path.join(
                    study_area_workspace,
                    study_area_name,
                    f"Chemical_{chem_num}",
                    f"Years_{year_range}"
                )
                
                # Cleanup if requested
                if cleanup_previous and os.path.exists(study_area_base):
                    shutil.rmtree(study_area_base)
                    arcpy.AddMessage(f"Cleaned up previous study area run for {study_area_name}")
                
                os.makedirs(study_area_base, exist_ok=True)
                
                structure['study_area_base'] = study_area_base
                structure['step3'] = os.path.join(study_area_base, "Step3_Clipped_Data")
                structure['step4'] = os.path.join(study_area_base, "Step4_Null_To_Zero")
                structure['step5'] = os.path.join(study_area_base, "Step5_IDW_Rasters")
                structure['step6'] = os.path.join(study_area_base, "Step6_Scaled_Rasters")
                structure['step7'] = os.path.join(study_area_base, "Step7_Integer_Rasters")
                structure['step8'] = os.path.join(study_area_base, "Step8_Combined_Rasters")
                
                # Create processing log
                log_file = os.path.join(study_area_base, "processing_log.txt")
                with open(log_file, 'w') as f:
                    f.write(f"AERMOD Study Area Processing - {study_area_name}\n")
                    f.write(f"Chemical: {chem_num}\n")
                    f.write(f"Years: {year_range}\n")
                    f.write(f"Timestamp: {datetime.datetime.now()}\n")
                    f.write(f"Study Area: {study_area_name}\n")
                    f.write(f"Scaling Factor: 1,000,000,000 (1 billion)\n\n")
        
        return structure

    def _ensure_step_directory(self, step_path):
        """Create step directory if it doesn't exist"""
        os.makedirs(step_path, exist_ok=True)
        return step_path

    def _scan_input_directory(self, input_directory, data_flow, chem_num, media):
        """Scan input directory and populate data_flow with existing outputs"""
        
        arcpy.AddMessage(f"Input directory: {input_directory}")
        
        # Check for Step 1 output
        potential_step1 = os.path.join(input_directory, "Step1_Aggregated_Media")
        if os.path.exists(potential_step1):
            data_flow['step1_output'] = potential_step1
            arcpy.AddMessage(f"✓ Found Step 1 output")
        elif "Step1_Aggregated_Media" in input_directory:
            data_flow['step1_output'] = input_directory
            arcpy.AddMessage(f"✓ Found Step 1 output (direct path)")
        
        # Check for Step 2 output
        potential_step2 = os.path.join(input_directory, "Step2_Joined_Points")
        if os.path.exists(potential_step2):
            data_flow['step2_output'] = self._reconstruct_step_output(potential_step2, chem_num, media, "Joined")
            arcpy.AddMessage(f"✓ Found Step 2 output (NATIONWIDE DATA)")
        elif "Step2_Joined_Points" in input_directory:
            data_flow['step2_output'] = self._reconstruct_step_output(input_directory, chem_num, media, "Joined")
            arcpy.AddMessage(f"✓ Found Step 2 output (NATIONWIDE DATA - direct path)")
        
        # Check for Step 3 output
        potential_step3 = os.path.join(input_directory, "Step3_Clipped_Data")
        if os.path.exists(potential_step3):
            data_flow['step3_output'] = self._reconstruct_step_output(potential_step3, chem_num, media, "Clipped")
            arcpy.AddMessage(f"✓ Found Step 3 output")
        elif "Step3_Clipped_Data" in input_directory:
            data_flow['step3_output'] = self._reconstruct_step_output(input_directory, chem_num, media, "Clipped")
            arcpy.AddMessage(f"✓ Found Step 3 output (direct path)")
        elif data_flow['step2_output']:
            data_flow['step3_output'] = data_flow['step2_output']
            arcpy.AddMessage(f"✓ Using Step 2 output for Step 3 (no clipping)")
        
        # Check for Step 4 output
        potential_step4 = os.path.join(input_directory, "Step4_Null_To_Zero")
        if os.path.exists(potential_step4):
            data_flow['step4_output'] = self._reconstruct_step_output(potential_step4, chem_num, media, "Zero")
            arcpy.AddMessage(f"✓ Found Step 4 output")
        elif "Step4_Null_To_Zero" in input_directory:
            data_flow['step4_output'] = self._reconstruct_step_output(input_directory, chem_num, media, "Zero")
            arcpy.AddMessage(f"✓ Found Step 4 output (direct path)")
        
        # Check for Step 5 output (IDW)
        potential_step5 = os.path.join(input_directory, "Step5_IDW_Rasters")
        if os.path.exists(potential_step5):
            data_flow['step5_output'] = self._reconstruct_step_output(potential_step5, chem_num, media, "IDW")
            arcpy.AddMessage(f"✓ Found Step 5 output")
        elif "Step5_IDW_Rasters" in input_directory:
            data_flow['step5_output'] = self._reconstruct_step_output(input_directory, chem_num, media, "IDW")
            arcpy.AddMessage(f"✓ Found Step 5 output (direct path)")
        
        # Check for Step 6 output
        potential_step6 = os.path.join(input_directory, "Step6_Scaled_Rasters")
        if os.path.exists(potential_step6):
            data_flow['step6_output'] = self._reconstruct_step6_output(potential_step6, chem_num, media)
            arcpy.AddMessage(f"✓ Found Step 6 output")
        elif "Step6_Scaled_Rasters" in input_directory:
            data_flow['step6_output'] = self._reconstruct_step6_output(input_directory, chem_num, media)
            arcpy.AddMessage(f"✓ Found Step 6 output (direct path)")
        
        # Check for Step 7 output
        potential_step7 = os.path.join(input_directory, "Step7_Integer_Rasters", "INT")
        if os.path.exists(potential_step7):
            data_flow['step7_output'] = self._reconstruct_step7_output(potential_step7, chem_num, media)
            arcpy.AddMessage(f"✓ Found Step 7 output")
        elif "Step7_Integer_Rasters" in input_directory:
            int_subfolder = os.path.join(input_directory, "INT")
            if os.path.exists(int_subfolder):
                data_flow['step7_output'] = self._reconstruct_step7_output(int_subfolder, chem_num, media)
                arcpy.AddMessage(f"✓ Found Step 7 output (with INT subfolder)")
            elif any(f.endswith('_INT.gdb') for f in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, f))):
                data_flow['step7_output'] = self._reconstruct_step7_output(input_directory, chem_num, media)
                arcpy.AddMessage(f"✓ Found Step 7 output (direct INT folder)")
        
        # Check for Step 8 output
        potential_step8 = os.path.join(input_directory, "Step8_Combined_Rasters")
        if os.path.exists(potential_step8):
            data_flow['step8_output'] = self._reconstruct_step8_output(potential_step8, chem_num)
            arcpy.AddMessage(f"✓ Found Step 8 output")
        elif "Step8_Combined_Rasters" in input_directory:
            data_flow['step8_output'] = self._reconstruct_step8_output(input_directory, chem_num)
            arcpy.AddMessage(f"✓ Found Step 8 output (direct path)")

    def _reconstruct_step_output(self, step_dir, chem_num, media, suffix):
        """Generic method to reconstruct step output dictionary"""
        output = {}
        media_types = []
        if media in ["M1", "Both"]:
            media_types.append("M1")
        if media in ["M2", "Both"]:
            media_types.append("M2")
        
        for media_type in media_types:
            gdb_name = f"Chemical_{chem_num}_{media_type}_{suffix}.gdb"
            gdb_path = os.path.join(step_dir, gdb_name)
            if os.path.exists(gdb_path):
                output[media_type] = gdb_path
        return output

    def _reconstruct_step6_output(self, step6_dir, chem_num, media):
        """Reconstruct step 6 output (scaled rasters)"""
        output = {}
        media_types = []
        if media in ["M1", "Both"]:
            media_types.append("M1")
        if media in ["M2", "Both"]:
            media_types.append("M2")
        
        for media_type in media_types:
            for item in os.listdir(step6_dir):
                if item.endswith(f"_{media_type}_Bil.gdb"):
                    gdb_path = os.path.join(step6_dir, item)
                    output[media_type] = gdb_path
                    break
        return output

    def _reconstruct_step7_output(self, step7_dir, chem_num, media):
        """Reconstruct step 7 output (integer rasters)"""
        output = {}
        media_types = []
        if media in ["M1", "Both"]:
            media_types.append("M1")
        if media in ["M2", "Both"]:
            media_types.append("M2")
        
        for media_type in media_types:
            for item in os.listdir(step7_dir):
                if item.endswith(f"_{media_type}_INT.gdb"):
                    gdb_path = os.path.join(step7_dir, item)
                    output[media_type] = gdb_path
                    break
        return output

    def _reconstruct_step8_output(self, step8_dir, chem_num):
        """Reconstruct step 8 output (combined rasters)"""
        output = {}
        gdb_name = f"Combined_{chem_num}_M1M2.gdb"
        gdb_path = os.path.join(step8_dir, gdb_name)
        if os.path.exists(gdb_path):
            output['M1M2_Combined'] = gdb_path
        return output

    # ========================================================================
    # STEP IMPLEMENTATIONS
    # ========================================================================

    def _run_step1_aggregate_media(self, year_dir, chem_num, year_input, workspace_structure, continue_on_skip):
        """Step 1: Aggregate media - sum concentration values by unique cell ID"""
        
        step1_dir = self._ensure_step_directory(workspace_structure['step1'])
        skipped_years = []
        
        # Determine years to process
        if year_input:
            year_list = [os.path.join(year_dir, year_input)]
        else:
            year_list = [f.path for f in os.scandir(year_dir) if f.is_dir()]

        for file_path in year_list:
            year_name = os.path.basename(file_path)
            
            # Find chemistry folder
            standard_path = os.path.join(file_path, f'Chem_{chem_num}')
            nested_path = os.path.join(file_path, year_name, f'Chem_{chem_num}')

            if os.path.isdir(standard_path):
                chem_folder = standard_path
            elif os.path.isdir(nested_path):
                chem_folder = nested_path
            else:
                arcpy.AddMessage(f"No valid Chem_{chem_num} folder found for {year_name}. Skipping.")
                skipped_years.append(year_name)
                continue

            # Create output directory for this year
            year_agg_dir = os.path.join(step1_dir, year_name, f'Chem_{chem_num}', 'Aggregated_Media')
            os.makedirs(year_agg_dir, exist_ok=True)

            # Process CSV files
            media_csv_files = glob.glob(os.path.join(chem_folder, '*.csv'))
            processed_files = 0
            
            for media_file in media_csv_files:
                try:
                    df = pd.read_csv(media_file)
                    
                    # Check if required columns exist
                    required_col = f'uID_{chem_num}'
                    if required_col not in df.columns:
                        arcpy.AddWarning(f"Missing column {required_col} in {media_file}. Skipping file.")
                        continue
                    
                    # Aggregate data
                    ID_group = df.groupby([f'uID_{chem_num}'], as_index=False).agg({
                        f'rels_{chem_num}': 'first',
                        f'chem_{chem_num}': 'first',
                        f'facl_{chem_num}': 'first',
                        f'media_{chem_num}': 'first',
                        f'conc_{chem_num}': 'sum',
                        f'toxc_{chem_num}': 'sum',
                        f'score_{chem_num}': 'sum',
                        f'canc_{chem_num}': 'sum',
                        f'nocan_{chem_num}': 'sum',
                        f'pop_{chem_num}': 'first'
                    })
                    
                    # Save to output directory
                    output_file = os.path.join(year_agg_dir, os.path.basename(media_file))
                    ID_group.to_csv(output_file, index=False, header=True)
                    processed_files += 1
                    
                except Exception as e:
                    arcpy.AddWarning(f"Error processing {media_file}: {str(e)}")
                    continue
            
            if processed_files > 0:
                arcpy.AddMessage(f"Processed {processed_files} files for year {year_name}")
            else:
                skipped_years.append(year_name)

        # Handle skipped years
        if skipped_years:
            skipped_str = ', '.join(skipped_years)
            arcpy.AddWarning(f"Skipped years during aggregation: {skipped_str}")
            
            if not continue_on_skip:
                raise Exception("User chose to stop when years are skipped")

        return step1_dir

    def _run_step2_simple_join_optimized(self, aggregated_dir, point_grid, chem_num, media, workspace_structure, startyr, endyr):
        """Step 2: Join point grid with aggregated CSV data (OPTIMIZED with JoinField - KEEPS ALL POINTS)"""
        
        step2_dir = self._ensure_step_directory(workspace_structure['step2'])
        
        # Determine media types
        media_types = []
        if media in ["M1", "Both"]:
            media_types.append("M1")
        if media in ["M2", "Both"]:
            media_types.append("M2")
        
        joined_outputs = {}
        
        # Verify point grid
        if not arcpy.Exists(point_grid):
            raise ValueError(f"Point grid not found: {point_grid}")
        
        point_desc = arcpy.Describe(point_grid)
        arcpy.AddMessage(f"Point grid spatial reference: {point_desc.spatialReference.name}")
        arcpy.AddMessage(f"Using OPTIMIZED JoinField method for faster processing")
        arcpy.AddMessage(f"Note: ALL points will be retained (including those without pollution data)")
        
        for media_type in media_types:
            # Create single output GDB for this media type
            output_gdb_name = f"Chemical_{chem_num}_{media_type}_Joined.gdb"
            output_gdb = os.path.join(step2_dir, output_gdb_name)
            
            if not arcpy.Exists(output_gdb):
                arcpy.management.CreateFileGDB(step2_dir, output_gdb_name)
                arcpy.AddMessage(f"Created output geodatabase: {output_gdb_name}")
            
            media_full = "Media_1" if media_type == "M1" else "Media_2"
            processed_years = []
            skipped_years = []
            existing_years = []
            
            # Process each year
            for year in range(int(startyr), int(endyr) + 1):
                year_str = str(year)
                
                try:
                    start_time = time.time()
                    
                    # Look for aggregated CSV
                    standard_path = os.path.join(aggregated_dir, year_str, f"Chem_{chem_num}", "Aggregated_Media", f"{media_full}.csv")
                    nested_path = os.path.join(aggregated_dir, year_str, year_str, f"Chem_{chem_num}", "Aggregated_Media", f"{media_full}.csv")
                    
                    csv_path = None
                    if os.path.exists(standard_path):
                        csv_path = standard_path
                    elif os.path.exists(nested_path):
                        csv_path = nested_path
                    else:
                        arcpy.AddMessage(f"No {media_full} data for year {year_str}. Skipping.")
                        skipped_years.append(year_str)
                        continue
                    
                    # Define output feature class
                    output_fc_name = f"USA_{year_str}_{media_type}"
                    output_fc_path = os.path.join(output_gdb, output_fc_name)
                    
                    # Skip if already exists
                    if arcpy.Exists(output_fc_path):
                        arcpy.AddMessage(f"Output already exists for year {year_str}. Skipping.")
                        existing_years.append(year_str)
                        continue
                    
                    arcpy.AddMessage(f"Processing year {year_str}...")
                    
                    # STEP 1: Copy point grid to temporary feature class
                    temp_fc = os.path.join(output_gdb, f"temp_{year_str}")
                    arcpy.AddMessage(f"  [1/4] Copying point grid template...")
                    arcpy.management.CopyFeatures(point_grid, temp_fc)
                    
                    # Get initial point count
                    result = arcpy.management.GetCount(temp_fc)
                    initial_count = int(result.getOutput(0))
                    arcpy.AddMessage(f"    Copied {initial_count:,} points")
                    
                    copy_time = time.time() - start_time
                    arcpy.AddMessage(f"    Completed in {copy_time:.1f} seconds")
                    
                    # STEP 2: Convert CSV to geodatabase table for faster joins
                    csv_table_name = f"csv_temp_{year_str}"
                    csv_table = os.path.join(output_gdb, csv_table_name)
                    arcpy.AddMessage(f"  [2/4] Loading CSV into geodatabase...")
                    csv_start = time.time()
                    arcpy.conversion.TableToTable(csv_path, output_gdb, csv_table_name)
                    
                    # Get CSV record count
                    result = arcpy.management.GetCount(csv_table)
                    csv_count = int(result.getOutput(0))
                    arcpy.AddMessage(f"    Loaded {csv_count:,} pollution records")
                    
                    csv_time = time.time() - csv_start
                    arcpy.AddMessage(f"    Completed in {csv_time:.1f} seconds")
                    
                    # STEP 3: Use JoinField - much faster than layer joins
                    arcpy.AddMessage(f"  [3/4] Joining attributes (this is the slow part)...")
                    join_start = time.time()
                    arcpy.management.JoinField(
                        in_data=temp_fc,
                        in_field="UniqueID",
                        join_table=csv_table,
                        join_field=f"uID_{chem_num}",
                        fields=None  # Join all fields
                    )
                    join_time = time.time() - join_start
                    arcpy.AddMessage(f"    Completed in {join_time:.1f} seconds")
                    
                    # STEP 4: Report on join results (NO DELETION - ALL POINTS KEPT)
                    arcpy.AddMessage(f"  [4/4] Finalizing output (keeping all points)...")
                    
                    # Count how many points have data vs no data
                    conc_field = self._find_concentration_field(temp_fc, chem_num)
                    if conc_field:
                        matched_count = 0
                        null_count = 0
                        with arcpy.da.SearchCursor(temp_fc, [conc_field]) as cursor:
                            for row in cursor:
                                if row[0] is not None:
                                    matched_count += 1
                                else:
                                    null_count += 1
                        
                        arcpy.AddMessage(f"    Points with pollution data: {matched_count:,}")
                        arcpy.AddMessage(f"    Points without pollution data (NULL values): {null_count:,}")
                        arcpy.AddMessage(f"    Total points retained: {initial_count:,}")
                        arcpy.AddMessage(f"    Match rate: {(matched_count/initial_count)*100:.2f}%")
                    else:
                        arcpy.AddWarning(f"    Could not find concentration field - unable to count matched points")
                    
                    # Rename to final output (KEEPS ALL POINTS INCLUDING NULLS)
                    arcpy.management.Rename(temp_fc, output_fc_path)
                    
                    # Clean up temporary CSV table
                    arcpy.management.Delete(csv_table)
                    
                    # Verify final output
                    result = arcpy.management.GetCount(output_fc_path)
                    final_count = int(result.getOutput(0))
                    
                    total_time = time.time() - start_time
                    arcpy.AddMessage(f"✓ Created {output_fc_name} with {final_count:,} features (all grid points retained)")
                    arcpy.AddMessage(f"  Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
                    processed_years.append(year_str)
                    
                except Exception as e:
                    arcpy.AddWarning(f"Error processing year {year_str}: {str(e)}")
                    import traceback
                    arcpy.AddWarning(traceback.format_exc())
                    skipped_years.append(year_str)
                    # Clean up on error
                    try:
                        if arcpy.Exists(temp_fc):
                            arcpy.management.Delete(temp_fc)
                        if arcpy.Exists(csv_table):
                            arcpy.management.Delete(csv_table)
                    except:
                        pass
                    continue
            
            # Print summary
            arcpy.AddMessage(f"\n=== SUMMARY FOR {media_type} ===")
            arcpy.AddMessage(f"Processed: {', '.join(processed_years) if processed_years else 'None'}")
            arcpy.AddMessage(f"Skipped (errors): {', '.join(skipped_years) if skipped_years else 'None'}")
            arcpy.AddMessage(f"Skipped (existing): {', '.join(existing_years) if existing_years else 'None'}")
            
            joined_outputs[media_type] = output_gdb
        
        return joined_outputs

    def _run_step3_clip_data(self, joined_outputs, workspace_structure, clip_boundary, chem_num):
        """Step 3: Clip feature classes to boundary outline and reproject to boundary's coordinate system"""
        
        step3_dir = self._ensure_step_directory(workspace_structure['step3'])
        
        clipped_outputs = {}
        
        # Validate clip boundary
        if not arcpy.Exists(clip_boundary):
            raise ValueError(f"Clip boundary feature class not found: {clip_boundary}")
        
        desc = arcpy.Describe(clip_boundary)
        arcpy.AddMessage(f"Clipping to boundary: {desc.name}")
        arcpy.AddMessage(f"Boundary type: {desc.shapeType}")
        arcpy.AddMessage(f"Boundary coordinate system: {desc.spatialReference.name}")
        
        # SET OUTPUT COORDINATE SYSTEM TO MATCH CLIP BOUNDARY
        # This is critical - ensures all clipped outputs use the clip boundary's projection
        arcpy.env.outputCoordinateSystem = desc.spatialReference
        arcpy.AddMessage(f"✓ Setting output coordinate system to: {desc.spatialReference.name}")
        arcpy.AddMessage(f"  All clipped features will be reprojected to this coordinate system")
        
        for media_type, input_gdb in joined_outputs.items():
            # Create output GDB
            output_gdb_name = f"Chemical_{chem_num}_{media_type}_Clipped.gdb"
            output_gdb = os.path.join(step3_dir, output_gdb_name)
            
            if not arcpy.Exists(output_gdb):
                arcpy.management.CreateFileGDB(step3_dir, output_gdb_name)
                arcpy.AddMessage(f"Created clipped geodatabase: {output_gdb_name}")
            
            # Clip each feature class
            arcpy.env.workspace = input_gdb
            feature_classes = arcpy.ListFeatureClasses("USA_*")
            
            for fc in feature_classes:
                try:
                    output_name = f"{fc}_Clip"
                    output_path = os.path.join(output_gdb, output_name)
                    
                    if arcpy.Exists(output_path):
                        arcpy.AddMessage(f"Clipped feature class already exists: {output_name}")
                        continue
                    
                    # Check input coordinate system
                    fc_desc = arcpy.Describe(fc)
                    arcpy.AddMessage(f"Clipping {fc}...")
                    arcpy.AddMessage(f"  Input coordinate system: {fc_desc.spatialReference.name}")
                    
                    # Clip will automatically reproject due to environment setting
                    arcpy.analysis.Clip(
                        in_features=fc,
                        clip_features=clip_boundary,
                        out_feature_class=output_path,
                        cluster_tolerance=""
                    )
                    
                    result = arcpy.management.GetCount(output_path)
                    feature_count = int(result.getOutput(0))
                    
                    if feature_count > 0:
                        # Verify output coordinate system
                        out_desc = arcpy.Describe(output_path)
                        arcpy.AddMessage(f"✓ Successfully clipped {fc} -> {output_name} ({feature_count} features)")
                        arcpy.AddMessage(f"  Output coordinate system: {out_desc.spatialReference.name}")
                        
                        # Verify reprojection occurred if needed
                        if fc_desc.spatialReference.name != out_desc.spatialReference.name:
                            arcpy.AddMessage(f"  Note: Data was reprojected from {fc_desc.spatialReference.name}")
                    else:
                        arcpy.AddWarning(f"Clipping {fc} resulted in no features - check boundary overlap")
                    
                except Exception as e:
                    arcpy.AddWarning(f"Error clipping {fc}: {str(e)}")
                    continue
            
            clipped_outputs[media_type] = output_gdb
        
        # RESET ENVIRONMENT AFTER CLIPPING
        arcpy.env.outputCoordinateSystem = None
        arcpy.AddMessage("\n✓ Reset output coordinate system to default")
        arcpy.AddMessage("  Subsequent steps will use the clipped data's coordinate system")
        
        return clipped_outputs

    def _run_step4_null_to_zero_points(self, input_outputs, workspace_structure, chem_num):
        """Step 4: Replace NULL concentration values with 0 in point feature classes"""
        
        step4_dir = self._ensure_step_directory(workspace_structure['step4'])
        
        zero_outputs = {}
        
        for media_type, input_gdb in input_outputs.items():
            # Create output GDB
            output_gdb_name = f"Chemical_{chem_num}_{media_type}_Zero.gdb"
            output_gdb = os.path.join(step4_dir, output_gdb_name)
            
            if not arcpy.Exists(output_gdb):
                arcpy.management.CreateFileGDB(step4_dir, output_gdb_name)
                arcpy.AddMessage(f"Created null-corrected geodatabase: {output_gdb_name}")
            
            # Process each feature class
            arcpy.env.workspace = input_gdb
            feature_classes = arcpy.ListFeatureClasses()
            
            for fc in feature_classes:
                try:
                    # Create output name
                    if "_Clip" in fc:
                        output_name = fc.replace("_Clip", "_Zero")
                    else:
                        output_name = f"{fc}_Zero"
                    
                    output_path = os.path.join(output_gdb, output_name)
                    
                    if arcpy.Exists(output_path):
                        arcpy.AddMessage(f"Null-corrected feature class already exists: {output_name}")
                        continue
                    
                    # Find concentration field
                    conc_field = self._find_concentration_field(fc, chem_num)
                    if not conc_field:
                        arcpy.AddWarning(f"No concentration field found for {fc}. Skipping.")
                        continue
                    
                    arcpy.AddMessage(f"Replacing NULL values with 0 for: {fc}")
                    
                    # Copy feature class
                    arcpy.management.CopyFeatures(fc, output_path)
                    
                    # Count NULL vs non-NULL values
                    null_count = 0
                    non_null_count = 0
                    with arcpy.da.SearchCursor(output_path, [conc_field]) as cursor:
                        for row in cursor:
                            if row[0] is None:
                                null_count += 1
                            else:
                                non_null_count += 1
                    
                    # Replace NULL values with 0
                    with arcpy.da.UpdateCursor(output_path, [conc_field]) as cursor:
                        for row in cursor:
                            if row[0] is None:
                                row[0] = 0
                                cursor.updateRow(row)
                    
                    arcpy.AddMessage(f"✓ Successfully created: {output_name}")
                    arcpy.AddMessage(f"  Points with pollution data: {non_null_count:,}")
                    arcpy.AddMessage(f"  Points set to zero (NULL): {null_count:,}")
                    arcpy.AddMessage(f"  Total points: {null_count + non_null_count:,}")
                    arcpy.AddMessage(f"  All points now ready for IDW interpolation")
                    
                except Exception as e:
                    arcpy.AddWarning(f"Error processing {fc}: {str(e)}")
                    continue
            
            zero_outputs[media_type] = output_gdb
        
        return zero_outputs

    def _run_step5_idw_interpolation(self, input_outputs, workspace_structure, chem_num, cell_size, 
                                      reference_raster=None, clip_boundary=None, idw_power=2.0, idw_num_points=4):
        """Step 5: Perform IDW interpolation on point feature classes (NULLs already replaced with 0 in Step 4)"""
        
        from arcpy.sa import Idw, RadiusVariable
        
        step5_dir = self._ensure_step_directory(workspace_structure['step5'])
        
        # Determine cell size
        final_cell_size = self._determine_cell_size(cell_size, reference_raster)
        
        # Set extent from clip boundary if provided
        if clip_boundary and arcpy.Exists(clip_boundary):
            clip_desc = arcpy.Describe(clip_boundary)
            arcpy.env.extent = clip_desc.extent
            arcpy.AddMessage(f"Setting raster extent to clip boundary")
        
        # Set snap raster for alignment
        if reference_raster and arcpy.Exists(reference_raster):
            arcpy.env.snapRaster = reference_raster
            arcpy.AddMessage(f"Using reference raster for cell alignment")
        
        raster_outputs = {}
        
        for media_type, input_gdb in input_outputs.items():
            # Create output GDB
            output_gdb_name = f"Chemical_{chem_num}_{media_type}_IDW.gdb"
            output_gdb = os.path.join(step5_dir, output_gdb_name)
            
            if not arcpy.Exists(output_gdb):
                arcpy.management.CreateFileGDB(step5_dir, output_gdb_name)
                arcpy.AddMessage(f"Created IDW raster geodatabase: {output_gdb_name}")
            
            # Process each feature class
            arcpy.env.workspace = input_gdb
            feature_classes = arcpy.ListFeatureClasses()
            
            # Verify coordinate system is projected
            if feature_classes:
                first_fc_desc = arcpy.Describe(feature_classes[0])
                if first_fc_desc.spatialReference.type != "Projected":
                    arcpy.AddError(f"ERROR: Input data is not in a projected coordinate system!")
                    arcpy.AddError(f"Current coordinate system: {first_fc_desc.spatialReference.name}")
                    arcpy.AddError(f"Cannot create rasters from geographic (lat/long) coordinates.")
                    arcpy.AddError(f"Solution: Provide a clip boundary in Step 3 with a projected coordinate system.")
                    raise ValueError("IDW interpolation requires projected coordinate system")
                else:
                    arcpy.AddMessage(f"✓ Input data is properly projected: {first_fc_desc.spatialReference.name}")
            
            for fc in feature_classes:
                try:
                    # Create output name
                    if "_Zero" in fc:
                        output_name = fc.replace("_Zero", "_IDW")
                    elif "_Clip" in fc:
                        output_name = fc.replace("_Clip", "_IDW")
                    else:
                        output_name = f"{fc}_IDW"
                    
                    output_path = os.path.join(output_gdb, output_name)
                    
                    if arcpy.Exists(output_path):
                        arcpy.AddMessage(f"IDW raster already exists: {output_name}")
                        continue
                    
                    # Find concentration field
                    conc_field = self._find_concentration_field(fc, chem_num)
                    if not conc_field:
                        arcpy.AddWarning(f"No concentration field found for {fc}. Skipping.")
                        continue
                    
                    arcpy.AddMessage(f"Running IDW interpolation on {fc}...")
                    arcpy.AddMessage(f"  Using field: {conc_field}")
                    arcpy.AddMessage(f"  Cell size: {final_cell_size}")
                    arcpy.AddMessage(f"  Power: {idw_power}")
                    arcpy.AddMessage(f"  Number of points: {idw_num_points}")
                    
                    start_time = time.time()
                    
                    # Perform IDW interpolation
                    # All points have valid values (0 or positive) from Tool 4
                    idw_raster = Idw(
                        in_point_features=fc,
                        z_field=conc_field,
                        cell_size=final_cell_size,
                        power=idw_power,
                        search_radius=RadiusVariable(idw_num_points)
                    )
                    
                    # Save the raster
                    idw_raster.save(output_path)
                    
                    elapsed_time = time.time() - start_time
                    
                    # Verify output
                    desc = arcpy.Describe(output_path)
                    arcpy.AddMessage(f"✓ Successfully created IDW raster: {output_name}")
                    arcpy.AddMessage(f"  Cell size: {desc.meanCellWidth}")
                    arcpy.AddMessage(f"  Coordinate system: {desc.spatialReference.name}")
                    arcpy.AddMessage(f"  Processing time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
                    
                except Exception as e:
                    arcpy.AddWarning(f"Error performing IDW on {fc}: {str(e)}")
                    import traceback
                    arcpy.AddWarning(traceback.format_exc())
                    continue
            
            raster_outputs[media_type] = output_gdb
        
        # Reset environment
        arcpy.env.extent = None
        arcpy.env.snapRaster = None
        
        return raster_outputs

    def _run_step6_scale_rasters(self, raster_outputs, workspace_structure, chem_num):
        """Step 6: Multiply rasters by 1 billion"""
        
        step6_dir = self._ensure_step_directory(workspace_structure['step6'])
        
        scaled_outputs = {}
        
        for media_type, input_gdb in raster_outputs.items():
            output_gdb_name = f"Orig_{chem_num}_{media_type}_Bil.gdb"
            output_gdb = os.path.join(step6_dir, output_gdb_name)
            
            if not arcpy.Exists(output_gdb):
                arcpy.management.CreateFileGDB(step6_dir, output_gdb_name)
                arcpy.AddMessage(f"Created scaled raster geodatabase: {output_gdb_name}")
            
            arcpy.env.workspace = input_gdb
            rasters = arcpy.ListRasters()
            
            for raster in rasters:
                try:
                    # Extract year
                    year = self._extract_year_from_name(raster)
                    if not year:
                        arcpy.AddWarning(f"Could not extract year from: {raster}")
                        continue
                    
                    output_raster_name = f"Chem_{chem_num}_{year}_{media_type}_Bil"
                    output_raster_path = os.path.join(output_gdb, output_raster_name)
                    
                    if arcpy.Exists(output_raster_path):
                        arcpy.AddMessage(f"Scaled raster already exists: {output_raster_name}")
                        continue
                    
                    arcpy.AddMessage(f"Scaling {raster} by 1,000,000,000...")
                    
                    scaled_raster = arcpy.sa.Times(raster, 1000000000)
                    scaled_raster.save(output_raster_path)
                    
                    arcpy.AddMessage(f"Successfully created: {output_raster_name}")
                    arcpy.AddMessage(f"  (Preserves 3 decimal places of original precision)")
                    
                except Exception as e:
                    arcpy.AddWarning(f"Error scaling raster {raster}: {str(e)}")
                    continue
            
            scaled_outputs[media_type] = output_gdb
        
        return scaled_outputs

    def _run_step7_convert_to_int(self, scaled_raster_outputs, workspace_structure, chem_num):
        """Step 7: Convert rasters to integer data type"""
        
        step7_dir = self._ensure_step_directory(workspace_structure['step7'])
        int_folder = self._ensure_step_directory(os.path.join(step7_dir, "INT"))
        
        int_outputs = {}
        
        for media_type, input_gdb in scaled_raster_outputs.items():
            output_gdb_name = f"Orig_{chem_num}_{media_type}_INT.gdb"
            output_gdb = os.path.join(int_folder, output_gdb_name)
            
            if not arcpy.Exists(output_gdb):
                arcpy.management.CreateFileGDB(int_folder, output_gdb_name)
                arcpy.AddMessage(f"Created integer raster geodatabase: {output_gdb_name}")
            
            arcpy.env.workspace = input_gdb
            rasters = arcpy.ListRasters()
            
            for raster in rasters:
                try:
                    year = self._extract_year_from_name(raster)
                    if not year:
                        arcpy.AddWarning(f"Could not extract year from: {raster}")
                        continue
                    
                    output_raster_name = f"Chem_{chem_num}_{year}_{media_type}_INT"
                    output_raster_path = os.path.join(output_gdb, output_raster_name)
                    
                    if arcpy.Exists(output_raster_path):
                        arcpy.AddMessage(f"Integer raster already exists: {output_raster_name}")
                        continue
                    
                    arcpy.AddMessage(f"Converting {raster} to integer...")
                    
                    int_raster = arcpy.sa.Int(raster)
                    int_raster.save(output_raster_path)
                    
                    arcpy.AddMessage(f"Successfully converted: {output_raster_name}")
                    
                except Exception as e:
                    arcpy.AddWarning(f"Error converting {raster} to integer: {str(e)}")
                    continue
            
            int_outputs[media_type] = output_gdb
        
        return int_outputs

    def _run_step8_add_media_rasters(self, int_raster_outputs, workspace_structure, chem_num, startyr, endyr):
        """Step 8: Add M1 and M2 integer rasters together"""
        
        step8_dir = self._ensure_step_directory(workspace_structure['step8'])
        
        combined_gdb_name = f"Combined_{chem_num}_M1M2.gdb"
        combined_gdb = os.path.join(step8_dir, combined_gdb_name)
        
        if not arcpy.Exists(combined_gdb):
            arcpy.management.CreateFileGDB(step8_dir, combined_gdb_name)
            arcpy.AddMessage(f"Created combined raster geodatabase: {combined_gdb_name}")
        
        m1_gdb = int_raster_outputs.get('M1')
        m2_gdb = int_raster_outputs.get('M2')
        
        if not m1_gdb or not m2_gdb:
            arcpy.AddWarning("Cannot combine rasters - missing M1 or M2 integer data")
            return {}
        
        arcpy.env.workspace = m1_gdb
        m1_rasters = arcpy.ListRasters("*", "ALL")
        
        arcpy.env.workspace = m2_gdb
        m2_rasters = arcpy.ListRasters("*", "ALL")
        
        combined_count = 0
        
        for year in range(int(startyr), int(endyr) + 1):
            year_str = str(year)
            
            m1_rasters_for_year = [r for r in m1_rasters if f"_{year_str}_" in r]
            m2_rasters_for_year = [r for r in m2_rasters if f"_{year_str}_" in r]
            
            if len(m1_rasters_for_year) > 0 and len(m2_rasters_for_year) > 0:
                try:
                    m1_raster_name = m1_rasters_for_year[0]
                    m2_raster_name = m2_rasters_for_year[0]
                    
                    m1_raster_path = os.path.join(m1_gdb, m1_raster_name)
                    m2_raster_path = os.path.join(m2_gdb, m2_raster_name)
                    
                    output_raster_name = f"USA_{year_str}_M1M2"
                    output_raster_path = os.path.join(combined_gdb, output_raster_name)
                    
                    if arcpy.Exists(output_raster_path):
                        arcpy.AddMessage(f"Combined raster already exists: {output_raster_name}")
                        combined_count += 1
                        continue
                    
                    arcpy.AddMessage(f"Combining M1 and M2 for year {year_str}...")
                    
                    combined_raster = arcpy.sa.Plus(
                        arcpy.sa.Raster(m1_raster_path),
                        arcpy.sa.Raster(m2_raster_path)
                    )
                    combined_raster.save(output_raster_path)
                    
                    arcpy.AddMessage(f"Successfully created: {output_raster_name}")
                    combined_count += 1
                    
                except Exception as e:
                    arcpy.AddWarning(f"Error combining rasters for year {year_str}: {str(e)}")
                    continue
        
        arcpy.AddMessage(f"Combined raster creation complete: {combined_count} rasters created")
        
        return {'M1M2_Combined': combined_gdb}

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _determine_cell_size(self, cell_size_param, reference_raster):
        """Determine cell size from parameter or reference raster"""
        
        if reference_raster and arcpy.Exists(reference_raster):
            try:
                desc = arcpy.Describe(reference_raster)
                ref_cell_size = desc.meanCellWidth
                arcpy.AddMessage(f"Using cell size from reference raster: {ref_cell_size}")
                return ref_cell_size
            except Exception as e:
                arcpy.AddWarning(f"Could not read reference raster cell size: {str(e)}")
        
        arcpy.AddMessage(f"Using parameter cell size: {cell_size_param}")
        return cell_size_param

    def _find_concentration_field(self, feature_class, chem_num):
        """Find the concentration field in the feature class"""
        
        fields = [f.name for f in arcpy.ListFields(feature_class)]
        
        # Try different possible field names
        possible_fields = [
            f"conc_{chem_num}",
            f"csv_temp_conc_{chem_num}",  # From TableToTable join
            f"Aggregated_Media_1_conc_{chem_num}",
            f"Aggregated_Media_2_conc_{chem_num}",
        ]
        
        # Also look for any field containing 'conc' and the chemical number
        for field in fields:
            if 'conc' in field.lower() and chem_num in field:
                if field not in possible_fields:
                    possible_fields.append(field)
        
        # Find first matching field
        for field_name in possible_fields:
            if field_name in fields:
                return field_name
        
        arcpy.AddWarning(f"Available fields in {feature_class}: {', '.join(fields[:20])}")
        return None

    def _extract_year_from_name(self, name):
        """Extract 4-digit year from raster/feature class name"""
        
        parts = name.split("_")
        for part in parts:
            if part.isdigit() and len(part) == 4:
                return part
        return None

    def postExecute(self, parameters):
        return

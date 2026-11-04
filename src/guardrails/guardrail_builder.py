import os
import glob
import pandas as pd
import json
import logging
import math

class GuardrailBuilder:
    """
    Parses raw LOINC nomenclature CSV files to build the Biological Guardrails JSON dataset.
    This module uses heuristics based on SCALE_TYP and EXAMPLE_UNITS to enforce 
    physical realism bounds (e.g., preventing negative weights).
    """

    def __init__(self, raw_data_dir: str = "datasets/guardrails_raw", output_file: str = "datasets/biological_guardrails.json"):
        self.raw_data_dir = raw_data_dir
        self.output_file = output_file
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("IRAP_GuardrailBuilder")

    def build(self):
        self.logger.info(f"Scanning for CSV files in {self.raw_data_dir}...")
        csv_files = glob.glob(os.path.join(self.raw_data_dir, "*.csv"))
        
        if not csv_files:
            self.logger.error("No CSV files found in the specified directory.")
            return

        self.logger.info(f"Found {len(csv_files)} files. Consolidating...")

        dfs = []
        cols_to_keep = ['LOINC_NUM', 'COMPONENT', 'SCALE_TYP', 'EXAMPLE_UNITS']
        
        for file in csv_files:
            try:
                # Read CSV, sometimes encoding issues occur so we use utf-8 with replacement or latin1
                df = pd.read_csv(file, usecols=lambda c: c in cols_to_keep, encoding='latin1')
                dfs.append(df)
            except Exception as e:
                self.logger.warning(f"Failed to read {file}: {e}")

        if not dfs:
            self.logger.error("No data could be extracted.")
            return

        master_df = pd.concat(dfs, ignore_index=True)
        self.logger.info(f"Successfully consolidated {len(master_df)} raw LOINC entries.")

        # Clean and process
        guardrails = {}
        processed_count = 0
        heuristic_count = 0

        # We drop duplicates based on COMPONENT to create a unique lookup dictionary
        # Realistically, multiple LOINC codes can map to one component name, 
        # but for column name matching, one definitive definition is best.
        unique_components = master_df.drop_duplicates(subset=['COMPONENT'])

        for _, row in unique_components.iterrows():
            component = str(row.get('COMPONENT', '')).strip().lower()
            if not component or component == 'nan':
                continue

            scale_typ = str(row.get('SCALE_TYP', '')).strip()
            units = str(row.get('EXAMPLE_UNITS', '')).strip()
            loinc = str(row.get('LOINC_NUM', '')).strip()

            if scale_typ == 'nan':
                scale_typ = None
            if units == 'nan':
                units = None

            guardrail_entry = {
                "loinc_num": loinc,
                "scale_typ": scale_typ,
                "expected_units": units,
                "min_value": None,
                "max_value": None
            }

            # -----------------------------------------------------
            # HEURISTIC RANGE GENERATOR
            # -----------------------------------------------------
            applied_heuristic = False

            if units:
                # Percentage Heuristic
                if '%' in units:
                    guardrail_entry["min_value"] = 0.0
                    guardrail_entry["max_value"] = 100.0
                    applied_heuristic = True
                
                # pH Heuristic
                elif 'pH' in units or component == 'ph':
                    guardrail_entry["min_value"] = 0.0
                    guardrail_entry["max_value"] = 14.0
                    applied_heuristic = True

            if scale_typ == 'Qn':
                # Quantitative Physical Metrics Heuristic (Weight, Pressure, Volume, Time, Count)
                # You cannot have a negative weight, length, volume, cell count, or blood pressure.
                # Only specific things (like temperature in Celsius/Fahrenheit, or relative changes) can be negative.
                non_negative_indicators = ['kg', 'g', 'mg', 'L', 'mL', 'dL', 'mm Hg', 'cm H2O', 'beats/min', '/uL', 'm', 'cm', 'mm', 's', 'min', 'h']
                
                # If the unit matches a physical non-negative metric, or if it's a concentration (which is non-negative)
                if units and any(ind in units for ind in non_negative_indicators):
                    if guardrail_entry["min_value"] is None:
                        guardrail_entry["min_value"] = 0.0
                        applied_heuristic = True
                
                # General safety net: Most Qn biological measurements are strictly >= 0 
                # (e.g. cholesterol, hemoglobin, cell counts).
                if guardrail_entry["min_value"] is None and ('temp' not in component and 'celsius' not in str(units).lower()):
                    guardrail_entry["min_value"] = 0.0
                    applied_heuristic = True

            guardrails[component] = guardrail_entry
            processed_count += 1
            if applied_heuristic:
                heuristic_count += 1

        # Save to JSON
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(guardrails, f, indent=4)

        self.logger.info(f"Guardrail building complete!")
        self.logger.info(f"Total Unique Components Processed: {processed_count}")
        self.logger.info(f"Heuristics Successfully Applied to: {heuristic_count} components")
        self.logger.info(f"Saved to: {self.output_file}")


if __name__ == "__main__":
    builder = GuardrailBuilder()
    builder.build()

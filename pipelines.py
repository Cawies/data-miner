# External libraries
from sklearn.pipeline import Pipeline

# Internal modules
from processing import preprocessors as pp
from config import config


"""
This module gathers the preprocessing transformers and organizes
them into sequential steps within relevant pipelines.
"""

cleaning_pipeline = Pipeline(
    [
        (
            "one_head_per_household",
            pp.OneHeadHouseholds(),
        ),
        (
            "align_household_targets",
            pp.HouseholdTargetAligner(),
        ),
        (
            "convert_related_binary_to_ordinal",
            pp.BinaryToOrdinal(),
        ),
        (
            "convert_strings_to_numerical_values",
            pp.StringsToNumerical(),
        ),
        (
            "correct_adhoc_report_flags",
            pp.ReportAdHocCorrections(),
        ),
        (
            "aggregate_households",
            pp.AggregateHouseholds(variables=config.INDIVIDUAL_LEVEL),
        ),
        (
            "export_clean_data",
            pp.ExportCleanData(output_folder=config.OUTPUT_DIR)
        )
    ]
)

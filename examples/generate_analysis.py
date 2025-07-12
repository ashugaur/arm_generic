# %% Dry test

## Dependencies
import os
from pathlib import Path
from datetime import datetime, timezone
from arm_generic.functions import (
    simulate_combinations,
    AssociationAnalyzer,
)

working_dir = Path("C:/my_disk/____tmp/arm_generic")
Path(working_dir).mkdir(parents=True, exist_ok=True)
os.chdir(working_dir)


combination_categories = {
    "main_dr": {
        "core": ["Cardiologist", "Surgeon"],
        "optional": ["Immunologist", "Psychiatrist"],
    },
    "Endocrine": {
        "core": ["Endocrinologist", "Diabetician"],
        "optional": ["Ayurveda", "Homeopathy"],
    },
    "Opthalmology": {
        "core": ["Opthalmologist", "Eye surgeon"],
        "optional": ["Eye care", "Eye testing"],
    },
    "Kidney": {
        "core": ["Nephrologist", "Liver and Kidney"],
        "optional": ["Pancreas", "Stomach"],
    },
}

df = simulate_combinations(combination_categories, 10000)
print(df)


## Automatically analyze ALL  pairs with enhanced export
print("\n🚀 Creating Complete  Relationships Matrix:")
print("=" * 60)


## Initialize analyzer
analyzer = AssociationAnalyzer(df)

# Step 1: Preprocess data
transactions = analyzer.preprocess_data()
print("\n📋 Transaction Matrix Shape: {transactions.shape}")
print("Sample of transaction matrix:")
print(transactions.head())

# Step 2: Get basic statistics
analyzer.get_item_statistics()

# Step 3: Find frequent itemsets
# Start with low min_support since we have small dataset
frequent_itemsets = analyzer.find_frequent_itemsets(min_support=0.2)

if frequent_itemsets is not None:
    print("\n📊 Top 10 Frequent Itemsets:")
    print(frequent_itemsets.nlargest(10, "support"))

# Step 4: Generate association rules
rules = analyzer.generate_association_rules(metric="confidence", min_threshold=0.5)

all_pairs_matrix = analyzer.create_all_pairs_relationship_matrix(
    export_to_excel=True,
    filename=f"complete__relationships_matrix_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.xlsx",
)


## Self run
"""
cd .\examples\
uv run .\generate_analysis.py
"""

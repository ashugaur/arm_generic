# %% Likelihood of combinations

## Dependencies
import pandas as pd
import random
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


## simulate_combinations
def simulate_combinations(combination_categories, num_records):
    # Define therapeutic categories and their associated combinations

    data = {"id": [], "combination": []}

    for _ in range(num_records):
        id = random.randint(1000, 9999)

        # Choose a primary therapeutic category
        category = random.choice(list(combination_categories.keys()))

        # Core combinations for the chosen category
        core_combinations = combination_categories[category]["core"]

        # Add optional combinations with 30% probability each
        additional_combinations = []
        if random.random() < 0.3:
            additional_combinations.extend(
                random.sample(
                    combination_categories[category]["optional"],
                    min(1, len(combination_categories[category]["optional"])),
                )
            )
        if random.random() < 0.2:
            additional_combinations.extend(
                random.sample(
                    [
                        item
                        for cat in combination_categories.keys()
                        for item in combination_categories[cat]["core"]
                    ],
                    min(
                        1,
                        len(
                            [
                                item
                                for cat in combination_categories.keys()
                                for item in combination_categories[cat]["core"]
                            ]
                        ),
                    ),
                )
            )

        all_combinations = core_combinations + additional_combinations

        data["id"].append(id)
        data["combination"].append(", ".join(all_combinations))

    df = pd.DataFrame(data)
    return df


# Example usage
if __name__ == "__main__":
    df = simulate_combinations(1000)
    print(df)
    # df.to_clipboard(index=False)


# %% Model


## AssociationAnalyzer
class AssociationAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        self.transactions = None
        self.frequent_itemsets = None
        self.rules = None

    def preprocess_data(self):
        """Convert item combinations to transaction format"""
        print("🔄 Preprocessing data...")

        # Split item combinations into lists
        item_lists = []
        for idx, row in self.df.iterrows():
            combinations = [item.strip() for item in row["combination"].split(",")]
            item_lists.append(combinations)

        # Convert to binary matrix using TransactionEncoder
        te = TransactionEncoder()
        te_ary = te.fit(item_lists).transform(item_lists)
        self.transactions = pd.DataFrame(te_ary, columns=te.columns_)

        print(
            f"✅ Data preprocessed: {len(self.transactions)} s, {len(self.transactions.columns)} unique combinations"
        )
        return self.transactions

    def find_frequent_itemsets(self, min_support=0.1):
        """Find frequent itemsets using Apriori algorithm"""
        print(f"🔍 Finding frequent itemsets with min_support={min_support}...")

        self.frequent_itemsets = apriori(
            self.transactions, min_support=min_support, use_colnames=True
        )

        if len(self.frequent_itemsets) == 0:
            print("⚠️ No frequent itemsets found. Try lowering min_support.")
            return None

        print(f"✅ Found {len(self.frequent_itemsets)} frequent itemsets")
        return self.frequent_itemsets

    def generate_association_rules(self, metric="confidence", min_threshold=0.5):
        """Generate association rules from frequent itemsets"""
        print(f"📊 Generating association rules with min_{metric}={min_threshold}...")

        if self.frequent_itemsets is None or len(self.frequent_itemsets) == 0:
            print(
                "❌ No frequent itemsets available. Run find_frequent_itemsets() first."
            )
            return None

        try:
            self.rules = association_rules(
                self.frequent_itemsets, metric=metric, min_threshold=min_threshold
            )
            print(f"✅ Generated {len(self.rules)} association rules")
            return self.rules
        except ValueError as e:
            print(f"❌ Error generating rules: {e}")
            print("💡 Try lowering the min_threshold or min_support values")
            return None

    def analyze_item_cooccurrence(self, focus_item=None):
        """Analyze co-occurrence patterns for a specific item"""
        if self.rules is None or len(self.rules) == 0:
            print("❌ No rules available for analysis")
            return None

        if focus_item:
            # Filter rules where focus_item is in antecedents
            focus_rules = self.rules[
                self.rules["antecedents"].apply(lambda x: focus_item in x)
            ]
            print(f"\n🎯 Analysis for {focus_item}:")
            print(
                f"Found {len(focus_rules)} rules where {focus_item} is the antecedent"
            )
            return focus_rules
        else:
            return self.rules

    def display_top_rules(self, n=10, sort_by="confidence"):
        """Display top N association rules"""
        if self.rules is None or len(self.rules) == 0:
            print("❌ No rules to display")
            return

        print(f"\n🏆 Top {n} Association Rules (sorted by {sort_by}):")
        print("=" * 80)

        top_rules = self.rules.nlargest(n, sort_by)

        for idx, rule in top_rules.iterrows():
            antecedents = ", ".join(list(rule["antecedents"]))
            consequents = ", ".join(list(rule["consequents"]))

            print(f"\n📋 Rule {idx + 1}:")
            print(f"   If  takes: {antecedents}")
            print(f"   Then  also takes: {consequents}")
            print(
                f"   📊 Support: {rule['support']:.3f} ({rule['support'] * len(self.transactions):.0f} s)"
            )
            print(
                f"   📊 Confidence: {rule['confidence']:.3f} ({rule['confidence'] * 100:.1f}%)"
            )
            print(f"   📊 Lift: {rule['lift']:.3f}")
            print(f"   📊 Conviction: {rule['conviction']:.3f}")

    def get_item_statistics(self):
        """Get basic statistics about item usage"""
        print("\n📈 item Usage Statistics:")
        print("=" * 50)

        # Count item frequencies
        item_counts = self.transactions.sum().sort_values(ascending=False)
        total_s = len(self.transactions)

        print(f"Total s: {total_s}")
        print(f"Total unique combinations: {len(item_counts)}")
        print("\nTop 10 most prescribed combinations:")

        for item, count in item_counts.head(10).items():
            percentage = (count / total_s) * 100
            print(f"  {item}: {count} s ({percentage:.1f}%)")

    def plot_support_confidence(self):
        """Plot support vs confidence scatter plot"""
        if self.rules is None or len(self.rules) == 0:
            print("❌ No rules to plot")
            return

        plt.figure(figsize=(10, 6))
        plt.scatter(
            self.rules["support"],
            self.rules["confidence"],
            c=self.rules["lift"],
            cmap="viridis",
            alpha=0.7,
        )
        plt.colorbar(label="Lift")
        plt.xlabel("Support")
        plt.ylabel("Confidence")
        plt.title("Association Rules: Support vs Confidence (colored by Lift)")
        plt.grid(True, alpha=0.3)
        plt.show()

    def create_comprehensive_relationship_table(
        self, min_confidence=0.0, min_support=0.0, min_lift=1.0
    ):
        """Create comprehensive table of all item relationships"""
        if self.rules is None or len(self.rules) == 0:
            print("❌ No rules available for creating relationship table")
            return None

        print("📋 Creating comprehensive relationship table...")
        print(
            f"   Filters: Confidence≥{min_confidence}, Support≥{min_support}, Lift≥{min_lift}"
        )

        # Filter rules based on criteria
        filtered_rules = self.rules[
            (self.rules["confidence"] >= min_confidence)
            & (self.rules["support"] >= min_support)
            & (self.rules["lift"] >= min_lift)
        ].copy()

        if len(filtered_rules) == 0:
            print(
                "⚠️ No rules meet the specified criteria. Try lowering the thresholds."
            )
            return None

        # Create comprehensive table
        relationship_table = []

        for idx, rule in filtered_rules.iterrows():
            # Extract antecedent and consequent combinations
            antecedents = list(rule["antecedents"])
            consequents = list(rule["consequents"])

            # Handle multiple combinations in antecedents/consequents
            antecedent_str = (
                " + ".join(antecedents) if len(antecedents) > 1 else antecedents[0]
            )
            consequent_str = (
                " + ".join(consequents) if len(consequents) > 1 else consequents[0]
            )

            # Calculate additional metrics
            total_s = len(self.transactions)
            s_with_both = int(rule["support"] * total_s)
            s_with_antecedent = int(
                (rule["support"] / rule["confidence"]) * total_s
            )

            relationship_table.append(
                {
                    "item_A (If)": antecedent_str,
                    "item_B (Then)": consequent_str,
                    "Support": round(rule["support"], 4),
                    "Confidence": round(rule["confidence"], 4),
                    "Confidence_%": round(rule["confidence"] * 100, 1),
                    "Lift": round(rule["lift"], 3),
                    "Conviction": round(rule["conviction"], 3),
                    "Leverage": round(rule["leverage"], 4),
                    "Zhang_Metric": round(rule["zhangs_metric"], 4),
                    "s_Both": s_with_both,
                    "s_item_A": s_with_antecedent,
                    "Rule_Strength": self._categorize_rule_strength(
                        rule["confidence"], rule["lift"]
                    ),
                    "Clinical_Priority": self._assign_priority(
                        rule["support"], rule["confidence"], rule["lift"]
                    ),
                }
            )

        comprehensive_df = pd.DataFrame(relationship_table)

        # Sort by confidence descending, then by lift descending
        comprehensive_df = comprehensive_df.sort_values(
            ["Confidence", "Lift", "Support"], ascending=[False, False, False]
        ).reset_index(drop=True)

        print(
            f"✅ Created comprehensive table with {len(comprehensive_df)} item relationships"
        )
        return comprehensive_df

    def _categorize_rule_strength(self, confidence, lift):
        """Categorize rule strength based on confidence and lift"""
        if confidence >= 0.8 and lift >= 2.0:
            return "Very Strong"
        elif confidence >= 0.6 and lift >= 1.5:
            return "Strong"
        elif confidence >= 0.4 and lift >= 1.2:
            return "Moderate"
        else:
            return "Weak"

    def _assign_priority(self, support, confidence, lift):
        """Assign clinical priority based on metrics"""
        # Weighted score considering all three metrics
        score = (support * 0.3) + (confidence * 0.5) + ((lift - 1) * 0.2)

        if score >= 0.6:
            return "High"
        elif score >= 0.3:
            return "Medium"
        else:
            return "Low"

    def export_relationship_table(
        self, comprehensive_df, filename="item_relationships.xlsx"
    ):
        """Export comprehensive relationship table to Excel and Parquet"""
        if comprehensive_df is None:
            print("❌ No relationship table to export")
            return

        # Always save parquet first
        parquet_filename = filename.replace(".xlsx", ".parquet")
        try:
            comprehensive_df.to_parquet(parquet_filename, index=False)
            print(f"✅ Saved parquet file: {parquet_filename}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save parquet file: {e}")

        # Check if data exceeds Excel row limit
        excel_row_limit = 1000000
        main_data_oversized = len(comprehensive_df) > excel_row_limit

        try:
            # Create Excel writer with multiple sheets
            with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                # Handle main data sheet based on size
                if main_data_oversized:
                    # Create a note sheet instead of the full data
                    note_df = pd.DataFrame(
                        {
                            "NOTICE": [
                                "Main data exceeds Excel row limit (1M+ rows)",
                                f"Total rows: {len(comprehensive_df):,}",
                                f"Complete data available in: {parquet_filename}",
                                "Use pandas.read_parquet() to load the full dataset",
                                "",
                                "Sample of first 1000 rows shown below:",
                            ]
                        }
                    )

                    # Write the notice
                    note_df.to_excel(
                        writer, sheet_name="All_Relationships", index=False, startrow=0
                    )

                    # Add sample data below the notice
                    sample_data = comprehensive_df.head(1000)
                    sample_data.to_excel(
                        writer, sheet_name="All_Relationships", index=False, startrow=8
                    )

                    print(
                        f"⚠️  Main data ({len(comprehensive_df):,} rows) exceeds Excel limit"
                    )
                    print("   📝 Added notice in Excel sheet pointing to parquet file")
                else:
                    # Normal case - write all data to Excel
                    comprehensive_df.to_excel(
                        writer, sheet_name="All_Relationships", index=False
                    )

                # High priority relationships (usually smaller)
                high_priority = comprehensive_df[
                    comprehensive_df["Clinical_Priority"] == "High"
                ]
                if len(high_priority) <= excel_row_limit:
                    high_priority.to_excel(
                        writer, sheet_name="High_Priority", index=False
                    )
                else:
                    high_priority_parquet = filename.replace(
                        ".xlsx", "_high_priority.parquet"
                    )
                    high_priority.to_parquet(high_priority_parquet, index=False)

                    note_df = pd.DataFrame(
                        {
                            "NOTICE": [
                                "High priority data exceeds Excel row limit",
                                f"Total rows: {len(high_priority):,}",
                                f"Complete data available in: {high_priority_parquet}",
                            ]
                        }
                    )
                    note_df.to_excel(writer, sheet_name="High_Priority", index=False)

                # Strong relationships
                strong_rules = comprehensive_df[
                    comprehensive_df["Rule_Strength"].isin(["Very Strong", "Strong"])
                ]
                if len(strong_rules) <= excel_row_limit:
                    strong_rules.to_excel(
                        writer, sheet_name="Strong_Rules", index=False
                    )
                else:
                    strong_rules_parquet = filename.replace(
                        ".xlsx", "_strong_rules.parquet"
                    )
                    strong_rules.to_parquet(strong_rules_parquet, index=False)

                    note_df = pd.DataFrame(
                        {
                            "NOTICE": [
                                "Strong rules data exceeds Excel row limit",
                                f"Total rows: {len(strong_rules):,}",
                                f"Complete data available in: {strong_rules_parquet}",
                            ]
                        }
                    )
                    note_df.to_excel(writer, sheet_name="Strong_Rules", index=False)

                # Top relationships by confidence (limit to top 1000 for Excel)
                top_confidence = comprehensive_df.nlargest(
                    min(1000, len(comprehensive_df)), "Confidence"
                )
                top_confidence.to_excel(
                    writer, sheet_name="Top_Confidence", index=False
                )

                # Summary statistics
                summary_stats = self._create_summary_stats(comprehensive_df)
                summary_stats.to_excel(writer, sheet_name="Summary_Stats", index=False)

                # Save individual parquet files for filtered data
                high_priority.to_parquet(
                    filename.replace(".xlsx", "_high_priority.parquet"), index=False
                )
                strong_rules.to_parquet(
                    filename.replace(".xlsx", "_strong_rules.parquet"), index=False
                )
                top_confidence.to_parquet(
                    filename.replace(".xlsx", "_top_confidence.parquet"), index=False
                )
                summary_stats.to_parquet(
                    filename.replace(".xlsx", "_summary_stats.parquet"), index=False
                )

            print(f"✅ Exported comprehensive relationship table to: {filename}")
            print(
                "   📊 Excel sheets created: All_Relationships, High_Priority, Strong_Rules, Top_Confidence, Summary_Stats"
            )
            print("   📁 Parquet files created:")
            print(f"      • {parquet_filename}: Complete dataset")
            print(
                f"      • {filename.replace('.xlsx', '_high_priority.parquet')}: High priority relationships"
            )
            print(
                f"      • {filename.replace('.xlsx', '_strong_rules.parquet')}: Strong rules"
            )
            print(
                f"      • {filename.replace('.xlsx', '_top_confidence.parquet')}: Top confidence relationships"
            )
            print(
                f"      • {filename.replace('.xlsx', '_summary_stats.parquet')}: Summary statistics"
            )

        except Exception as e:
            print(f"❌ Error exporting to Excel: {e}")
            print("💡 Saving as CSV and parquet instead...")
            comprehensive_df.to_csv(filename.replace(".xlsx", ".csv"), index=False)
            print(f"✅ Saved as CSV: {filename.replace('.xlsx', '.csv')}")

    def _create_summary_stats(self, comprehensive_df):
        """Create summary statistics for the relationship table"""
        summary_data = []

        # Overall statistics
        total_relationships = len(comprehensive_df)
        avg_confidence = comprehensive_df["Confidence"].mean()
        avg_lift = comprehensive_df["Lift"].mean()
        avg_support = comprehensive_df["Support"].mean()

        summary_data.append(
            {
                "Metric": "Total item Relationships",
                "Value": total_relationships,
                "Description": "Total number of significant item associations found",
            }
        )

        summary_data.append(
            {
                "Metric": "Average Confidence",
                "Value": f"{avg_confidence:.3f} ({avg_confidence * 100:.1f}%)",
                "Description": "Average likelihood of consequent given antecedent",
            }
        )

        summary_data.append(
            {
                "Metric": "Average Lift",
                "Value": f"{avg_lift:.3f}",
                "Description": "Average lift above random chance",
            }
        )

        summary_data.append(
            {
                "Metric": "Average Support",
                "Value": f"{avg_support:.3f}",
                "Description": "Average frequency of item combinations",
            }
        )

        # Rule strength distribution
        strength_counts = comprehensive_df["Rule_Strength"].value_counts()
        for strength, count in strength_counts.items():
            summary_data.append(
                {
                    "Metric": f"{strength} Rules",
                    "Value": count,
                    "Description": f"Number of {strength.lower()} association rules",
                }
            )

        # Priority distribution
        priority_counts = comprehensive_df["Clinical_Priority"].value_counts()
        for priority, count in priority_counts.items():
            summary_data.append(
                {
                    "Metric": f"{priority} Priority Rules",
                    "Value": count,
                    "Description": f"Number of {priority.lower()} priority rules",
                }
            )

        return pd.DataFrame(summary_data)

    def create_all_pairs_relationship_matrix(
        self, export_to_excel=True, filename="all_item_pairs_analysis.xlsx"
    ):
        """Create comprehensive analysis of ALL possible item pairs"""
        if self.transactions is None:
            print("❌ No transaction data available")
            return None

        print("🔄 Calculating relationships for ALL item pairs...")

        # Get all unique combinations
        all_combinations = list(self.transactions.columns)
        total_s = len(self.transactions)

        print(
            f"   Analyzing {len(all_combinations)} combinations = {len(all_combinations) * (len(all_combinations) - 1)} possible relationships"
        )

        # Create comprehensive item pair analysis
        all_pairs_data = []

        for item_a in all_combinations:
            for item_b in all_combinations:
                if item_a != item_b:  # Skip same item relationships
                    # Calculate metrics for this item pair
                    s_with_a = self.transactions[item_a].sum()
                    s_with_b = self.transactions[item_b].sum()
                    s_with_both = (
                        self.transactions[item_a] & self.transactions[item_b]
                    ).sum()

                    # Calculate ARM metrics
                    if s_with_a > 0:
                        support = s_with_both / total_s
                        confidence = s_with_both / s_with_a
                        support_b = s_with_b / total_s
                        lift = confidence / support_b if support_b > 0 else 0

                        # Calculate additional metrics
                        leverage = support - (s_with_a / total_s) * (
                            s_with_b / total_s
                        )
                        conviction = (
                            (1 - support_b) / (1 - confidence)
                            if confidence < 1
                            else float("inf")
                        )

                        # Only include relationships with meaningful support
                        if support > 0:  # Include all non-zero relationships
                            all_pairs_data.append(
                                {
                                    "item_A": item_a,
                                    "item_B": item_b,
                                    "sum_A": int(s_with_a),
                                    "sum_B": int(s_with_b),
                                    "sum_Both": int(s_with_both),
                                    "Support": round(support, 4),
                                    "Confidence": round(confidence, 4),
                                    "Confidence_%": round(confidence * 100, 1),
                                    "Lift": round(lift, 3),
                                    "Leverage": round(leverage, 4),
                                    "Conviction": round(conviction, 3)
                                    if conviction != float("inf")
                                    else 999,
                                    "Rule_Strength": self._categorize_rule_strength(
                                        confidence, lift
                                    ),
                                    "Clinical_Priority": self._assign_priority(
                                        support, confidence, lift
                                    ),
                                    "Relationship_Type": self._classify_relationship_type(
                                        confidence, lift
                                    ),
                                }
                            )

        all_pairs_df = pd.DataFrame(all_pairs_data)

        if len(all_pairs_df) == 0:
            print("❌ No item relationships found")
            return None

        # Sort by confidence, then lift
        all_pairs_df = all_pairs_df.sort_values(
            ["Confidence", "Lift", "Support"], ascending=[False, False, False]
        ).reset_index(drop=True)

        print(f"✅ Calculated {len(all_pairs_df)} item-to-item relationships")

        if export_to_excel:
            self._export_all_pairs_analysis(all_pairs_df, filename)

        return all_pairs_df

    def _classify_relationship_type(self, confidence, lift):
        """Classify the type of item relationship"""
        if confidence >= 0.8:
            return "Highly Predictive"
        elif confidence >= 0.6:
            return "Moderately Predictive"
        elif confidence >= 0.4:
            return "Weakly Predictive"
        else:
            return "Low Predictive Value"

    def _export_all_pairs_analysis(self, all_pairs_df, filename):
        """Export comprehensive all-pairs analysis to Excel with parquet support"""
        try:
            # Rule 2: Always save parquet files
            parquet_filename = filename.replace(".xlsx", ".parquet")
            all_pairs_df.to_parquet(parquet_filename, index=False)
            print(f"✅ Saved parquet file: {parquet_filename}")

            # Rule 1: Check if main data exceeds 1 million rows
            excel_row_limit = 1000000
            main_data_oversized = len(all_pairs_df) > excel_row_limit

            with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                # Handle main data sheet based on size
                if main_data_oversized:
                    # Create a note sheet instead of the full data
                    note_df = pd.DataFrame(
                        {
                            "NOTICE": [
                                "Data exceeds Excel row limit (1M+ rows)",
                                f"Total rows: {len(all_pairs_df):,}",
                                f"Complete data available in: {parquet_filename}",
                                "Use pandas.read_parquet() to load the full dataset",
                                "",
                                "Sample of first 100 rows shown below:",
                            ]
                        }
                    )

                    # Write the notice
                    note_df.to_excel(
                        writer, sheet_name="All_item_Pairs", index=False, startrow=0
                    )

                    # Add sample data below the notice
                    sample_data = all_pairs_df.head(100)
                    sample_data.to_excel(
                        writer, sheet_name="All_item_Pairs", index=False, startrow=8
                    )

                    print(
                        f"⚠️  Main data ({len(all_pairs_df):,} rows) exceeds Excel limit"
                    )
                    print("   📝 Added notice in Excel sheet pointing to parquet file")
                else:
                    # Normal case - write all data to Excel
                    all_pairs_df.to_excel(
                        writer, sheet_name="All_item_Pairs", index=False
                    )

                # Create filtered sheets (these are typically much smaller)
                high_conf = all_pairs_df[all_pairs_df["Confidence"] >= 0.6]
                if len(high_conf) <= excel_row_limit:
                    high_conf.to_excel(
                        writer, sheet_name="High_Confidence_60+", index=False
                    )
                else:
                    # Save filtered data to separate parquet
                    high_conf_parquet = filename.replace(
                        ".xlsx", "_high_confidence.parquet"
                    )
                    high_conf.to_parquet(high_conf_parquet, index=False)

                    note_df = pd.DataFrame(
                        {
                            "NOTICE": [
                                "High confidence data exceeds Excel row limit",
                                f"Total rows: {len(high_conf):,}",
                                f"Complete data available in: {high_conf_parquet}",
                            ]
                        }
                    )
                    note_df.to_excel(
                        writer, sheet_name="High_Confidence_60+", index=False
                    )

                # Strong relationships
                strong_lift = all_pairs_df[all_pairs_df["Lift"] >= 2.0]
                if len(strong_lift) <= excel_row_limit:
                    strong_lift.to_excel(
                        writer, sheet_name="Strong_Association_Lift2+", index=False
                    )
                else:
                    strong_lift_parquet = filename.replace(
                        ".xlsx", "_strong_lift.parquet"
                    )
                    strong_lift.to_parquet(strong_lift_parquet, index=False)

                    note_df = pd.DataFrame(
                        {
                            "NOTICE": [
                                "Strong lift data exceeds Excel row limit",
                                f"Total rows: {len(strong_lift):,}",
                                f"Complete data available in: {strong_lift_parquet}",
                            ]
                        }
                    )
                    note_df.to_excel(
                        writer, sheet_name="Strong_Association_Lift2+", index=False
                    )

                # Clinical priority relationships
                high_priority = all_pairs_df[
                    all_pairs_df["Clinical_Priority"] == "High"
                ]
                high_priority.to_excel(
                    writer, sheet_name="High_Clinical_Priority", index=False
                )

                # Create pivot tables for easy analysis (limit to reasonable size)
                # Sample data for pivot if too large
                pivot_sample = (
                    all_pairs_df.head(10000)
                    if len(all_pairs_df) > 10000
                    else all_pairs_df
                )

                confidence_pivot = pivot_sample.pivot_table(
                    index="item_A", columns="item_B", values="Confidence", fill_value=0
                )
                confidence_pivot.to_excel(writer, sheet_name="Confidence_Matrix")

                lift_pivot = pivot_sample.pivot_table(
                    index="item_A", columns="item_B", values="Lift", fill_value=1
                )
                lift_pivot.to_excel(writer, sheet_name="Lift_Matrix")

                # Summary by item
                item_summary = self._create_item_summary(all_pairs_df)
                item_summary.to_excel(writer, sheet_name="item_Summary", index=False)

                # Top relationships for each item
                top_relationships = self._create_top_relationships_per_item(
                    all_pairs_df
                )
                top_relationships.to_excel(
                    writer, sheet_name="Top_Relations_Per_item", index=False
                )

                # Save individual sheet parquets
                high_conf.to_parquet(
                    filename.replace(".xlsx", "_high_confidence.parquet"), index=False
                )
                strong_lift.to_parquet(
                    filename.replace(".xlsx", "_strong_lift.parquet"), index=False
                )
                high_priority.to_parquet(
                    filename.replace(".xlsx", "_high_priority.parquet"), index=False
                )
                item_summary.to_parquet(
                    filename.replace(".xlsx", "_item_summary.parquet"), index=False
                )
                top_relationships.to_parquet(
                    filename.replace(".xlsx", "_top_relationships.parquet"), index=False
                )

            print("✅ Exported comprehensive item pairs analysis to: {filename}")
            print("   📊 Excel sheets created:")
            if main_data_oversized:
                print("      • All_item_Pairs: Notice + sample (full data in parquet)")
            else:
                print("      • All_item_Pairs: Complete relationship matrix")
            print("      • High_Confidence_60+: Relationships with 60%+ confidence")
            print("      • Strong_Association_Lift2+: Relationships with lift ≥ 2.0")
            print(
                "      • High_Clinical_Priority: Most clinically relevant relationships"
            )
            print("      • Confidence_Matrix: Pivot table of confidence scores")
            print("      • Lift_Matrix: Pivot table of lift scores")
            print("      • item_Summary: Summary statistics per item")
            print("      • Top_Relations_Per_item: Best relationships for each item")

            print("\n   📁 Parquet files created:")
            print("      • {parquet_filename}: Complete dataset")
            print(
                "      • {filename.replace('.xlsx', '_high_confidence.parquet')}: High confidence relationships"
            )
            print(
                "      • {filename.replace('.xlsx', '_strong_lift.parquet')}: Strong lift relationships"
            )
            print(
                "      • {filename.replace('.xlsx', '_high_priority.parquet')}: High priority relationships"
            )
            print(
                "      • {filename.replace('.xlsx', '_item_summary.parquet')}: item summary"
            )
            print(
                "      • {filename.replace('.xlsx', '_top_relationships.parquet')}: Top relationships per item"
            )

        except Exception as e:
            print(f"❌ Error exporting to Excel: {e}")
            # Fallback to CSV and parquet
            all_pairs_df.to_csv(filename.replace(".xlsx", ".csv"), index=False)
            all_pairs_df.to_parquet(filename.replace(".xlsx", ".parquet"), index=False)
            print(
                f"✅ Saved as CSV and Parquet: {filename.replace('.xlsx', '.csv/.parquet')}"
            )

    def _create_item_summary(self, all_pairs_df):
        """Create summary statistics for each item"""
        summary_data = []

        for item in all_pairs_df["item_A"].unique():
            item_data = all_pairs_df[all_pairs_df["item_A"] == item]

            # Calculate summary metrics
            avg_confidence = item_data["Confidence"].mean()
            max_confidence = item_data["Confidence"].max()
            avg_lift = item_data["Lift"].mean()
            max_lift = item_data["Lift"].max()
            total_s = item_data["s_A"].iloc[
                0
            ]  # Same for all rows of this item
            strong_relationships = len(
                item_data[item_data["Rule_Strength"].isin(["Very Strong", "Strong"])]
            )

            # Find best companion item
            best_companion = item_data.loc[item_data["Confidence"].idxmax(), "item_B"]
            best_confidence = max_confidence

            summary_data.append(
                {
                    "item": item,
                    "Total_s_Taking": total_s,
                    "Avg_Confidence": round(avg_confidence, 3),
                    "Max_Confidence": round(max_confidence, 3),
                    "Avg_Lift": round(avg_lift, 3),
                    "Max_Lift": round(max_lift, 3),
                    "Strong_Relationships_Count": strong_relationships,
                    "Best_Companion_item": best_companion,
                    "Best_Companion_Confidence_%": round(best_confidence * 100, 1),
                }
            )

        return pd.DataFrame(summary_data).sort_values("Max_Confidence", ascending=False)

    def _create_top_relationships_per_item(self, all_pairs_df, top_n=3):
        """Create top N relationships for each item"""
        top_relations = []

        for item in all_pairs_df["item_A"].unique():
            item_data = all_pairs_df[all_pairs_df["item_A"] == item].head(top_n)

            for idx, row in item_data.iterrows():
                top_relations.append(
                    {
                        "Primary_item": row["item_A"],
                        "Companion_item": row["item_B"],
                        "Rank": len(top_relations)
                        - len([x for x in top_relations if x["Primary_item"] == item])
                        + 1,
                        "Confidence_%": row["Confidence_%"],
                        "Lift": row["Lift"],
                        "s_Both": row["s_Both"],
                        "Rule_Strength": row["Rule_Strength"],
                    }
                )

        return pd.DataFrame(top_relations)


# Function to query specific item relationships
def query_item_relationship(analyzer, item_a, item_b):
    """Query the likelihood of item_b given item_a"""
    if analyzer.rules is None:
        print("❌ No rules available")
        return None

    # Find rules where item_a is antecedent and item_b is consequent
    specific_rules = analyzer.rules[
        (analyzer.rules["antecedents"].apply(lambda x: item_a in x))
        & (analyzer.rules["consequents"].apply(lambda x: item_b in x))
    ]

    if len(specific_rules) > 0:
        rule = specific_rules.iloc[0]
        print(f"\n🔍 Relationship: {item_a} → {item_b}")
        print(
            f"   Confidence: {rule['confidence']:.3f} ({rule['confidence'] * 100:.1f}%)"
        )
        print(f"   Support: {rule['support']:.3f}")
        print(f"   Lift: {rule['lift']:.3f}")
        return rule
    else:
        print(f"\n❌ No significant association found between {item_a} and {item_b}")
        return None


def load_parquet_data(parquet_filename):
    """Helper function to load and examine parquet data"""
    try:
        df = pd.read_parquet(parquet_filename)
        print(f"✅ Loaded parquet file: {parquet_filename}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"❌ Error loading parquet file: {e}")
        return None


def compare_file_sizes(base_filename):
    """Compare file sizes between Excel and Parquet formats"""
    import os

    excel_file = base_filename + ".xlsx"
    parquet_file = base_filename + ".parquet"
    csv_file = base_filename + ".csv"

    print("\n📊 File Size Comparison:")
    print("=" * 40)

    for file_path, file_type in [
        (excel_file, "Excel"),
        (parquet_file, "Parquet"),
        (csv_file, "CSV"),
    ]:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"{file_type:>8}: {size_mb:.2f} MB")
        else:
            print(f"{file_type:>8}: File not found")


# %% Extended example usage for comprehensive table

if __name__ == "__main__":
    ## Automatically analyze ALL item pairs with enhanced export
    print("\n🚀 Creating Complete item Relationships Matrix:")
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
        export_to_excel=True, filename="complete_item_relationships_matrix.xlsx"
    )

    if all_pairs_matrix is not None:
        print("\n📈 Complete Analysis Summary:")
        print(f"   Total item pairs analyzed: {len(all_pairs_matrix):,}")
        print(
            f"   High confidence relationships (60%+): {len(all_pairs_matrix[all_pairs_matrix['Confidence'] >= 0.6]):,}"
        )
        print(
            f"   Strong associations (Lift 2.0+): {len(all_pairs_matrix[all_pairs_matrix['Lift'] >= 2.0]):,}"
        )
        print(
            f"   High priority relationships: {len(all_pairs_matrix[all_pairs_matrix['Clinical_Priority'] == 'High']):,}"
        )

        # Compare file sizes for the complete matrix
        compare_file_sizes("complete_item_relationships_matrix")

        print("\n📁 Output Files Generated:")
        print("   📊 Excel File: complete_item_relationships_matrix.xlsx")
        print("      • Multiple sheets with filtered views")
        print("      • Pivot tables for easy analysis")
        print("      • Summary statistics")
        print("   📦 Parquet Files: Multiple .parquet files for different views")
        print("      • Faster loading for large datasets")
        print("      • Better compression than CSV")
        print("      • Preserves data types")

    # Demonstration of loading parquet data
    print("\n🔄 Demonstration: Loading Parquet Data")
    sample_parquet = load_parquet_data("complete_item_relationships_matrix.parquet")
    if sample_parquet is not None:
        print("   Sample of loaded data:")
        print(
            f"   {sample_parquet.head(3)[['item_A', 'item_B', 'Confidence_%', 'Lift', 'Rule_Strength']].to_string(index=False)}"
        )

    print("\n" + "=" * 60)
    print("🎉 Enhanced Analysis Complete with Dual Export!")
    print("📋 Key Features:")
    print("   ✅ Excel export with multiple sheets")
    print("   ✅ Parquet export for large datasets")
    print("   ✅ Automatic handling of Excel row limits")
    print("   ✅ File size comparisons")
    print("   ✅ Easy data loading utilities")
    print("=" * 60)

    """
    Excel output:
        All_item_Pairs: Complete relationship matrix
        High_Confidence_60+: Only relationships with 60%+ confidence
        Strong_Association_Lift2+: Relationships with lift ≥ 2.0
        Confidence_Matrix: Easy lookup of item A → item B confidence scores.
        Lift_Matrix: Pivot table showing lift scores
        item_Summary: Best companion combinations for each item. Shows which item has the most predictable companions
        Top_Relations_Per_item: Top 3 relationships per item
    """

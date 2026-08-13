"""
extractor.py
-------------
Core extraction engine. Wraps `pbixray` to pull a complete, structured
snapshot of a Power BI (.pbix) semantic model: tables, columns/schema,
relationships, DAX measures, calculated columns, Power Query (M) source,
and sample data rows per table.

All extraction is defensive (try/except per feature) because different
pbix files expose different subsets of metadata depending on how they
were built (Import vs DirectQuery, Composite models, older pbix versions).
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional

import pandas as pd
from pbixray import PBIXRay


def _df_to_records(df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    """Safely convert a pandas DataFrame to a list of JSON-safe dicts."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    clean = df.copy()
    # Replace NaN/NaT/inf with None so JSON serialization never breaks
    clean = clean.replace([float("inf"), float("-inf")], None)
    records = clean.where(pd.notnull(clean), None).to_dict(orient="records")

    def _sanitize(v):
        if isinstance(v, float) and math.isnan(v):
            return None
        return v

    return [{k: _sanitize(v) for k, v in row.items()} for row in records]


class PBIXExtractor:
    """Extracts a full, structured snapshot of a .pbix model."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.model = PBIXRay(file_path)

    # ---------- Basic building blocks ----------

    def get_tables(self) -> List[str]:
        try:
            return list(self.model.tables)
        except Exception:
            return []

    def get_schema(self) -> List[Dict[str, Any]]:
        """All tables + columns + data types."""
        try:
            return _df_to_records(self.model.schema)
        except Exception:
            return []

    def get_schema_grouped(self) -> Dict[str, List[Dict[str, Any]]]:
        """Schema grouped by table name -> [ {column, type}, ... ]."""
        rows = self.get_schema()
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            table = row.get("TableName") or row.get("table_name") or "Unknown"
            grouped.setdefault(table, []).append(row)
        return grouped

    def get_relationships(self) -> List[Dict[str, Any]]:
        try:
            return _df_to_records(self.model.relationships)
        except Exception:
            return []

    def get_measures(self) -> List[Dict[str, Any]]:
        try:
            return _df_to_records(self.model.dax_measures)
        except Exception:
            return []

    def get_calculated_columns(self) -> List[Dict[str, Any]]:
        try:
            return _df_to_records(self.model.dax_columns)
        except Exception:
            return []

    def get_calculated_tables(self) -> List[Dict[str, Any]]:
        try:
            return _df_to_records(self.model.dax_tables)
        except Exception:
            return []

    def get_power_query(self) -> List[Dict[str, Any]]:
        """M code / source queries behind each table (Power Query)."""
        try:
            return _df_to_records(self.model.power_query)
        except Exception:
            return []

    def get_metadata(self) -> Dict[str, Any]:
        try:
            meta = self.model.metadata
            if isinstance(meta, pd.DataFrame):
                return _df_to_records(meta)
            return dict(meta) if meta else {}
        except Exception:
            return {}

    def get_kpis(self) -> List[Dict[str, Any]]:
        try:
            return _df_to_records(self.model.tmschema_kpis)
        except Exception:
            return []

    def get_hierarchies(self) -> List[Dict[str, Any]]:
        try:
            return _df_to_records(self.model.tmschema_hierarchies)
        except Exception:
            return []

    def get_perspectives(self) -> List[Dict[str, Any]]:
        try:
            return _df_to_records(self.model.perspectives)
        except Exception:
            return []

    def get_roles_rls(self) -> List[Dict[str, Any]]:
        """Row-Level Security roles/filters, if defined."""
        try:
            return _df_to_records(self.model.rls)
        except Exception:
            return []

    def get_statistics(self) -> List[Dict[str, Any]]:
        """Size/row-count statistics per table/column, useful for perf tuning."""
        try:
            return _df_to_records(self.model.statistics)
        except Exception:
            return []

    # ---------- Sample data ----------

    def get_sample_data(self, table_name: str, rows: int = 10) -> List[Dict[str, Any]]:
        try:
            df = self.model.get_table(table_name)
            if df is None:
                return []
            return _df_to_records(df.head(rows))
        except Exception:
            return []

    def get_all_sample_data(self, rows: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        samples = {}
        for t in self.get_tables():
            samples[t] = self.get_sample_data(t, rows=rows)
        return samples

    # ---------- Full combined extract ----------

    def full_extract(self, sample_rows: int = 5) -> Dict[str, Any]:
        """Everything in one structured payload — the main endpoint for the agent."""
        return {
            "file": self.file_path,
            "metadata": self.get_metadata(),
            "tables": self.get_tables(),
            "schema_by_table": self.get_schema_grouped(),
            "relationships": self.get_relationships(),
            "measures": self.get_measures(),
            "calculated_columns": self.get_calculated_columns(),
            "calculated_tables": self.get_calculated_tables(),
            "hierarchies": self.get_hierarchies(),
            "kpis": self.get_kpis(),
            "perspectives": self.get_perspectives(),
            "row_level_security": self.get_roles_rls(),
            "power_query_m_code": self.get_power_query(),
            "statistics": self.get_statistics(),
            "sample_data": self.get_all_sample_data(rows=sample_rows),
        }

    def full_extract_markdown(self, sample_rows: int = 5) -> str:
        """
        A human/LLM-friendly markdown rendering of the full model —
        ideal to hand directly to an agent as grounding context.
        """
        data = self.full_extract(sample_rows=sample_rows)
        lines: List[str] = []
        lines.append(f"# Power BI Model Extract: {data['file']}\n")

        lines.append("## Tables & Columns\n")
        for table, cols in data["schema_by_table"].items():
            lines.append(f"### {table}")
            for c in cols:
                col_name = c.get("ColumnName") or c.get("column_name") or "?"
                col_type = c.get("PandasDataType") or c.get("data_type") or "?"
                lines.append(f"- {col_name} ({col_type})")
            lines.append("")

        if data["relationships"]:
            lines.append("## Relationships\n")
            for r in data["relationships"]:
                lines.append(f"- {r}")
            lines.append("")

        if data["measures"]:
            lines.append("## Measures\n")
            for m in data["measures"]:
                name = m.get("Name", "Unnamed")
                table = m.get("TableName", "")
                expr = m.get("Expression", "")
                lines.append(f"**{table}[{name}]**")
                lines.append("```DAX")
                lines.append(str(expr).strip())
                lines.append("```")
            lines.append("")

        if data["calculated_columns"]:
            lines.append("## Calculated Columns\n")
            for c in data["calculated_columns"]:
                lines.append(f"- {c.get('TableName','')}[{c.get('ColumnName','')}] = `{c.get('Expression','')}`")
            lines.append("")

        if data["sample_data"]:
            lines.append("## Sample Data\n")
            for table, rows in data["sample_data"].items():
                if not rows:
                    continue
                lines.append(f"### {table} (first {len(rows)} rows)")
                lines.append("```")
                lines.append(str(rows))
                lines.append("```")
            lines.append("")

        return "\n".join(lines)

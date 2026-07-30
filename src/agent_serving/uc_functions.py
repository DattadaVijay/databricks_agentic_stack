# Databricks notebook source
# COMMAND ----------
%sql
CREATE OR REPLACE FUNCTION main.default.get_employee_leave_balance(
  employee_id STRING COMMENT 'Employee ID e.g. EMP001'
)
RETURNS STRUCT<name: STRING, remaining_days: INT>
LANGUAGE PYTHON
COMMENT 'Returns leave balance for one employee. Use for individual leave queries.'
AS $$
  data = {
    "EMP001": {"name": "Vijay Kumar",  "remaining_days": 12},
    "EMP002": {"name": "Suma Reddy",   "remaining_days": 5},
    "EMP003": {"name": "Alex Thomson", "remaining_days": 8},
  }
  emp = data.get(employee_id, {"name": "Unknown", "remaining_days": 0})
  return emp
$$;


# COMMAND ----------
%sql
CREATE OR REPLACE FUNCTION main.default.get_department_headcount(
  department STRING COMMENT 'Department name e.g. Engineering'
)
RETURNS STRUCT<headcount: INT, open_roles: INT>
LANGUAGE PYTHON
COMMENT 'Returns headcount and open roles for a department.'
AS $$
  data = {
    "Engineering": {"headcount": 42, "open_roles": 5},
    "HR":          {"headcount": 8,  "open_roles": 1},
    "Finance":     {"headcount": 12, "open_roles": 0},
  }
  return data.get(department, {"headcount": 0, "open_roles": 0})
$$;
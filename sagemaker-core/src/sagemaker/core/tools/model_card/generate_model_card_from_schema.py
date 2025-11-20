#!/usr/bin/env python3
"""
Script to generate Pydantic classes from JSON schema
"""

import json
from typing import Dict, Any, Set


def to_snake_case(name: str) -> str:
    """Convert camelCase or PascalCase to snake_case"""
    import re

    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def to_pascal_case(name: str) -> str:
    """Convert snake_case to PascalCase"""
    return "".join(word.capitalize() for word in name.split("_"))


def get_python_type(json_type: str, format_info: Dict[str, Any] = None) -> str:
    """Convert JSON schema type to Python type"""
    type_mapping = {
        "string": "str",
        "number": "float",
        "integer": "int",
        "boolean": "bool",
        "array": "List",
        "object": "Dict",
    }
    return type_mapping.get(json_type, "Any")


def generate_enum_class(name: str, enum_values: list) -> str:
    """Generate enum class from schema enum"""
    class_name = to_pascal_case(name)
    lines = [f"class {class_name}(str, Enum):"]
    for value in enum_values:
        enum_name = value.upper().replace(" ", "_").replace("-", "_")
        lines.append(f'    {enum_name} = "{value}"')
    return "\n".join(lines) + "\n\n"


def find_nested_class_name(prop_name: str, known_classes: Set[str]) -> str:
    """Find the correct nested class name for a property"""
    # Common mappings for nested classes
    nested_mappings = {
        "training_job_details": "TrainingJobDetails",
        "training_environment": "TrainingEnvironment",
        "model_overview": "ModelOverview",
        "intended_uses": "IntendedUses",
        "business_details": "BusinessDetails",
        "training_details": "TrainingDetails",
        "additional_information": "AdditionalInformation",
    }

    return nested_mappings.get(prop_name, "")


def generate_field_definition(
    prop_name: str,
    prop_schema: Dict[str, Any],
    required: bool = False,
    additional_classes=None,
    known_classes: Set[str] = None,
) -> str:
    """Generate Pydantic field definition"""
    if additional_classes is None:
        additional_classes = []
    if known_classes is None:
        known_classes = set()

    field_type = prop_schema.get("type")

    # Handle $ref
    if "$ref" in prop_schema:
        ref_name = prop_schema["$ref"].split("/")[-1]
        class_name = to_pascal_case(ref_name)
        # Map specific missing types
        type_mapping = {
            "axis_name_string": "str",
            "axis_name_array": "List[str]",
            "custom_property": "str",
        }

        # Special handling for function enum
        if ref_name == "function" and prop_name == "function":
            field_type_str = "Function"
        elif ref_name in type_mapping:
            field_type_str = type_mapping[ref_name]
        elif class_name not in known_classes:
            # Create a new class for undefined references
            additional_classes.append(
                (class_name, {"type": "object", "additionalProperties": True})
            )
            field_type_str = class_name
        else:
            field_type_str = class_name

        if not required:
            field_type_str = f"Optional[{field_type_str}]"
        return f"{prop_name}: {field_type_str} = None"

    # Handle enum
    if "enum" in prop_schema:
        # For single-value enums or specific cases, use Literal
        if len(prop_schema["enum"]) == 1:
            enum_value = prop_schema["enum"][0]
            field_type_str = (
                f'Literal["{enum_value}"]' if required else f'Optional[Literal["{enum_value}"]]'
            )
        elif prop_name == "type" and "enum" in prop_schema:
            # For type fields with enums, use Literal with Union
            enum_values = prop_schema["enum"]
            literal_values = ", ".join([f'"{val}"' for val in enum_values])
            field_type_str = (
                f"Literal[{literal_values}]" if required else f"Optional[Literal[{literal_values}]]"
            )
        else:
            enum_class = to_pascal_case(prop_name)
            field_type_str = enum_class if required else f"Optional[{enum_class}]"
        return f"{prop_name}: {field_type_str} = None"

    # Handle array
    if field_type == "array":
        items = prop_schema.get("items", {})
        if "type" in items and items["type"] != "object":
            item_type = get_python_type(items["type"])
            field_type_str = f"List[{item_type}]"
        elif not items:
            # Empty items - create a generic item class
            item_class_name = to_pascal_case(f"{prop_name}_item")
            additional_classes.append(
                (item_class_name, {"type": "object", "additionalProperties": True})
            )
            field_type_str = f"List[{item_class_name}]"
        elif "$ref" in items:
            ref_name = items["$ref"].split("/")[-1]
            item_type = to_pascal_case(ref_name)
            field_type_str = f"List[{item_type}]"
        elif "anyOf" in items:
            # Handle union types in arrays
            union_types = []
            for any_of_item in items["anyOf"]:
                if "$ref" in any_of_item:
                    ref_name = any_of_item["$ref"].split("/")[-1]
                    union_types.append(to_pascal_case(ref_name))
            field_type_str = f"List[Union[{', '.join(union_types)}]]"
        elif items.get("type") == "object":
            # Create a class for array items
            item_class_name = to_pascal_case(f"{prop_name}_item")
            additional_classes.append((item_class_name, items))
            field_type_str = f"List[{item_class_name}]"
        else:
            # Create a generic item class
            item_class_name = to_pascal_case(f"{prop_name}_item")
            additional_classes.append(
                (item_class_name, {"type": "object", "additionalProperties": True})
            )
            field_type_str = f"List[{item_class_name}]"

        if not required:
            field_type_str = f"Optional[{field_type_str}]"

    # Handle object
    elif field_type == "object":
        if "additionalProperties" in prop_schema:
            add_props = prop_schema["additionalProperties"]
            if isinstance(add_props, dict) and "$ref" in add_props:
                ref_name = add_props["$ref"].split("/")[-1]
                field_type_str = f"Dict[str, {to_pascal_case(ref_name)}]"
            elif isinstance(add_props, dict) and "type" in add_props:
                value_type = get_python_type(add_props["type"])
                field_type_str = f"Dict[str, {value_type}]"
            elif isinstance(add_props, bool) and add_props:
                # additionalProperties: true - check for existing nested class first
                nested_class_name = find_nested_class_name(prop_name, known_classes)
                if nested_class_name:
                    # Check if this should be a direct reference or Dict based on schema structure
                    if prop_name in [
                        "training_job_details",
                        "training_environment",
                        "model_overview",
                        "intended_uses",
                        "business_details",
                        "training_details",
                        "additional_information",
                    ]:
                        field_type_str = nested_class_name
                    else:
                        field_type_str = f"Dict[str, {nested_class_name}]"
                else:
                    value_class_name = to_pascal_case(f"{prop_name}_value")
                    additional_classes.append(
                        (value_class_name, {"type": "object", "additionalProperties": True})
                    )
                    field_type_str = f"Dict[str, {value_class_name}]"
            else:
                # Create a value class for complex additionalProperties
                nested_class_name = find_nested_class_name(prop_name, known_classes)
                if nested_class_name:
                    # Check if this should be a direct reference or Dict based on schema structure
                    if prop_name in [
                        "training_job_details",
                        "training_environment",
                        "model_overview",
                        "intended_uses",
                        "business_details",
                        "training_details",
                        "additional_information",
                    ]:
                        field_type_str = nested_class_name
                    else:
                        field_type_str = f"Dict[str, {nested_class_name}]"
                else:
                    value_class_name = to_pascal_case(f"{prop_name}_value")
                    additional_classes.append(
                        (
                            value_class_name,
                            (
                                add_props
                                if isinstance(add_props, dict)
                                else {"type": "object", "additionalProperties": True}
                            ),
                        )
                    )
                    field_type_str = f"Dict[str, {value_class_name}]"
        elif "properties" in prop_schema:
            # This should be a separate class - use direct reference
            class_name = to_pascal_case(prop_name)
            field_type_str = class_name
        else:
            # Create a class for generic objects
            class_name = to_pascal_case(prop_name)
            additional_classes.append(
                (class_name, {"type": "object", "additionalProperties": True})
            )
            field_type_str = class_name

        if not required:
            field_type_str = f"Optional[{field_type_str}]"

    # Handle anyOf/oneOf
    elif "anyOf" in prop_schema:
        # Create a union type or a generic class
        union_types = []
        for i, any_of_item in enumerate(prop_schema["anyOf"]):
            if "type" in any_of_item and any_of_item["type"] != "object":
                union_types.append(get_python_type(any_of_item["type"]))
            elif "type" in any_of_item and any_of_item["type"] == "array":
                # Handle array types in anyOf
                items = any_of_item.get("items", {})
                if "type" in items:
                    item_type = get_python_type(items["type"])
                    union_types.append(f"List[{item_type}]")
                else:
                    union_types.append("List")
            else:
                # Create a class for complex anyOf items
                item_class_name = to_pascal_case(f"{prop_name}_variant_{i}")
                additional_classes.append((item_class_name, any_of_item))
                union_types.append(item_class_name)

        if len(union_types) == 1:
            field_type_str = union_types[0]
        else:
            field_type_str = f"Union[{', '.join(union_types)}]"

        if not required:
            field_type_str = f"Optional[{field_type_str}]"

    # Handle fields without explicit type but with nested structure
    elif not field_type and ("function" in prop_schema or "notes" in prop_schema):
        # This is likely the objective_function case with nested structure
        class_name = to_pascal_case(prop_name)
        field_type_str = class_name if required else f"Optional[{class_name}]"

    # Handle basic types
    else:
        python_type = get_python_type(field_type)
        field_type_str = python_type if required else f"Optional[{python_type}]"

    # Add Field constraints
    constraints = []
    if "maxLength" in prop_schema:
        constraints.append(f"max_length={prop_schema['maxLength']}")
    if "minLength" in prop_schema:
        constraints.append(f"min_length={prop_schema['minLength']}")
    if "maxItems" in prop_schema:
        constraints.append(f"max_length={prop_schema['maxItems']}")
    if "minItems" in prop_schema:
        constraints.append(f"min_length={prop_schema['minItems']}")
    if "pattern" in prop_schema:
        constraints.append(f'pattern="{prop_schema["pattern"]}"')
    if field_type == "string" and "enum" in prop_schema and len(prop_schema["enum"]) == 1:
        constraints.append(f"const=True")

    if required and not constraints:
        field_def = f"{prop_name}: {field_type_str}"
    elif required and constraints:
        field_def = f"{prop_name}: {field_type_str} = Field({', '.join(constraints)})"
    elif not required and constraints:
        field_def = f"{prop_name}: {field_type_str} = Field(None, {', '.join(constraints)})"
    else:
        default_val = "None"
        if field_type == "array" and "default" in prop_schema:
            default_val = "Field(default_factory=list)"
        field_def = f"{prop_name}: {field_type_str} = {default_val}"

    return field_def


def generate_class(
    class_name: str, schema: Dict[str, Any], additional_classes=None, known_classes: Set[str] = None
) -> str:
    """Generate Pydantic class from schema"""
    if additional_classes is None:
        additional_classes = []
    if known_classes is None:
        known_classes = set()

    lines = [f"class {class_name}(BaseModel):"]

    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))

    if not properties:
        lines.append("    pass")
        return "\n".join(lines) + "\n\n"

    for prop_name, prop_schema in properties.items():
        is_required = prop_name in required_fields
        field_def = generate_field_definition(
            prop_name, prop_schema, is_required, additional_classes, known_classes
        )
        lines.append(f"    {field_def}")

    return "\n".join(lines) + "\n\n"


def extract_class_dependencies(class_def: str) -> set:
    """Extract class names that this class depends on"""
    import re

    dependencies = set()

    # Find type hints that reference other classes
    patterns = [
        r":\s*([A-Z][a-zA-Z0-9_]*)",  # field: ClassName
        r"Optional\[([A-Z][a-zA-Z0-9_]*)\]",  # Optional[ClassName]
        r"List\[([A-Z][a-zA-Z0-9_]*)\]",  # List[ClassName]
        r"Dict\[str,\s*([A-Z][a-zA-Z0-9_]*)\]",  # Dict[str, ClassName]
        r"Union\[([^\]]+)\]",  # Union[...]
    ]

    for pattern in patterns:
        matches = re.findall(pattern, class_def)
        for match in matches:
            if pattern.endswith(r"Union\[([^\]]+)\]"):
                # Handle Union types
                union_types = [t.strip() for t in match.split(",")]
                for union_type in union_types:
                    if union_type[0].isupper():
                        dependencies.add(union_type)
            elif match[0].isupper():  # Only class names (start with uppercase)
                dependencies.add(match)

    return dependencies


def topological_sort_classes(class_definitions: list) -> list:
    """Sort classes based on their dependencies using topological sort"""
    # Build dependency graph
    dependencies = {}
    class_names = set()

    for class_def in class_definitions:
        # Extract class name
        lines = class_def.strip().split("\n")
        if lines and lines[0].startswith("class "):
            class_name = lines[0].split("(")[0].replace("class ", "").strip()
            class_names.add(class_name)
            dependencies[class_name] = extract_class_dependencies(class_def)

    # Remove dependencies that are not in our class set (external dependencies)
    for class_name in dependencies:
        dependencies[class_name] = dependencies[class_name].intersection(class_names)

    # Topological sort
    sorted_classes = []
    visited = set()
    temp_visited = set()

    def visit(class_name):
        if class_name in temp_visited:
            return  # Circular dependency, skip
        if class_name in visited:
            return

        temp_visited.add(class_name)
        for dep in dependencies.get(class_name, set()):
            visit(dep)
        temp_visited.remove(class_name)
        visited.add(class_name)

        # Find the class definition
        for class_def in class_definitions:
            if class_def.strip().startswith(f"class {class_name}("):
                sorted_classes.append(class_def)
                break

    for class_name in class_names:
        visit(class_name)

    return sorted_classes


def generate_pydantic_from_schema(schema_path: str, output_path: str):
    """Main function to generate Pydantic classes from JSON schema"""

    with open(schema_path, "r") as f:
        schema = json.load(f)

    output_lines = [
        "from typing import List, Optional, Dict, Union, Literal",
        "from pydantic import BaseModel, Field",
        "from enum import Enum",
        "",
        "",
    ]

    additional_classes = []
    known_classes = set()
    all_class_definitions = []

    # Collect all known class names first
    definitions = schema.get("definitions", {})
    for def_name, def_schema in definitions.items():
        class_name = to_pascal_case(def_name)
        known_classes.add(class_name)
        # Also add the original snake_case name for reference matching
        known_classes.add(def_name)

    properties = schema.get("properties", {})
    for prop_name, prop_schema in properties.items():
        if prop_schema.get("type") == "object":
            known_classes.add(to_pascal_case(prop_name))

    # Generate enums from definitions
    for def_name, def_schema in definitions.items():
        if "enum" in def_schema:
            enum_class = generate_enum_class(def_name, def_schema["enum"])
            output_lines.append(enum_class)

    # No type aliases needed - use direct types

    # Generate Function enum based on objective_function definition
    if "objective_function" in definitions:
        obj_func_def = definitions["objective_function"]
        if "properties" in obj_func_def and "function" in obj_func_def["properties"]:
            func_prop = obj_func_def["properties"]["function"]
            if "enum" in func_prop:
                enum_class = generate_enum_class("function", func_prop["enum"])
                output_lines.append(enum_class)

    # Generate classes from definitions (including those without explicit type)
    for def_name, def_schema in definitions.items():
        if "enum" not in def_schema and (
            def_schema.get("type") == "object" or "properties" in def_schema
        ):
            class_name = to_pascal_case(def_name)
            class_def = generate_class(class_name, def_schema, additional_classes, known_classes)
            all_class_definitions.append(class_def)

    # Generate classes from main properties
    for prop_name, prop_schema in properties.items():
        if prop_schema.get("type") == "object":
            class_name = to_pascal_case(prop_name)
            class_def = generate_class(class_name, prop_schema, additional_classes, known_classes)
            all_class_definitions.append(class_def)

    # Generate nested object classes
    def find_nested_objects(obj, parent_name=""):
        nested_classes = []
        if isinstance(obj, dict):
            properties = obj.get("properties", {})
            for prop_name, prop_schema in properties.items():
                if prop_schema.get("type") == "object" and "properties" in prop_schema:
                    class_name = to_pascal_case(prop_name)
                    class_def = generate_class(
                        class_name, prop_schema, additional_classes, known_classes
                    )
                    nested_classes.append(class_def)
                    nested_classes.extend(find_nested_objects(prop_schema, prop_name))
                elif prop_schema.get("type") == "array":
                    items = prop_schema.get("items", {})
                    if items.get("type") == "object" and "properties" in items:
                        class_name = to_pascal_case(f"{prop_name}_item")
                        class_def = generate_class(
                            class_name, items, additional_classes, known_classes
                        )
                        nested_classes.append(class_def)
        return nested_classes

    nested_classes = find_nested_objects(schema)
    all_class_definitions.extend(nested_classes)

    # Generate additional classes
    for class_name, class_schema in additional_classes:
        if isinstance(class_schema, dict):
            # Handle generic objects with additionalProperties
            if class_schema.get("additionalProperties") is True and not class_schema.get(
                "properties"
            ):
                # Create a simple pass class for generic objects
                all_class_definitions.append(f"class {class_name}(BaseModel):\n    pass\n\n")
            else:
                class_def = generate_class(class_name, class_schema)
                all_class_definitions.append(class_def)

    # Generate main schema class
    main_class_name = schema.get("title", "Schema")
    if main_class_name == "SageMakerModelCardSchema":
        main_class_name = "ModelCardContent"
    elif main_class_name.endswith("Schema"):
        main_class_name = main_class_name
    else:
        main_class_name = f"{main_class_name}Schema"

    main_class = generate_class(main_class_name, schema, additional_classes, known_classes)
    all_class_definitions.append(main_class)

    # Sort classes by dependencies
    sorted_classes = topological_sort_classes(all_class_definitions)
    output_lines.extend(sorted_classes)

    # Remove duplicates while preserving order
    seen_classes = set()
    final_output = []
    for line in output_lines:
        if line.strip().startswith("class "):
            class_name = line.strip().split("(")[0].replace("class ", "").strip()
            if class_name not in seen_classes:
                seen_classes.add(class_name)
                final_output.append(line)
            # Skip duplicate class definitions
        else:
            final_output.append(line)

    # Post-process to replace any remaining type aliases
    final_content = "\n".join(final_output)
    final_content = final_content.replace("CustomProperty", "str")
    final_content = final_content.replace("AxisNameString", "str")
    final_content = final_content.replace("AxisNameArray", "List[str]")

    # Write output
    with open(output_path, "w") as f:
        f.write(final_content)

    print(f"Generated Pydantic classes in {output_path}")
